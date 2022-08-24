from __future__ import annotations
from functools import reduce
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset, Dataset, get_worker_info
from os.path import join
from os import environ
from typing import Optional, Iterator, Iterable, Callable, List, Dict, TypeVar, Set, Union, Generic
from typing_extensions import TypeAlias
from sqlite3 import Connection, Cursor
from operator import add
from multiprocessing import cpu_count

from .db import create_connection
from .booru_db import get_file_ids_from_nth, get_first_n_file_ids, file_ids_to_dtos, get_tag_records, BooruFileId, TagRecord, TagCategory
from .booru_chars_caption_dataset import BooruCharsCaptionsDataset, BooruCharsCaptionsDatasetParams, Example
from argparse import ArgumentParser, Namespace
from pytorch_lightning.utilities.argparse import from_argparse_args
from more_itertools import partition
from util.enumeration_to_value import enumeration_to_value
from contextlib import closing
from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import groupby, chain
from random import shuffle, sample, random

BooruCharsCaptionsDatasetFactory: TypeAlias = Callable[[BooruCharsCaptionsDatasetParams], BooruCharsCaptionsDataset]

@dataclass
class Batch:
  unmasked: LongTensor
  masked: LongTensor

Tokenize: TypeAlias = Callable[[List[str]], List[int]]
PadTokens: TypeAlias = Callable[[List[int], int], List[int]]
GetIntsFromExample: TypeAlias = Callable[[Example], List[int]]
PadTokensKnownLength: TypeAlias = Callable[[List[int]], List[int]]

class BooruCharsCaptions(LightningDataModule):
  batch_size: int
  validation_split_percentage: int
  test_quantity: int
  validation_workers: int
  train_workers: int
  pad_tokens: PadTokens
  dataset_factory: BooruCharsCaptionsDatasetFactory
  sqlite_db_path: str
  train_dataset: Optional[BooruCharsCaptionsDataset] = None
  validation_dataset: Optional[BooruCharsCaptionsDataset] = None
  test_dataset: Optional[BooruCharsCaptionsDataset] = None
  close_fit: Optional[Callable[[], None]] = None
  close_test: Optional[Callable[[], None]] = None

  @staticmethod
  def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("BooruCharsCaptions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--test_quantity', type=int, default=32)
    parser.add_argument('--sqlite_db_path', type=str, default=join(environ['HOME'], 'machine-learning/booru-chars/booru-chars.db'))
    parser.add_argument('--data_max_workers', type=int, default=max(1, cpu_count() - 3))
    return parent_parser

  def __init__(
    self,
    batch_size: int,
    validation_split_percentage: int,
    test_quantity: int,
    data_max_workers: int,
    sqlite_db_path: str,
    pad_tokens: PadTokens,
    dataset_factory: BooruCharsCaptionsDatasetFactory,
  ) -> None:
    super().__init__()
    self.pad_tokens = pad_tokens
    self.dataset_factory = dataset_factory
    self.batch_size = batch_size
    self.validation_split_percentage = validation_split_percentage
    self.test_quantity = test_quantity
    self.validation_workers = max(1, data_max_workers//2)
    self.train_workers = max(1, data_max_workers-self.validation_workers)
    self.sqlite_db_path = sqlite_db_path
  
  @staticmethod
  def _close_handles(cur: Cursor, conn: Connection) -> None:
    cur.close()
    conn.close()
  
  @classmethod
  def _get_handle_closer(cls: BooruCharsCaptions, cur: Cursor, conn: Connection) -> Callable[[], None]:
    return lambda: cls._close_handles(cur, conn)
  
  @staticmethod
  def _value_from_enumeration(cur: Cursor, conn: Connection) -> None:
    cur.close()
    conn.close()

  def setup(self, stage: Optional[str] = None) -> None:
    # Assign train/val datasets for use in dataloaders
    if stage == "fit" or stage is None:
      if callable(self.close_fit):
        self.close_fit()
      conn: Connection = create_connection(self.sqlite_db_path)
      cur: Cursor = conn.cursor()
      self.close_fit = self._get_handle_closer(cur, conn)
      file_ids: Cursor = get_file_ids_from_nth(cur, self.test_quantity)
      file_id_dtos: Iterator[BooruFileId] = file_ids_to_dtos(file_ids)
      retain = lambda enumeration: enumeration[0] % 100 < self.validation_split_percentage
      validation, training = partition(retain, enumerate(file_id_dtos))
      get_cursor = lambda: conn.cursor()

      self.train_dataset = self.dataset_factory(
        BooruCharsCaptionsDatasetParams(
          file_ids=map(enumeration_to_value, training),
          get_cursor=get_cursor
        )
      )
      self.validation_dataset = self.dataset_factory(
        BooruCharsCaptionsDatasetParams(
          file_ids=map(enumeration_to_value, validation),
          get_cursor=get_cursor
        )
      )

    # Assign test dataset for use in dataloader(s)
    if stage == "test" or stage is None:
      if callable(self.close_test):
        self.close_test()
      conn: Connection = create_connection(self.sqlite_db_path)
      cur: Cursor = conn.cursor()
      self.close_test = self._get_handle_closer(cur, conn)
      get_first_n_file_ids(cur, self.test_quantity)
      file_id_dtos: Iterator[BooruFileId] = file_ids_to_dtos(file_ids)
      get_cursor = lambda: conn.cursor()
      self.test_dataset = self.dataset_factory(
        BooruCharsCaptionsDatasetParams(
          file_ids=file_id_dtos,
          get_cursor=get_cursor
        )
      )
  
  def teardown(self, stage: Optional[str] = None) -> None:
    if stage == "fit" or stage is None:
      if callable(self.close_fit):
        self.close_fit()
      self.train_dataset = None
      self.validation_dataset = None
    
    if stage == "test" or stage is None:
      if callable(self.close_test):
        self.close_test()
      self.test_dataset = None
  
  @staticmethod
  def _pad_example_series(pad_tokens: PadTokensKnownLength, get_ints: GetIntsFromExample, examples: List[Example]) -> List[List[int]]:
    # TODO: should we be padding with -100 token?
    return [pad_tokens(tokenized) for tokenized in (get_ints(example) for example in examples)]
  
  def collate_fn(self, examples: List[Example]) -> Batch:
    pad_length: int = max(map(lambda example: len(example.unmasked), examples))
    pad_tokens: PadTokensKnownLength = lambda tokenized: self.pad_tokens(tokenized, pad_length)
    padded_maskeds: List[List[int]] = self._pad_example_series(
      pad_tokens=pad_tokens,
      get_ints=lambda example: example.masked,
      examples=examples,
      )
    padded_unmaskeds: List[List[int]] = self._pad_example_series(
      pad_tokens=pad_tokens,
      get_ints=lambda example: example.unmasked,
      examples=examples,
      )
    batch = Batch(
      masked=LongTensor(padded_maskeds),
      unmasked=LongTensor(padded_unmaskeds),
    )
    return batch
  
  def _generic_dataloader(self, dataset: Dataset, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)#, num_workers=num_workers)

  def train_dataloader(self) -> DataLoader:
    assert self.train_dataset is not None
    return self._generic_dataloader(self.train_dataset, num_workers=self.train_workers)

  def val_dataloader(self) -> DataLoader:
    assert self.validation_dataset is not None
    return self._generic_dataloader(self.validation_dataset, num_workers=self.validation_workers)

  def test_dataloader(self) -> DataLoader:
    assert self.test_dataset is not None
    return self._generic_dataloader(self.test_dataset, max_workers=1)