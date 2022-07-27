from __future__ import annotations
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset
from os.path import join
from os import environ
from typing import NamedTuple, Optional, Iterator, Iterable, Callable, List, Dict, Tuple
from typing_extensions import TypeAlias
from sqlite3 import Connection, Cursor

from boorupiece_simple.boorupiece import BooruPiece
from .db import create_connection
from .booru_db import get_file_ids_from_nth, get_first_n_file_ids, file_ids_to_dtos, get_tag_dtos, BooruFileId, Tag, TagCategory
from argparse import ArgumentParser, Namespace
from more_itertools import partition
from util.enumeration_to_value import enumeration_to_value
from contextlib import closing
from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import groupby
from operator import itemgetter
from random import sample, shuffle

Tokenize: TypeAlias = Callable[[List[str]], List[int]]
PadTokens: TypeAlias = Callable[[List[int], int], List[int]]

class TagRetentionCategory(IntEnum):
  EXPENDABLE = auto()
  CRUCIAL = auto()

tag_category_retentions: Dict[TagCategory, TagRetentionCategory] = {
  TagCategory.GENERAL: TagRetentionCategory.EXPENDABLE,
  TagCategory.ARTIST: TagRetentionCategory.CRUCIAL,
  TagCategory.FRANCHISE_0: TagRetentionCategory.CRUCIAL,
  TagCategory.FRANCHISE_1: TagRetentionCategory.CRUCIAL,
}
def _classify_tag_category(tag_category: Optional[TagCategory]) -> TagRetentionCategory:
  # we are targeting Python 3.9 so sadly cannot use structural pattern matching
  return TagRetentionCategory.EXPENDABLE if tag_category is None else tag_category_retentions[tag_category]


class TagWithTokens(NamedTuple):
  tag: Tag
  tokens: List[str]

class RetentionClassified(NamedTuple):
  retention: TagRetentionCategory
  with_tokens: TagWithTokens

@dataclass
class Batch:
  unmasked: LongTensor
  masked: LongTensor

class BooruCharsCaptionsDataset(IterableDataset):
  file_ids: Iterable[BooruFileId]
  get_cursor: Callable[[], Cursor]
  def __init__(
    self,
    file_ids: Iterable[BooruFileId],
    get_cursor: Callable[[], Cursor],
  ) -> None:
    super(BooruCharsCaptionsDataset).__init__()
    self.file_ids = file_ids
    self.get_cursor = get_cursor
  
  def to_caption(self, file_id: BooruFileId) -> List[Tag]:
    # print(f'file_ids for {BOORU}, {FID}:')
    # cur: Cursor = self.get_cursor()
    with closing(self.get_cursor()) as cur:
      tags: List[Tag] = get_tag_dtos(cur, file_id)
      # print(f'len: {len(tags)}')
      # print(tags)
      return tags
  
  def __iter__(self) -> Iterator[List[Tag]]:
    for file_id in self.file_ids:
      tag_dtos: List[Tag] = self.to_caption(file_id)
      yield tag_dtos

class BooruCharsCaptions(LightningDataModule):
  batch_size: int
  validation_split_percentage: int
  test_quantity: int
  tokenize: Callable[[List[str]], List[int]]
  tokenizer: BooruPiece
  pad_tokens: PadTokens
  caption_max_labels: int
  caption_max_tokens: int
  sqlite_db_path: str
  train_dataset: Optional[BooruCharsCaptionsDataset] = None
  validation_dataset: Optional[BooruCharsCaptionsDataset] = None
  test_dataset: Optional[BooruCharsCaptionsDataset] = None
  close_fit: Optional[Callable[[], None]] = None
  close_test: Optional[Callable[[], None]] = None

  @staticmethod
  def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("BooruCharsCaptions")
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--test_quantity', type=int, default=32)
    parser.add_argument('--sqlite_db_path', type=str, default=join(environ['HOME'], 'machine-learning/booru-chars/booru-chars.db'))
    return parent_parser

  def __init__(
    self,
    args: Namespace,
    tokenize: Tokenize,
    tokenizer: BooruPiece,
    pad_tokens: PadTokens,
    caption_max_labels = 32,
    caption_max_tokens = 32,
  ) -> None:
    super().__init__()
    self.tokenize = tokenize
    self.tokenizer = tokenizer
    self.pad_tokens = pad_tokens
    self.caption_max_labels = caption_max_labels
    self.caption_max_tokens = caption_max_tokens
    self.batch_size = args.batch_size
    self.validation_split_percentage = args.validation_split_percentage
    self.test_quantity = args.test_quantity
    self.sqlite_db_path = args.sqlite_db_path
  
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
      self.train_dataset = BooruCharsCaptionsDataset(
        file_ids=map(enumeration_to_value, training),
        get_cursor=get_cursor
      )
      self.validation_dataset = BooruCharsCaptionsDataset(
        file_ids=map(enumeration_to_value, validation),
        get_cursor=get_cursor
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
      self.test_dataset = BooruCharsCaptionsDataset(
        file_ids=file_id_dtos,
        get_cursor=get_cursor
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
    
  def _pad(self, tokenized: List[int], length: int) -> List[int]:
    padded: List[int] = [*tokenized, *[self.pad_token_id] * (length - len(tokenized))]
    return padded

  def _cull_caption(self, tag_dtos: List[Tag]) -> List[str]:
    return None
  
  def _process_caption(self, tag_dtos: List[Tag]) -> List[int]:
    with_tokens: List[TagWithTokens] = [
      [
        tag_dto,
        self.tokenizer.tokenize_label(self.tokenizer.regularize_label(tag_dto.TAG))
      ] for tag_dto in tag_dtos
    ]

    sort_key_fn: Callable[[TagWithTokens], TagRetentionCategory] = itemgetter(0)
    # with_retentions: 
    by_retention_list: List[Tuple[TagRetentionCategory, str, List[str]]] = [
      (
        _classify_tag_category(tag_dto.CAT),
        tag_dto.TAG,
        [self.tokenizer.tokenize_label(self.tokenizer.regularize_label(tag_dto.TAG))]
      ) for tag_dto in tag_dtos
    ]
    by_retention_list.sort(key=sort_key_fn)
    by_retention: Dict[TagRetentionCategory, List[str]] = {
      key: list(valuesiter) for key, valuesiter in groupby(by_retention_list, key=sort_key_fn)
    }
    expendable = by_retention[TagRetentionCategory.EXPENDABLE]
    shuffle(expendable)
    crucial = by_retention[TagRetentionCategory.CRUCIAL]
    shuffle(crucial)

    self.caption_max_labels
    sample()
    # TODO
    tags: List[str] = [tag_dto.TAG for tag_dto in tag_dtos]
    tokens: List[int] = self.tokenize(tags)
    return tokens
  
  def collate_fn(self, tag_dtos_batch: List[List[Tag]]) -> Batch:
    tokens_batch: List[List[int]] = [self._process_caption(tag_dtos) for tag_dtos in tag_dtos_batch]
    pad_length: int = max(map(len, tokens_batch))
    padded: List[List[int]] = [self._pad(tokenized, pad_length) for tokenized in tokens_batch]
    # TODO
    batch = Batch(
      masked=LongTensor(padded),
      unmasked=LongTensor(padded),
    )
    return batch

  def train_dataloader(self) -> DataLoader:
    assert self.train_dataset is not None
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

  def val_dataloader(self) -> DataLoader:
    assert self.validation_dataset is not None
    return DataLoader(self.validation_dataset, batch_size=self.batch_size)

  def test_dataloader(self) -> DataLoader:
    assert self.test_dataset is not None
    return DataLoader(self.test_dataset, batch_size=self.batch_size)

  # def predict_dataloader(self) -> DataLoader:
  #   return DataLoader(self.mnist_predict, batch_size=self.batch_size)