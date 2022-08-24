from __future__ import annotations
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, List
from typing_extensions import TypeAlias
from multiprocessing import cpu_count
from contextlib import closing

from .db import create_connection
from .booru_db import get_dataset_split, DatasetSplit
from .booru_chars_caption_dataset import BooruCharsCaptionsDataset, BooruCharsCaptionsDatasetParams, Example
from argparse import ArgumentParser
from dataclasses import dataclass

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
  validation_split_coeff: int
  validation_workers: int
  train_workers: int
  pad_tokens: PadTokens
  dataset_factory: BooruCharsCaptionsDatasetFactory
  sqlite_db_path: str
  train_dataset: Optional[BooruCharsCaptionsDataset] = None
  validation_dataset: Optional[BooruCharsCaptionsDataset] = None
  close_fit: Optional[Callable[[], None]] = None
  close_test: Optional[Callable[[], None]] = None

  @staticmethod
  def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("BooruCharsCaptions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--sqlite_db_path', type=str, default='booru-chars.db')
    parser.add_argument('--data_max_workers', type=int, default=2)#max(1, cpu_count() - 3))
    return parent_parser

  def __init__(
    self,
    batch_size: int,
    validation_split_percentage: int,
    data_max_workers: int,
    sqlite_db_path: str,
    pad_tokens: PadTokens,
    dataset_factory: BooruCharsCaptionsDatasetFactory,
  ) -> None:
    super().__init__()
    self.pad_tokens = pad_tokens
    self.dataset_factory = dataset_factory
    self.batch_size = batch_size
    self.validation_split_coeff = validation_split_percentage/100
    self.validation_workers = max(1, data_max_workers//2)
    self.train_workers = max(1, data_max_workers-self.validation_workers)
    self.sqlite_db_path = sqlite_db_path

  def setup(self, stage: Optional[str] = None) -> None:
    # Assign train/val datasets for use in dataloaders
    if stage == "fit" or stage is None:
      if callable(self.close_fit):
        self.close_fit()
      
      dataset_split: Optional[DatasetSplit]
      
      # auto-closes connection object
      with closing(create_connection(self.sqlite_db_path)) as conn:
        # auto-commit to the database
        with conn:
          with closing(conn.cursor()) as cur:
            dataset_split: DatasetSplit = get_dataset_split(cur, validation_split_coeff=self.validation_split_coeff)
      
      assert dataset_split is not None

      self.train_dataset = self.dataset_factory(
        BooruCharsCaptionsDatasetParams(
          dataset_split=dataset_split,
          is_validation=False,
          sqlite_db_path=self.sqlite_db_path,
        )
      )
      self.validation_dataset = self.dataset_factory(
        BooruCharsCaptionsDatasetParams(
          dataset_split=dataset_split,
          is_validation=False,
          sqlite_db_path=self.sqlite_db_path,
        )
      )
      self.close_fit = lambda: (dataset.teardown() for dataset in (self.train_dataset, self.validation_dataset))
  
  def teardown(self, stage: Optional[str] = None) -> None:
    if stage == "fit" or stage is None:
      if callable(self.close_fit):
        self.close_fit()
      self.train_dataset = None
      self.validation_dataset = None
  
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
    return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=num_workers)

  def train_dataloader(self) -> DataLoader:
    assert self.train_dataset is not None
    return self._generic_dataloader(self.train_dataset, num_workers=self.train_workers)

  def val_dataloader(self) -> DataLoader:
    assert self.validation_dataset is not None
    return self._generic_dataloader(self.validation_dataset, num_workers=self.validation_workers)
