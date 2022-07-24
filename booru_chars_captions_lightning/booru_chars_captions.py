from __future__ import annotations
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from os.path import join
from os import environ
from typing import Optional, Iterator, Iterable, Callable, List, TypedDict
from typing_extensions import TypeAlias
from sqlite3 import Connection, Cursor
from .db import create_connection
from .booru_db import get_file_ids_from_nth, get_first_n_file_ids, file_ids_to_dtos, get_tags, BooruFileId
from argparse import ArgumentParser, Namespace
from more_itertools import partition
from util.enumeration_to_value import enumeration_to_value
from contextlib import closing

Caption: TypeAlias = List[str]
Tokenize: TypeAlias = Callable[[Caption], List[int]]

class BooruCharsCaptionsDatasetParams(TypedDict):
  file_ids: Iterable[BooruFileId]
  get_cursor: Callable[[], Cursor]

class BooruCharsCaptionsDataset(IterableDataset):
  file_ids: Iterable[BooruFileId]
  get_cursor: Callable[[], Cursor]
  tokenize: Callable[[Caption], List[int]]
  def __init__(
    self,
    file_ids: Iterable[BooruFileId],
    get_cursor: Callable[[], Cursor],
    tokenize: Tokenize,
  ) -> None:
    super(BooruCharsCaptionsDataset).__init__()
    self.file_ids = file_ids
    self.get_cursor = get_cursor
    self.tokenize = tokenize
  
  def to_caption(self, file_id: BooruFileId) -> Caption:
    # print(f'file_ids for {BOORU}, {FID}:')
    # cur: Cursor = self.get_cursor()
    with closing(self.get_cursor()) as cur:
      tags: List[str] = get_tags(cur, file_id)
      # print(f'len: {len(tags)}')
      # print(tags)
      return tags
  
  def __iter__(self) -> Iterator[List[int]]:
    for file_id in self.file_ids:
      caption: Caption = self.to_caption(file_id)
      tokenized: List[int] = self.tokenize(caption)
      yield tokenized

# inversion-of-control. we don't want BooruCharsCaptions to be polluted with pass-through params.
# this factory gives us a way to supply a tokenize() function to the BooruCharsCaptionsDataset constructor,
# without making BooruCharsCaptions depend on tokenize() too.
BooruCharsCaptionsDatasetFactory: TypeAlias = Callable[[BooruCharsCaptionsDatasetParams], BooruCharsCaptionsDataset]

class BooruCharsCaptions(LightningDataModule):
  batch_size: int
  validation_split_percentage: int
  test_quantity: int
  sqlite_db_path: str
  dataset_factory: BooruCharsCaptionsDatasetFactory
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
    dataset_factory: BooruCharsCaptionsDatasetFactory,
  ) -> None:
    super().__init__()
    self.dataset_factory = dataset_factory
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

  def train_dataloader(self) -> DataLoader:
    assert self.train_dataset is not None
    return DataLoader(self.train_dataset, batch_size=self.batch_size)

  def val_dataloader(self) -> DataLoader:
    assert self.validation_dataset is not None
    return DataLoader(self.validation_dataset, batch_size=self.batch_size)

  def test_dataloader(self) -> DataLoader:
    assert self.test_dataset is not None
    return DataLoader(self.test_dataset, batch_size=self.batch_size)

  # def predict_dataloader(self) -> DataLoader:
  #   return DataLoader(self.mnist_predict, batch_size=self.batch_size)