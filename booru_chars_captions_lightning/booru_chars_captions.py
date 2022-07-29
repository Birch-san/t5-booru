from __future__ import annotations
from functools import reduce
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset
from os.path import join
from os import environ
from typing import NamedTuple, Optional, Iterator, Iterable, Callable, List, Dict, Tuple
from typing_extensions import Self, TypeAlias
from sqlite3 import Connection, Cursor
from operator import add

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
from random import shuffle

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
  def __len__(self) -> int:
    return len(self.tokens)

class RetentionClassified(NamedTuple):
  retention: TagRetentionCategory
  tag: TagWithTokens
  def __len__(self) -> int:
    return len(self.tag)

class AccumulatedTags(NamedTuple):
  token_count: int
  tags: List[TagWithTokens]
  def __len__(self) -> int:
    return len(self.tags)

class AccumulatedTagsByRetention(NamedTuple):
  crucial: AccumulatedTags
  expendable: AccumulatedTags
  def __len__(self) -> int:
    return len(self.crucial) + len(self.expendable)

@dataclass
class Batch:
  unmasked: LongTensor
  masked: LongTensor

@dataclass
class BooruCharsCaptionsDatasetParams:
  file_ids: Iterable[BooruFileId]
  get_cursor: Callable[[], Cursor]

TokenizeLabel: TypeAlias = Callable[[str], Iterable[str]]
IsKnownToken: TypeAlias = Callable[[str], bool]

class BooruCharsCaptionsDataset(IterableDataset):
  file_ids: Iterable[BooruFileId]
  get_cursor: Callable[[], Cursor]
  tokenize_label: TokenizeLabel
  is_known_token: IsKnownToken
  caption_min_tokens: int
  caption_max_crucial_tokens: int
  caption_max_tokens: int
  def __init__(
    self,
    params: BooruCharsCaptionsDatasetParams,
    tokenize_label: TokenizeLabel,
    is_known_token: IsKnownToken,
    caption_min_tokens = 4,
    caption_max_crucial_tokens = 4,
    caption_max_tokens = 32,
  ) -> None:
    super(BooruCharsCaptionsDataset).__init__()
    self.file_ids = params.file_ids
    self.get_cursor = params.get_cursor
    self.tokenize_label = tokenize_label
    self.is_known_token = is_known_token
    self.caption_min_tokens = caption_min_tokens
    self.caption_max_crucial_tokens = caption_max_crucial_tokens
    self.caption_max_tokens = caption_max_tokens
  
  def _to_caption(self, file_id: BooruFileId) -> List[Tag]:
    # print(f'file_ids for {BOORU}, {FID}:')
    # cur: Cursor = self.get_cursor()
    with closing(self.get_cursor()) as cur:
      tags: List[Tag] = get_tag_dtos(cur, file_id)
      # print(f'len: {len(tags)}')
      # print(tags)
      return tags
  
  def _needs_skipping(self, token_count: int) -> bool:
    return token_count < self.caption_min_tokens
  
  def _needs_shortening(self, token_count: int) -> bool:
    return token_count > self.caption_max_tokens
  
  def _join_tokens(self, tags: List[Tag]) -> List[TagWithTokens]:
    return [
      TagWithTokens(
        tag=tag,
        tokens=self.tokenize_label(tag.TAG)
       ) for tag in tags
    ]
  
  def _without_unknown_labels(self, tags: Iterable[TagWithTokens]) -> List[TagWithTokens]:
    return [tag for tag in tags if all(map(self.is_known_token, tag.tokens))]
  
  @staticmethod
  def _count_tokens(tags: Iterable[TagWithTokens]) -> int:
    return reduce(add, map(len, tags), 0)
  
  sort_key_fn: Callable[[RetentionClassified], TagRetentionCategory] = lambda classified: classified.retention
  @classmethod
  def _classify_tag_priorities(cls: BooruCharsCaptionsDataset, tags: List[TagWithTokens]) -> Dict[TagRetentionCategory, List[RetentionClassified]]:
    """
    We prefer not to throw away crucial tags like character names.
    Attach to each tag a designation to help us prioritize.
    """
    by_retention_list: List[RetentionClassified] = [
      RetentionClassified(
        retention=_classify_tag_category(tag.tag.CAT),
        tag=tag
      ) for tag in tags
    ]
    by_retention_list.sort(key=cls.sort_key_fn)
    by_retention: Dict[TagRetentionCategory, List[RetentionClassified]] = {
      key: list(valuesiter) for key, valuesiter in groupby(by_retention_list, key=cls.sort_key_fn)
    }
    return by_retention
  
  @staticmethod
  def _accumulate_until(tags: Iterable[TagWithTokens], limit: int) -> AccumulatedTags:
    acc: List[TagWithTokens] = []
    token_count = 0
    for tag in tags:
      tokens_len: int = len(tag)
      if token_count + tokens_len > limit:
        continue
      acc.append(tag)
      token_count += tokens_len
      if token_count == limit:
        break
    return AccumulatedTags(token_count=token_count, tags=acc)

  def _shorten(self, tags: List[TagWithTokens]) -> AccumulatedTagsByRetention:
    """
    Some of these captions have about 100 tags.
    We wanna get it down to about 32.
    """
    classifieds: Dict[TagRetentionCategory, List[RetentionClassified]] = self._classify_tag_priorities(tags)

    expendable: List[TagWithTokens] = [classified.tag for classified in classifieds.get(TagRetentionCategory.EXPENDABLE, [])]
    crucial: List[TagWithTokens] = [classified.tag for classified in classifieds.get(TagRetentionCategory.CRUCIAL, [])]
    shuffle(expendable)
    shuffle(crucial)

    retained_crucial: AccumulatedTags = self._accumulate_until(crucial, self.caption_max_crucial_tokens)
    retain_expendable_count: int = self.caption_max_tokens - retained_crucial.token_count
    retained_expendable: AccumulatedTags = self._accumulate_until(expendable, retain_expendable_count)
    return AccumulatedTagsByRetention(
      crucial=retained_crucial,
      expendable=retained_expendable,
    )

  def __iter__(self) -> Iterator[List[TagWithTokens]]:
    for file_id in self.file_ids:
      tags: List[Tag] = self._to_caption(file_id)
      with_tokens: List[TagWithTokens] = self._join_tokens(tags)
      without_unknowns: List[TagWithTokens] = self._without_unknown_labels(with_tokens)
      tokens_total: int = self._count_tokens(without_unknowns)
      if self._needs_skipping(tokens_total):
        continue

      if not self._needs_shortening(tokens_total):
        yield without_unknowns
      
      shortened: AccumulatedTagsByRetention = self._shorten(without_unknowns)
      tokens_total: int = len(shortened)

      if self._needs_skipping(tokens_total):
        continue

      union: List[TagWithTokens] = [*shortened.crucial.tags, *shortened.expendable.tags]
      union.sort(key=lambda tag: tag.tag.TAG)

      yield union

BooruCharsCaptionsDatasetFactory: TypeAlias = Callable[[BooruCharsCaptionsDatasetParams], BooruCharsCaptionsDataset]

class BooruCharsCaptions(LightningDataModule):
  batch_size: int
  validation_split_percentage: int
  test_quantity: int
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
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--test_quantity', type=int, default=32)
    parser.add_argument('--sqlite_db_path', type=str, default=join(environ['HOME'], 'machine-learning/booru-chars/booru-chars.db'))
    return parent_parser

  def __init__(
    self,
    args: Namespace,
    pad_tokens: PadTokens,
    dataset_factory: BooruCharsCaptionsDatasetFactory,
  ) -> None:
    super().__init__()
    self.pad_tokens = pad_tokens
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
    
  def _pad(self, tokenized: List[int], length: int) -> List[int]:
    padded: List[int] = [*tokenized, *[self.pad_token_id] * (length - len(tokenized))]
    return padded
  
  def collate_fn(self, tag_dtos_batch: List[List[TagWithTokens]]) -> Batch:
    # TODO sort the labels, decide which labels or tokens to mask, then map to token integers
    # tokens_batch: List[List[int]] = [self._process_caption(tag_dtos) for tag_dtos in tag_dtos_batch]
    pad_length: int = max(map(len, tag_dtos_batch))
    padded: List[List[int]] = [self._pad(tokenized, pad_length) for tokenized in tag_dtos_batch]
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