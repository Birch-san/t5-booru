from __future__ import annotations
from functools import reduce
from git import WorkTreeRepositoryUnsupported
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader, IterableDataset, Dataset
from os.path import join
from os import environ
from typing import Optional, Iterator, Iterable, Callable, List, Dict, TypeVar, Set, Union
from typing_extensions import TypeAlias
from sqlite3 import Connection, Cursor
from operator import add
from multiprocessing import cpu_count
from threading import Thread, local, Lock
# import threading
from queue import Queue

from .db import create_connection
from .booru_db import get_file_ids_from_nth, get_first_n_file_ids, file_ids_to_dtos, get_tag_records, BooruFileId, TagRecord, TagCategory
from argparse import ArgumentParser, Namespace
from pytorch_lightning.utilities.argparse import from_argparse_args
from more_itertools import partition
from util.enumeration_to_value import enumeration_to_value
from contextlib import closing
from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import groupby, chain
from random import shuffle, sample, random

Tokenize: TypeAlias = Callable[[List[str]], List[int]]
PadTokens: TypeAlias = Callable[[List[int], int], List[int]]

class TagRetentionCategory(IntEnum):
  EXPENDABLE = auto()
  CRUCIAL = auto()

tag_category_retentions: Dict[TagCategory, TagRetentionCategory] = {
  TagCategory.GENERAL: TagRetentionCategory.EXPENDABLE,
  TagCategory.ARTIST: TagRetentionCategory.CRUCIAL,
  TagCategory.COPYRIGHT: TagRetentionCategory.CRUCIAL,
  TagCategory.CHARACTER: TagRetentionCategory.CRUCIAL,
}
def _classify_tag_category(tag_category: Optional[TagCategory]) -> TagRetentionCategory:
  # we are targeting Python 3.9 so sadly cannot use structural pattern matching
  return TagRetentionCategory.EXPENDABLE if tag_category is None else tag_category_retentions[tag_category]

@dataclass
class HasTokens:
  tokens: List[int]
  def __len__(self) -> int:
    return len(self.tokens)

THasTokens = TypeVar('THasTokens', bound=HasTokens)

@dataclass
class TagRecordWithTokens(HasTokens):
  record: TagRecord

@dataclass
class TagWithTokens(HasTokens):
  tag: str

def retain_tag_only(record_dto: TagRecordWithTokens) -> TagWithTokens:
  return TagWithTokens(
    tokens=record_dto.tokens,
    tag=record_dto.record.tag,
  )

@dataclass
class RetentionClassified:
  retention: TagRetentionCategory
  tag_dto: TagWithTokens

@dataclass
class CountedTagDtos:
  tag_dtos: List[TagWithTokens]
  token_count: int

@dataclass
class ClassifiedCountedTagDtos:
  crucial: CountedTagDtos
  expendable: CountedTagDtos
  def token_count(self) -> int:
    return self.crucial.token_count + self.expendable.token_count

TokenMaskStategy: TypeAlias = Callable[[List[TagWithTokens]], List[int]]

@dataclass
class Example:
  unmasked: List[int]
  masked: List[int]

GetIntsFromExample: TypeAlias = Callable[[Example], List[int]]
PadTokensKnownLength: TypeAlias = Callable[[List[int]], List[int]]

@dataclass
class Batch:
  unmasked: LongTensor
  masked: LongTensor

GetCursor: TypeAlias = Callable[[], Cursor]
SignalFinished: TypeAlias = Callable[[], None]

@dataclass
class BooruCharsCaptionsDatasetParams:
  file_ids: Iterable[BooruFileId]
  get_cursor: GetCursor
  is_validation: bool

TokenizeLabel: TypeAlias = Callable[[str], Iterable[str]]
EncodeToken: TypeAlias = Callable[[str], int]
IsKnownToken: TypeAlias = Callable[[int], bool]

GetBooruFids: TypeAlias = Callable[[], Iterable[BooruFileId]]
Enqueue: TypeAlias = Callable[[Example], None]

class DatasetWorker(Thread):
  enqueue: Enqueue
  get_booru_fids: GetBooruFids
  get_cursor: GetCursor
  signal_finished: SignalFinished
  def __init__(
    self,
    enqueue: Enqueue,
    get_booru_fids: GetBooruFids,
    get_cursor: GetCursor,
    signal_finished: SignalFinished,
    *args,
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.enqueue = enqueue
    self.get_booru_fids = get_booru_fids
    self.get_cursor = get_cursor
    self.signal_finished = signal_finished
  
  def _to_caption(self, file_id: BooruFileId) -> List[TagRecord]:
    # print(f'file_ids for {booru}, {fid}:')
    # cur: Cursor = self.get_cursor()
    with closing(self.get_cursor()) as cur:
      tags: List[TagRecord] = get_tag_records(cur, file_id)
      # print(f'len: {len(tags)}')
      # print(tags)
      return tags
  
  def _needs_skipping(self, token_count: int) -> bool:
    return token_count < self.caption_min_tokens
  
  def _needs_shortening(self, token_count: int) -> bool:
    return token_count > self.caption_max_tokens
  
  def _join_tokens(self, records: List[TagRecord]) -> List[TagRecordWithTokens]:
    return [
      TagRecordWithTokens(
        record=record,
        tokens=[self.encode_token(token) for token in self.tokenize_label(record.tag)]
       ) for record in records
    ]
  
  def _without_unknown_labels(self, token_havers: Iterable[THasTokens]) -> List[THasTokens]:
    return [token_haver for token_haver in token_havers if all(map(self.is_known_token, token_haver.tokens))]
  
  @staticmethod
  def _count_tokens(token_havers: Iterable[HasTokens]) -> int:
    return reduce(add, map(len, token_havers), 0)
  
  sort_key_fn: Callable[[RetentionClassified], TagRetentionCategory] = lambda classified: classified.retention
  @classmethod
  def _classify_tag_priorities(cls: BooruCharsCaptionsDataset, tag_record_dtos: List[TagRecordWithTokens]) -> Dict[TagRetentionCategory, List[RetentionClassified]]:
    """
    We prefer not to throw away crucial tags like character names.
    Attach to each tag a designation to help us prioritize.
    """
    by_retention_list: List[RetentionClassified] = [
      RetentionClassified(
        retention=_classify_tag_category(tag_record_dto.record.cat),
        tag_dto=retain_tag_only(tag_record_dto)
      ) for tag_record_dto in tag_record_dtos
    ]
    by_retention_list.sort(key=cls.sort_key_fn)
    by_retention: Dict[TagRetentionCategory, List[RetentionClassified]] = {
      key: list(valuesiter) for key, valuesiter in groupby(by_retention_list, key=cls.sort_key_fn)
    }
    return by_retention
  
  @staticmethod
  def _accumulate_until(tag_dtos: Iterable[TagWithTokens], limit: int) -> CountedTagDtos:
    retained: List[TagWithTokens] = []
    token_count = 0
    for tag_dto in tag_dtos:
      tokens_len: int = len(tag_dto.tokens)
      if token_count + tokens_len > limit:
        continue
      retained.append(tag_dto)
      token_count += tokens_len
      if token_count == limit:
        break
    return CountedTagDtos(token_count=token_count, tag_dtos=retained)

  def _shorten(self, tag_record_dtos: List[TagRecordWithTokens]) -> ClassifiedCountedTagDtos:
    """
    Some of these captions have about 100 tags.
    We wanna get it down to about 32.
    """
    classifieds: Dict[TagRetentionCategory, List[RetentionClassified]] = self._classify_tag_priorities(tag_record_dtos)

    expendable: List[TagWithTokens] = [classified.tag_dto for classified in classifieds.get(TagRetentionCategory.EXPENDABLE, [])]
    crucial: List[TagWithTokens] = [classified.tag_dto for classified in classifieds.get(TagRetentionCategory.CRUCIAL, [])]
    shuffle(expendable)
    shuffle(crucial)

    retained_crucial: CountedTagDtos = self._accumulate_until(crucial, self.caption_max_crucial_tokens)
    retain_expendable_count: int = self.caption_max_tokens - retained_crucial.token_count
    retained_expendable: CountedTagDtos = self._accumulate_until(expendable, retain_expendable_count)
    return ClassifiedCountedTagDtos(
      crucial=retained_crucial,
      expendable=retained_expendable,
    )
  
  @staticmethod
  def _sort_tag_dtos(tag_dtos: List[TagWithTokens]) -> None:
    tag_dtos.sort(key=lambda tag_dto: tag_dto.tag)
  
  def _tokens_of_suitable_captions(self, candidate: List[TagRecord]) -> Optional[List[TagWithTokens]]:
    with_tokens: List[TagRecordWithTokens] = self._join_tokens(candidate)
    without_unknowns: List[TagRecordWithTokens] = self._without_unknown_labels(with_tokens)
    tokens_total: int = self._count_tokens(without_unknowns)
    if self._needs_skipping(tokens_total):
      return None

    if not self._needs_shortening(tokens_total):
      tag_dtos: List[TagWithTokens] = [retain_tag_only(tag_record_dto) for tag_record_dto in without_unknowns]
      # sorting _should_ be redundant if we trust our database query and collation
      # but on the basis that sorting 32 items is cheap and that it'd be catastrophic
      # if our sorts were inconsistent: let's sort anyway
      self._sort_tag_dtos(tag_dtos)
      return tag_dtos
    
    shortened: ClassifiedCountedTagDtos = self._shorten(without_unknowns)
    tokens_total: int = shortened.token_count()

    if self._needs_skipping(tokens_total):
      return None

    tag_dtos: List[TagWithTokens] = [*shortened.crucial.tag_dtos, *shortened.expendable.tag_dtos]
    self._sort_tag_dtos(tag_dtos)

    return tag_dtos
  
  def _mask_tokens(self, tag_dtos: List[TagWithTokens]) -> List[int]:
    """
    Given [['black', 'dress'], ['white', 'hat']]
    (i.e. where a multiple tokens constitute a single label)
    Masks tokens rather than entire labels, deleting like so then flattening
    [['black', *], [*, 'hat']]
    ['black', 'hat']
    If one wanted to be a bit more focused on the "label-completion" objective:
    This could be rewritten to avoid masking single-token labels, and to never mask a label's every token.
    """
    count: int = self._count_tokens(tag_dtos)
    to_mask_count: int = max(1, min(self.max_tokens_masked, count))
    remove_indices: Set[int] = set(sample(range(0, count), k=to_mask_count))
    retained: List[int] = []
    ix = 0
    for tag_dto in tag_dtos:
      for token in tag_dto.tokens:
        if ix not in remove_indices:
          retained.append(token)
        ix += 1
    return retained
  
  @staticmethod
  def _tags_to_token_ids(tag_dtos: Iterable[TagWithTokens]) -> List[int]:
    return list(chain.from_iterable(map(lambda tag_dto: tag_dto.tokens, tag_dtos)))
  
  def _mask_labels(self, tag_dtos: List[TagWithTokens]) -> List[int]:
    """
    Given [['black', 'dress'], ['white', 'hat']]
    Masks entire labels, deleting like so then flattening
    [['white', 'hat'], *]
    ['white', 'hat']
    """
    tag_dtos_shuffled: List[TagWithTokens] = sample(tag_dtos, k=len(tag_dtos))
    count: int = self._count_tokens(tag_dtos_shuffled)
    retain_count = max(1, count - self.max_tokens_masked)
    retained: List[TagWithTokens] = self._accumulate_until(tag_dtos_shuffled, retain_count).tag_dtos
    self._sort_tag_dtos(retained)
    token_ids: List[int] = self._tags_to_token_ids(retained)
    return token_ids

  def _mask(self, tag_dtos: List[TagWithTokens]) -> List[TagWithTokens]:
    # use two strategies randomly:
    # _mask_tokens (label-completion)
    # _mask_labels (caption-completion)
    strategy: TokenMaskStategy = self._mask_labels if random() < self.mask_strategy_label_chance else self._mask_tokens
    masked: List[int] = strategy(tag_dtos)
    return masked

  def _to_example(self, unmasked_dtos: List[TagWithTokens]) -> Example:
    masked: List[int] = self._mask(unmasked_dtos)
    unmasked: List[int] = self._tags_to_token_ids(unmasked_dtos)
    return Example(
      unmasked=unmasked,
      masked=masked,
    )
  
  def run(self) -> None:
    booru_fids = self.get_booru_fids()
    for file_id in booru_fids:
      tags: List[TagRecord] = self._to_caption(file_id)
      tag_dtos: Optional[List[TagWithTokens]] = self._tokens_of_suitable_captions(tags)
      if tag_dtos is None:
        continue
      example: Example = self._to_example(tag_dtos)
      self.enqueue(example)
    self.signal_finished()
    

class BooruCharsCaptionsDataset(IterableDataset):
  file_ids: Iterable[BooruFileId]
  get_cursor: GetCursor
  tokenize_label: TokenizeLabel
  encode_token: EncodeToken
  is_known_token: IsKnownToken
  caption_min_tokens: int
  caption_max_crucial_tokens: int
  caption_max_tokens: int
  max_tokens_masked: int
  mask_strategy_label_chance: int
  num_threads: int
  q: Queue
  threads_finished: int

  @staticmethod
  def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("BooruCharsCaptionsDataset")
    parser.add_argument('--caption_max_tokens', type=int, default=32)
    parser.add_argument('--dataset_num_threads', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--dataset_queue_max_size', type=int, default=512)
    return parent_parser
  
  @classmethod
  def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
    return from_argparse_args(cls, args, **kwargs)

  def __init__(
    self,
    params: BooruCharsCaptionsDatasetParams,
    tokenize_label: TokenizeLabel,
    encode_token: EncodeToken,
    is_known_token: IsKnownToken,
    caption_max_tokens: int,
    dataset_num_threads: int,
    dataset_queue_max_size: int,
    caption_min_tokens = 4,
    caption_max_crucial_tokens = 4,
    max_tokens_masked = 8,
    mask_strategy_label_chance = 0.5,
  ) -> None:
    super(BooruCharsCaptionsDataset).__init__()
    self.file_ids = params.file_ids
    self.get_cursor = params.get_cursor
    self.tokenize_label = tokenize_label
    self.encode_token = encode_token
    self.is_known_token = is_known_token
    self.caption_min_tokens = caption_min_tokens
    self.caption_max_crucial_tokens = caption_max_crucial_tokens
    self.caption_max_tokens = caption_max_tokens
    self.max_tokens_masked = max_tokens_masked
    self.mask_strategy_label_chance = mask_strategy_label_chance
    self.threads_finished = 0
    self.num_threads = max(dataset_num_threads//2 if params.is_validation else dataset_num_threads-dataset_num_threads//2, 1)
    self.q = Queue(maxsize=dataset_queue_max_size)
  
  def _thread() -> None:
    return

  def __iter__(self) -> Iterator[Example]:
    for rank in range(self.num_threads):
      DatasetWorker().start()
    while self.threads_finished < self.num_threads:

    for file_id in self.file_ids:
      tags: List[TagRecord] = self._to_caption(file_id)
      tag_dtos: Optional[List[TagWithTokens]] = self._tokens_of_suitable_captions(tags)
      if tag_dtos is None:
        continue
      example: Example = self._to_example(tag_dtos)
      yield example


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--test_quantity', type=int, default=32)
    parser.add_argument('--sqlite_db_path', type=str, default=join(environ['HOME'], 'machine-learning/booru-chars/booru-chars.db'))
    return parent_parser

  def __init__(
    self,
    batch_size: int,
    validation_split_percentage: int,
    test_quantity: int,
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
          get_cursor=get_cursor,
          is_validation=False
        )
      )
      self.validation_dataset = self.dataset_factory(
        BooruCharsCaptionsDatasetParams(
          file_ids=map(enumeration_to_value, validation),
          get_cursor=get_cursor,
          is_validation=True
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
  
  def _generic_dataloader(self, dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

  def train_dataloader(self) -> DataLoader:
    assert self.train_dataset is not None
    return self._generic_dataloader(self.train_dataset, num_workers=max(self.num_workers-self.num_workers//2, 1))

  def val_dataloader(self) -> DataLoader:
    assert self.validation_dataset is not None
    return self._generic_dataloader(self.validation_dataset, num_workers=max(self.num_workers//2, 1))

  def test_dataloader(self) -> DataLoader:
    assert self.test_dataset is not None
    return self._generic_dataloader(self.test_dataset, num_workers=1)