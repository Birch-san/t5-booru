from __future__ import annotations
from functools import reduce
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from typing import Optional, Iterator, Iterable, Callable, List, Dict, TypeVar, Set, Union, Generic
from typing_extensions import TypeAlias
from sqlite3 import Cursor, Connection
from contextlib import closing
from operator import add

from .db import create_connection
from .booru_db import get_tag_records, BooruFileId, TagRecord, TagCategory, DatasetSplit, get_train_fids, get_validation_fids
from argparse import ArgumentParser, Namespace
from pytorch_lightning.utilities.argparse import from_argparse_args
from contextlib import closing
from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import groupby, chain
from random import shuffle, sample, random

@dataclass
class Example:
  unmasked: List[int]
  masked: List[int]

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

@dataclass
class BooruCharsCaptionsDatasetParams:
  sqlite_db_path: str
  dataset_split: DatasetSplit
  is_validation: bool

TokenMaskStategy: TypeAlias = Callable[[List[TagWithTokens]], List[int]]

T = TypeVar('T', bound=Dataset)

class _WorkerInfo(Generic[T]):
  id: int
  num_workers: int
  seed: int
  dataset: T

TokenizeLabel: TypeAlias = Callable[[str], Iterable[str]]
EncodeToken: TypeAlias = Callable[[str], int]
IsKnownToken: TypeAlias = Callable[[int], bool]
CloseHandle: TypeAlias = Callable[[], None]
GetCursor: TypeAlias = Callable[[], Cursor]

class BooruCharsCaptionsDataset(IterableDataset):
  get_cursor: Optional[GetCursor]
  close_conn: Optional[CloseHandle]
  tokenize_label: TokenizeLabel
  encode_token: EncodeToken
  is_known_token: IsKnownToken
  caption_min_tokens: int
  caption_max_crucial_tokens: int
  caption_max_tokens: int
  max_tokens_masked: int
  mask_strategy_label_chance: int
  is_validation: bool
  dataset_split: DatasetSplit
  sqlite_db_path: str

  @staticmethod
  def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("BooruCharsCaptionsDataset")
    parser.add_argument('--caption_max_tokens', type=int, default=32)
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
    caption_min_tokens = 4,
    caption_max_crucial_tokens = 4,
    max_tokens_masked = 8,
    mask_strategy_label_chance = 0.5,
  ) -> None:
    super(BooruCharsCaptionsDataset).__init__()
    self.is_validation = params.is_validation
    self.dataset_split = params.dataset_split
    self.sqlite_db_path = params.sqlite_db_path
    self.tokenize_label = tokenize_label
    self.encode_token = encode_token
    self.is_known_token = is_known_token
    self.caption_min_tokens = caption_min_tokens
    self.caption_max_crucial_tokens = caption_max_crucial_tokens
    self.caption_max_tokens = caption_max_tokens
    self.max_tokens_masked = max_tokens_masked
    self.mask_strategy_label_chance = mask_strategy_label_chance
  
  def teardown(self) -> None:
    if (callable(self.close_conn)):
      self.close_conn()
    self.close_conn = None
    self.get_cursor = None
  
  def _to_caption(self, file_id: BooruFileId) -> List[TagRecord]:
    # print(f'file_ids for {booru}, {fid}:')
    # cur: Cursor = self.get_cursor()
    assert callable(self.get_cursor)
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
  
  def _booru_fid_to_example(self, booru_fid: BooruFileId) -> Optional[Example]:
    tags: List[TagRecord] = self._to_caption(booru_fid)
    tag_dtos: Optional[List[TagWithTokens]] = self._tokens_of_suitable_captions(tags)
    if tag_dtos is None:
      return None
    example: Example = self._to_example(tag_dtos)
    return example

  def __iter__(self) -> Iterator[Example]:
    worker_info: Optional[_WorkerInfo[BooruCharsCaptionsDataset]] = get_worker_info()
    assert worker_info is not None
    worker_info: _WorkerInfo[BooruCharsCaptionsDataset] = worker_info
    worker_id = worker_info.id
    num_workers = worker_info.num_workers

    total, validation_count = self.dataset_split
    train_count: int = total-validation_count

    with closing(create_connection(self.sqlite_db_path)) as conn:
      self.get_cursor = conn.cursor
      self.close_conn = conn.close
      with closing(self.get_cursor()) as cur:
        file_ids: Iterable[BooruFileId] = get_validation_fids(
          cur=cur,
          validation_quantity=self.dataset_split.validation,
          rank=worker_id,
          workers=num_workers,
        ) if self.is_validation else get_train_fids(
          cur=cur,
          train_quantity=train_count,
          validation_count=validation_count,
          rank=worker_id,
          workers=num_workers,
        )
        for booru_fid in file_ids:
          example: Optional[Example] = self._booru_fid_to_example(booru_fid=booru_fid)
          if example is not None:
            yield example