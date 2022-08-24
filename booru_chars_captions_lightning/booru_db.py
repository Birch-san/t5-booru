from __future__ import annotations
from sqlite3 import Connection, Cursor
from typing import Iterator, NamedTuple, List, Optional, Iterable, Union, TypedDict
from typing_extensions import TypeAlias
from enum import IntEnum
from dataclasses import dataclass

_Queryable: TypeAlias = Union[Cursor, Connection]

class TagCategory(IntEnum):
  GENERAL = 0
  ARTIST = 1
  # no examples had TAG_CAT 2 (perhaps category 2 is 'meta', for tags like 'animated')
  COPYRIGHT = 3
  CHARACTER = 4

  @classmethod
  def parse(cls: TagCategory, category: int) -> Optional[TagCategory]:
    return cls(category) if category in cls._value2member_map_ else None
  
  def is_franchise(self: TagCategory) -> bool:
    return self is self.COPYRIGHT or self.CHARACTER

# order by MD5 to get a deterministic random, otherwise it's ordered by primary key (booru, fid)
def get_file_ids(cur: _Queryable) -> Cursor:
  return cur.execute("""\
select booru, fid from files
order by torr_md5
""")

def get_file_ids_from_nth(cur: _Queryable, start: int) -> Cursor:
  return cur.execute("""\
select booru, fid from files
order by torr_md5
limit -1
offset :offset
""", {"offset": start})

def get_first_n_file_ids(cur: _Queryable, length: int) -> Cursor:
  return cur.execute("""\
select booru, fid from files
order by torr_md5
limit :limit
""", {"limit": length})

class BooruFileId(NamedTuple):
  booru: str
  fid: str

class BooruFileIdDict(TypedDict):
  booru: str
  fid: str

def file_ids_to_dtos(cur: Cursor) -> Iterator[BooruFileId]:
  for booru, fid in cur:
    yield BooruFileId(booru=booru, fid=fid)

def get_tags(cur: _Queryable, foreign_key: BooruFileId) -> List[str]:
  booru, fid = foreign_key
  cur.execute("""\
select distinct tag from tags
where booru = :booru
  and fid = :fid
order by tag asc
""", BooruFileIdDict(booru=booru, fid=fid))
  return list(map(lambda result: result[0], cur.fetchall()))

@dataclass
class TagRecord:
  tag: str
  cat: Optional[TagCategory]

def get_tag_records(cur: _Queryable, foreign_key: BooruFileId) -> List[TagRecord]:
  booru, fid = foreign_key
  cur.execute("""\
select tag, tag_cat from tags
where booru = :booru
  and fid = :fid
group by tag
order by tag asc
""", BooruFileIdDict(booru=booru, fid=fid))
  return [TagRecord(
    tag=result[0],
    cat=TagCategory.parse(int(result[1])),
  ) for result in cur.fetchall()]

class DatasetSplit(NamedTuple):
  total: int
  validation: int

def get_dataset_split(
  cur: _Queryable,
  validation_split_coeff = 0.05,
  validation_shards: Iterable[str] = ('Safebooru 2021a', 'Safebooru 2021b', 'Safebooru 2021c', 'Safebooru 2021d', 'Safebooru 2022a', 'Safebooru 2022b', 'manual_walfie_0'),
) -> DatasetSplit:
  shards_template: str = ', '.join(f':validation_shard{ix}' for ix, _ in enumerate(validation_shards))
  cur.execute(f"""\
select
count(*) as total,
floor(count(*) * :validation_split) as validation
from files
where shard in ({shards_template})
""", {
    "validation_split": validation_split_coeff,
    **{
      f'validation_shard{ix}': value for ix, value in enumerate(validation_shards)
    },
  })
  return DatasetSplit(*cur.fetchone())

def get_validation_fids(
  cur: _Queryable,
  validation_quantity: int,
  validation_shards: Iterable[str] = ('Safebooru 2021a', 'Safebooru 2021b', 'Safebooru 2021c', 'Safebooru 2021d', 'Safebooru 2022a', 'Safebooru 2022b', 'manual_walfie_0'),
  rank = 0,
  workers = 1,
) -> Iterator[BooruFileId]:
  shard_size: float = validation_quantity / workers
  shards_template: str = ', '.join(f':validation_shard{ix}' for ix, _ in enumerate(validation_shards))
  cur.execute(f"""\
select f.booru, f.fid
from files f
where shard in ({shards_template})
order by torr_md5
limit :limit
offset :offset
""", {
    "limit": int(shard_size),
    "offset": int(shard_size * rank),
    **{
      f'validation_shard{ix}': value for ix, value in enumerate(validation_shards)
    },
  })
  return file_ids_to_dtos(cur)

def get_recognisable_validation_fids(
  cur: _Queryable,
  validation_quantity: int,
  example_quantity: int,
  validation_shards: Iterable[str] = ('Safebooru 2021a', 'Safebooru 2021b', 'Safebooru 2021c', 'Safebooru 2021d', 'Safebooru 2022a', 'Safebooru 2022b', 'manual_walfie_0'),
  blessed_franchises: Iterable[str] = ('hololive', 'touhou'),
) -> Iterator[BooruFileId]:
  shards_template: str = ', '.join(f':validation_shard{ix}' for ix, _ in enumerate(validation_shards))
  cur.execute(f"""\
select f.booru, f.fid
from (
  select f2.booru, f2.fid
  from files f2
  where f2.shard in ({shards_template})
  order by torr_md5
  limit :validation_quantity
) f
where exists(
  select null
    from tags t
  where f.booru = t.booru
    and f.fid = t.fid
    and t.tag in :blessed_franchises
)
limit :example_quantity
""", {
    "validation_quantity": validation_quantity,
    "example_quantity": min(validation_quantity, example_quantity),
    "blessed_franchises": blessed_franchises,
    **{
      f'validation_shard{ix}': value for ix, value in enumerate(validation_shards)
    },
  })
  return file_ids_to_dtos(cur)


def get_train_fids(
  cur: _Queryable,
  train_quantity: int,
  validation_quantity: int,
  validation_shards: Iterable[str] = ('Safebooru 2021a', 'Safebooru 2021b', 'Safebooru 2021c', 'Safebooru 2021d', 'Safebooru 2022a', 'Safebooru 2022b', 'manual_walfie_0'),
  rank = 0,
  workers = 1,
) -> Iterator[BooruFileId]:
  shard_size: float = train_quantity / workers
  shards_template: str = ', '.join(f':validation_shard{ix}' for ix, _ in enumerate(validation_shards))
  cur.execute(f"""\
select f.booru, f.fid
from files f
where (f.booru, f.fid) not in (
  select f2.booru, f2.fid
  from files f2
  where f2.shard in ({shards_template})
  order by torr_md5
  limit :validation_limit
)
order by torr_md5
limit :limit
offset :offset
""", {
    "validation_limit": validation_quantity,
    "limit": int(shard_size),
    "offset": int(shard_size * rank),
    **{
      f'validation_shard{ix}': value for ix, value in enumerate(validation_shards)
    },
  })
  return file_ids_to_dtos(cur)