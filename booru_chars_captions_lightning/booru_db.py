from __future__ import annotations
from sqlite3 import Cursor
from typing import Iterator, NamedTuple, List, Optional
from enum import IntEnum
from dataclasses import dataclass

class TagCategory(IntEnum):
  GENERAL = 0
  ARTIST = 1
  # no examples had TAG_CAT 2
  # yeah TAG_CAT 3 and 4 both appear to represent franchise
  FRANCHISE_0 = 3
  FRANCHISE_1 = 4

  @classmethod
  def parse(cls: TagCategory, category: int) -> Optional[TagCategory]:
    return cls(category) if category in cls._value2member_map_ else None
  
  def is_franchise(self: TagCategory) -> bool:
    return self is self.FRANCHISE_0 or self.FRANCHISE_1

# order by MD5 to get a deterministic random, otherwise it's ordered by primary key (booru, fid)
def get_file_ids(cur: Cursor) -> Cursor:
  return cur.execute("""\
select BOORU, FID from files
order by TORR_MD5
""")

def get_file_ids_from_nth(cur: Cursor, start: int) -> Cursor:
  return cur.execute("""\
select BOORU, FID from files
order by TORR_MD5
limit -1
offset :offset
""", {"offset": start})

def get_first_n_file_ids(cur: Cursor, length: int) -> Cursor:
  return cur.execute("""\
select BOORU, FID from files
order by TORR_MD5
limit :limit
""", {"limit": length})

class BooruFileId(NamedTuple):
  BOORU: str
  FID: str

def file_ids_to_dtos(cur: Cursor) -> Iterator[BooruFileId]:
  for BOORU, FID in cur:
    yield BooruFileId(BOORU=BOORU, FID=FID)

def get_tags(cur: Cursor, foreign_key: BooruFileId) -> List[str]:
  BOORU, FID = foreign_key
  cur.execute("""\
select distinct TAG from tags
where BOORU = :BOORU
  and FID = :FID
order by TAG ASC
""", {"BOORU": BOORU, "FID": FID})
  return list(map(lambda result: result[0], cur.fetchall()))

@dataclass
class TagRecord:
  TAG: str
  CAT: Optional[TagCategory]

def get_tag_records(cur: Cursor, foreign_key: BooruFileId) -> List[TagRecord]:
  BOORU, FID = foreign_key
  cur.execute("""\
select TAG, TAG_CAT from tags
where BOORU = :BOORU
  and FID = :FID
group by TAG
order by TAG ASC
""", {"BOORU": BOORU, "FID": FID})
  return [TagRecord(
    TAG=result[0],
    CAT=TagCategory.parse(int(result[1])),
  ) for result in cur.fetchall()]
