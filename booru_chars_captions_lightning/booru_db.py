from sqlite3 import Cursor
from typing import Iterator, NamedTuple, List

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
