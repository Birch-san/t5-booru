from sqlite3 import Cursor

# order by MD5 to get a deterministic random, otherwise it's ordered by primary key (booru, fid)
def get_file_ids(cur: Cursor) -> Cursor:
  cur.execute("""\
select BOORU, FID from files
order by TORR_MD5
""")