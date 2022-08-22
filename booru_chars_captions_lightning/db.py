import sqlite3
from sqlite3 import Connection
import functools as ft

@ft.lru_cache(None)
def get_sqlite3_thread_safety():
  # Map value from SQLite's THREADSAFE to Python's DBAPI 2.0
  # threadsafety attribute.
  sqlite_threadsafe2python_dbapi = {0: 0, 2: 1, 1: 3}
  conn = sqlite3.connect(":memory:")
  threadsafety = conn.execute("""
select * from pragma_compile_options
where compile_options like 'THREADSAFE=%'
""").fetchone()[0]
  conn.close()

  threadsafety_value = int(threadsafety.split("=")[1])

  return sqlite_threadsafe2python_dbapi[threadsafety_value]

def create_connection(db_file: str) -> Connection:
  """ create a database connection to a SQLite database """
  # https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
  check_same_thread=get_sqlite3_thread_safety() is not 3
  return sqlite3.connect(db_file, check_same_thread=check_same_thread)
