import sqlite3
from sqlite3 import Error, Connection

def create_connection(db_file: str) -> Connection:
  """ create a database connection to a SQLite database """
  conn = None
  try:
    conn = sqlite3.connect(db_file)
    return conn
  except Error as e:
    print(e)
  finally:
    if conn:
      conn.close()
