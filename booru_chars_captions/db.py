import sqlite3
from sqlite3 import Connection

def create_connection(db_file: str) -> Connection:
  """ create a database connection to a SQLite database """
  return sqlite3.connect(db_file)
