from typing import TypeVar, Tuple

T = TypeVar('T')

def enumeration_to_value(enumeration: Tuple[int, T]) -> T:
  _, val = enumeration
  return val