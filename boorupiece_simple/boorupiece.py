from __future__ import annotations
from typing import List
from itertools import chain
import re
from .token_registry import TokenRegistryWithWellKnownTokens
from .booru_chars_token_registry import BooruCharsTokenRegistry

class BooruPiece():
  token_registry: TokenRegistryWithWellKnownTokens
  def __init__(
    self,
    token_registry: TokenRegistryWithWellKnownTokens = BooruCharsTokenRegistry()
  ) -> None:
    self.token_registry = token_registry
  
  @staticmethod
  def caption_to_labels(caption: str) -> List[str]:
    return caption.split(' ')
  
  regex_delimiter = r'[-_]'
  def tokenize_label(self, word: str) -> List[str]:
    lower: str = word.lower()
    if (self.token_registry.has_token(lower)):
      return [lower]
    # we don't split short tokens on hyphens/underscores, because they're likely to be kaomoji
    if (len(lower) > 4 and re.search(self.regex_delimiter, lower)):
      splits: list[str] = re.split(self.regex_delimiter, lower)
      return list(chain.from_iterable(self.tokenize_label(token) for token in splits))
    return [self.token_registry.unk_token]
  
  def tokenize_labels(self, labels: List[str]) -> List[str]:
    tokens: List[str] = list(chain.from_iterable(self.tokenize_label(label) for label in labels))
    return tokens
  
  def encode_tokens(self, tokens: List[str]) -> List[int]:
    return [self.token_registry.token_to_id(token) for token in tokens]
  
  def pad_tokens(self, tokens: List[int], length: int) -> List[int]:
    return [*tokens, *[self.token_registry.pad_token_id] * (length - len(tokens))]
  
  # def encode_batch(self, tokens: List[str]) -> List[int]:
  #   return [self.token_registry.token_to_id(token) for token in tokens]