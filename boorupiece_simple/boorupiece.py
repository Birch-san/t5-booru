from __future__ import annotations
from typing import List, Iterable
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

  @staticmethod
  def regularize_label(label: str) -> str:
    return label.lower()
  
  @staticmethod
  def _label_eligible_for_split(label: str) -> bool:
    # we don't split short tokens on hyphens/underscores, because they're likely to be kaomoji
    return len(label) > 4
  
  @staticmethod
  def _split_label(label: str) -> List[str]:
    return re.split(r'[-_]', label)
  
  def tokenize_label(self, label: str) -> Iterable[str]:
    """
    Tokenize a label.
    Be sure to regularize your label first.
    """
    if (self.token_registry.has_token(label)):
      return (label,)
    if self._label_eligible_for_split(label):
      splits: list[str] = self._split_label(label)
      if len(splits) > 1:
        return chain.from_iterable(self.tokenize_label(token) for token in splits)
    return (self.token_registry.unk_token,)
  
  def tokenize_labels(self, labels: Iterable[str]) -> Iterable[str]:
    tokens: Iterable[str] = chain.from_iterable(self.tokenize_label(label) for label in labels)
    return tokens
  
  def encode_tokens(self, tokens: Iterable[str]) -> List[int]:
    return [self.token_registry.token_to_id(token) for token in tokens]
  
  def pad_tokens(self, tokens: List[int], length: int) -> List[int]:
    return [*tokens, *[self.token_registry.pad_token_id] * (length - len(tokens))]
  
  # def encode_batch(self, tokens: List[str]) -> List[int]:
  #   return [self.token_registry.token_to_id(token) for token in tokens]