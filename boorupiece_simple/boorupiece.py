from __future__ import annotations
from typing import List, Iterable, Optional, NamedTuple
from itertools import chain
import re
from .booru_chars_token_registry import BooruCharsTokenRegistry

class QualifiedLabel(NamedTuple):
  nominal: str
  qualifier: str

class BooruPiece():
  token_registry: BooruCharsTokenRegistry
  def __init__(
    self,
    token_registry: BooruCharsTokenRegistry = BooruCharsTokenRegistry()
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
  def _split_on_qualifier(label: str) -> Optional[QualifiedLabel]:
    matches = re.search(r"^(.*)_\(([^)]*?)\)$", label)
    if matches is None:
      return None
    return QualifiedLabel(matches[1], matches[2])
  
  @staticmethod
  def _split_on_delimiter(label: str) -> List[str]:
    return re.split(r'[-_]', label)
  
  def _tokenize_unsplittable_label(self, label: str) -> str:
    return label if self.token_registry.has_token(label) else self.token_registry.unk_token
  
  def _tokenize_qualified_label(self, qualification: QualifiedLabel) -> Iterable[str]:
    nominal, qualifier = qualification
    qualifier_token: str = self._tokenize_unsplittable_label(qualifier)
    if qualifier == "cosplay":
      nominal_token: str = self._tokenize_unsplittable_label(nominal)
      return (nominal_token, qualifier_token)
    return [*self._tokenize_splittable_label(nominal), qualifier_token]

  def _tokenize_splittable_label(self, label: str) -> Iterable[str]:
    qualification: Optional[QualifiedLabel] = self._split_on_qualifier(label)
    if qualification is not None:
      return self._tokenize_qualified_label(qualification)
    splits: list[str] = self._split_on_delimiter(label)
    return tuple(self._tokenize_unsplittable_label(token) for token in splits)
  
  def tokenize_label(self, label: str) -> Iterable[str]:
    """
    Tokenize a label.
    Be sure to regularize your label first.
    """
    if self.token_registry.has_token(label):
      return (label,)
    if self._label_eligible_for_split(label):
      return self._tokenize_splittable_label(label)
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