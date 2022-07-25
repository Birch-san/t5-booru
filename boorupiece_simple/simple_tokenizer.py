from __future__ import annotations
from abc import abstractmethod
from typing import Iterable, Set, List
from enum import IntEnum
from itertools import chain

class Tokenizer:
  vocab: IntEnum
  def __init__(
    self,
    tokens: Iterable[str],
  ) -> None:
    tokens_list: List[str] = []
    tokens_set: Set[str] = set()

    for token in tokens:
      if token not in tokens_set:
        tokens_list.append(token)
        tokens_set.add(token)
    
    self.vocab = IntEnum('Tokens', tokens_list, start=0)
  
  def _convert_token_to_id(self, token: str) -> int:
    return self.vocab[token]

  def _convert_id_to_token(self, index: int) -> str:
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.vocab(index)
  
  @property
  def vocab_size(self) -> int:
    return len(self.vocab)
  
  def has_token(self, token: str) -> bool:
    return self.vocab.__members__.get(token) is not None

  @abstractmethod
  def tokenize(self, **kwargs) -> List[str]:
    raise NotImplementedError
  

class TokenizerWithWellKnownTokens(Tokenizer):
  vocab: IntEnum
  eos_token: str
  unk_token: str
  pad_token: str
  eos_token_id: int
  unk_token_id: int
  pad_token_id: int
  def __init__(
    self,
    tokens: Iterable[str],
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
  ) -> None:
    super().__init__(
      tokens=chain([eos_token, unk_token, pad_token], tokens)
    )
    self.eos_token = eos_token
    self.unk_token = unk_token
    self.pad_token = pad_token
    self.eos_token_id = self._convert_token_to_id(eos_token)
    self.unk_token_id = self._convert_token_to_id(unk_token)
    self.pad_token_id = self._convert_token_to_id(pad_token)
  