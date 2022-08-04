from __future__ import annotations
from typing import Iterable, Set, List, TYPE_CHECKING
from enum import IntEnum
from itertools import chain

class TokenRegistry:
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
  
  def token_to_id(self, token: str) -> int:
    return self.vocab[token]

  def id_to_token(self, index: int) -> str:
    return self.vocab(index)
  
  @property
  def vocab_size(self) -> int:
    return len(self.vocab)
  
  def has_token(self, token: str) -> bool:
    return self.vocab.__members__.get(token) is not None

if TYPE_CHECKING:
  _RegistryMixinBase = TokenRegistry
else:
  _RegistryMixinBase = object

class PadTokenMixin(_RegistryMixinBase):
  pad_token: str
  pad_token_id: int
  def __init__(
    self,
    tokens: Iterable[str],
    pad_token="<pad>",
    **kwargs,
  ) -> None:
    super().__init__(
      tokens=chain((pad_token,), tokens),
      **kwargs,
    )
    self.pad_token = pad_token
    self.pad_token_id = self.token_to_id(pad_token)

class UnkTokenMixin(_RegistryMixinBase):
  unk_token: str
  unk_token_id: int
  def __init__(
    self,
    tokens: Iterable[str],
    unk_token="<unk>",
    **kwargs,
  ) -> None:
    super().__init__(
      tokens=chain((unk_token,), tokens),
      **kwargs,
    )
    self.unk_token = unk_token
    self.unk_token_id = self.token_to_id(unk_token)
