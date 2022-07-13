from __future__ import annotations
from typing import Set, List, Union, Tuple, Dict
from os import PathLike
from transformers import PretrainedConfig, PreTrainedTokenizer
from enum import IntEnum

class BooruPieceConfig(PretrainedConfig):
  model_type = "boorupiece"
  def __init__(
    self,
    **kwargs
  ) -> None:
    super().__init__(**kwargs)

class BooruPiece(PreTrainedTokenizer):
  tokens: Set[str]
  labels: Set[str]
  vocab: IntEnum
  _extra_ids: int
  def __init__(
    self,
    tokens: Set[str],
    labels: Set[str],
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    extra_ids=100,
    additional_special_tokens: Union[List[str], Tuple[str, ...], None] = None,
    **kwargs
  ) -> None:
    # Add extra_ids to the special token list
    if extra_ids > 0 and additional_special_tokens is None:
      additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
    elif extra_ids > 0 and additional_special_tokens is not None:
      # Check that we have the right number of extra_id special tokens
      extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
      if extra_tokens != extra_ids:
        raise ValueError(
          f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
          " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
          " tokens"
        )

    super().__init__(
      eos_token=eos_token,
      unk_token=unk_token,
      pad_token=pad_token,
      extra_ids=extra_ids,
      additional_special_tokens=additional_special_tokens,
      **kwargs
    )

    self._extra_ids = extra_ids

    labels_filtered: Set[str] = labels-tokens
    self.labels = labels_filtered
    self.tokens = tokens
    self.vocab = IntEnum('Tokens', tuple(self.labels.union(self.tokens)), start=0)
  
  @property
  def vocab_size(self) -> int:
    return len(self.vocab) + self._extra_ids
  
  def get_vocab(self) -> Dict[str, int]:
    return { **self.vocab.__members__, **self.added_tokens_encoder }
  
  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: Union[str, PathLike], *init_inputs, **kwargs) -> BooruPiece:
    tokenizer = cls(*init_inputs, tokens=set({}), labels=set({}), **kwargs)
    return tokenizer