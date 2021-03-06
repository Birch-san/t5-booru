from __future__ import annotations
from typing import Set, List, Union, Tuple, Dict#, Optional
from transformers import PretrainedConfig, PreTrainedTokenizer
from enum import IntEnum
from io import TextIOWrapper
from os.path import exists, splitext
import gzip, shutil
from transformers.tokenization_utils_base import TextInput, AddedToken, TruncationStrategy, PreTokenizedInput, EncodedInput
from itertools import chain
import re
# from transformers.utils import PaddingStrategy
# from transformers.utils.generic import TensorType

VOCAB_FILES_NAMES = {
  'compressed_general_tokens_file': 'general_tokens.tsv.gz',
  'compressed_label_tokens_file': 'label_tokens.tsv.gz'
}

class BooruPieceConfig(PretrainedConfig):
  model_type = "boorupiece"
  def __init__(
    self,
    **kwargs
  ) -> None:
    super().__init__(**kwargs)

# TODO: regularize input captions, e.g. lowercase, sort labels, split labels on underscore and on hyphen (except where length <= 4, because face tokens)
class BooruPiece(PreTrainedTokenizer):
  vocab_files_names = VOCAB_FILES_NAMES
  vocab: IntEnum
  vocab_plus_added_tokens: Dict[str, int]
  _extra_ids: int
  def __init__(
    self,
    compressed_general_tokens_file: str,
    compressed_label_tokens_file: str,
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    extra_ids=100,
    additional_special_tokens: Union[List[str], Tuple[str, ...], None] = None,
    model_max_length=512,
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
      model_max_length=model_max_length,
      **kwargs
    )

    self._extra_ids = extra_ids

    label_tokens_file = self.get_decompressed_path(compressed_label_tokens_file)
    general_tokens_file = self.get_decompressed_path(compressed_general_tokens_file)

    self.ensure_decompressed(compressed_label_tokens_file, label_tokens_file)
    self.ensure_decompressed(compressed_general_tokens_file, general_tokens_file)

    # we avoid ingesting directly to set(), in order to preserve insertion-order.
    labels: List[str] = self.vocab_file_lines(label_tokens_file)
    tokens: List[str] = self.vocab_file_lines(general_tokens_file)
    tokens_set: Set[str] = set(tokens)

    labels_filtered: List[str] = [label for label in labels if label not in tokens_set]
    self.vocab = IntEnum('Tokens', [pad_token, eos_token, unk_token] + labels_filtered + tokens + (additional_special_tokens or []), start=0)
    self.vocab_plus_added_tokens = self._compute_vocab_plus_added_tokens()
  
  @staticmethod
  def get_decompressed_path(compressed_file_path: str) -> str:
    decompressed_file_path, *_ = splitext(compressed_file_path)
    return decompressed_file_path
  
  @staticmethod
  def ensure_decompressed(compressed_file_path: str, decompressed_file_path: str) -> None:
    if exists(compressed_file_path) and not exists(decompressed_file_path):
      with gzip.open(compressed_file_path, 'r') as f_in, open(decompressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
  
  @staticmethod
  def vocab_file_lines(file_path: str) -> list[str]:
    file: TextIOWrapper = open(file_path, 'r')
    lines: list[str] = [cleaned for cleaned in (line.rstrip('\n') for line in file.readlines()) if cleaned != '']
    return lines
  
  def _convert_token_to_id(self, token: str) -> int:
    return self.vocab[token]

  def _convert_id_to_token(self, index: int) -> str:
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.vocab(index)
  
  @property
  def vocab_size(self) -> int:
    """
    `int`: Size of the base vocabulary (without the added tokens).
    """
    return len(self.vocab) + self._extra_ids
  
  def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
    super()._add_tokens(new_tokens, special_tokens)
    self.vocab_plus_added_tokens = self._compute_vocab_plus_added_tokens()
  
  def _compute_vocab_plus_added_tokens(self) -> Dict[str, int]:
    return { **self.vocab.__members__, **self.added_tokens_encoder }
  
  def get_vocab(self) -> Dict[str, int]:
    return self.vocab_plus_added_tokens
  
  def has_token(self, token: str) -> bool:
    return self.vocab_plus_added_tokens.get(token) is not None
  
  regex_delimiter = r'[-_]'
  def tokenize_label(self, word: str) -> List[str]:
    lower: str = word.lower()
    if (self.has_token(lower)):
      return [lower]
    # we don't split short tokens on hyphens/underscores, because they're likely to be kaomoji
    if (len(lower) > 4 and re.search(self.regex_delimiter, lower)):
      splits: list[str] = re.split(self.regex_delimiter, lower)
      return list(chain.from_iterable(self.tokenize_label(token) for token in splits))
    return [self.unk_token]
  
  def tokenize(
    self,
    text: TextInput,
    is_split_into_words = False,
    **kwargs
    ) -> List[str]:
      words: List[str] = [text] if is_split_into_words else text.split(' ')
      tokens: List[str] = list(chain.from_iterable(self.tokenize_label(label) for label in words))
      return tokens

  # def encode(
  #   self,
  #   text: Union[TextInput, PreTokenizedInput, EncodedInput],
  #   text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
  #   add_special_tokens: bool = True,
  #   padding: Union[bool, str, PaddingStrategy] = False,
  #   truncation: Union[bool, str, TruncationStrategy] = False,
  #   max_length: Optional[int] = None,
  #   stride: int = 0,
  #   return_tensors: Optional[Union[str, TensorType]] = None,
  #   **kwargs
  #   ) -> List[int]:
  #     return super().encode(
  #       text,
  #       text_pair,
  #       add_special_tokens,
  #       padding,
  #       truncation,
  #       max_length,
  #       stride,
  #       return_tensors,
  #       **kwargs
  #       )