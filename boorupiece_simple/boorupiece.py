from __future__ import annotations
from typing import Iterable, List
from enum import IntEnum
from io import TextIOWrapper
from os.path import exists, splitext
import gzip, shutil
from itertools import chain
from .simple_tokenizer import TokenizerWithWellKnownTokens

VOCAB_FILES_NAMES = {
  'compressed_general_tokens_file': 'general_tokens.tsv.gz',
  'compressed_label_tokens_file': 'label_tokens.tsv.gz'
}

class BooruPiece(TokenizerWithWellKnownTokens):
  vocab: IntEnum
  def __init__(
    self,
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    extra_tokens: Iterable[str] = [],
  ) -> None:
    compressed_label_tokens_file = VOCAB_FILES_NAMES['compressed_label_tokens_file']
    compressed_general_tokens_file = VOCAB_FILES_NAMES['compressed_general_tokens_file']

    label_tokens_file = self.get_decompressed_path(VOCAB_FILES_NAMES['compressed_label_tokens_file'])
    general_tokens_file = self.get_decompressed_path(VOCAB_FILES_NAMES['compressed_general_tokens_file'])

    self.ensure_decompressed(compressed_label_tokens_file, label_tokens_file)
    self.ensure_decompressed(compressed_general_tokens_file, general_tokens_file)

    label_tokens: List[str] = self.vocab_file_lines(label_tokens_file)
    general_tokens: List[str] = self.vocab_file_lines(general_tokens_file)

    tokens = chain(label_tokens, general_tokens, extra_tokens)

    super().__init__(
      eos_token=eos_token,
      unk_token=unk_token,
      pad_token=pad_token,
      tokens=tokens
    )
    

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