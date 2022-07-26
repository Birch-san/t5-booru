from os.path import exists
from itertools import chain
from typing import Iterable, List
from .token_registry import TokenRegistryWithWellKnownTokens
from io import TextIOWrapper
import gzip, shutil

vocab_files = {
  'general_tokens': 'general_tokens.tsv',
  'label_tokens': 'label_tokens.tsv',
}
compressed_vocab_files = {
  name: f'{path}.gz' for name, path in vocab_files.items()
}

class BooruCharsTokenRegistry(TokenRegistryWithWellKnownTokens):
  def __init__(
    self,
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    extra_tokens: Iterable[str] = [],
  ) -> None:
    for name, path in vocab_files.items():
      self.ensure_decompressed(compressed_vocab_files[name], path)

    label_tokens: List[str] = self.vocab_file_lines(vocab_files['label_tokens'])
    general_tokens: List[str] = self.vocab_file_lines(vocab_files['general_tokens'])

    tokens = chain(label_tokens, general_tokens, extra_tokens)

    super().__init__(
      eos_token=eos_token,
      unk_token=unk_token,
      pad_token=pad_token,
      tokens=tokens
    )

  @staticmethod
  def ensure_decompressed(compressed_file_path: str, decompressed_file_path: str) -> None:
    if exists(compressed_file_path) and not exists(decompressed_file_path):
      with gzip.open(compressed_file_path, 'r') as f_in, open(decompressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
  
  @staticmethod
  def vocab_file_lines(file_path: str) -> List[str]:
    file: TextIOWrapper = open(file_path, 'r')
    lines: list[str] = [cleaned for cleaned in (line.rstrip('\n') for line in file.readlines()) if cleaned != '']
    return lines