from os.path import exists, join
from itertools import chain
from typing import Iterable, List
from .token_registry import PadTokenMixin, UnkTokenMixin, TokenRegistry
from io import TextIOWrapper
import gzip, shutil
from pathlib import Path

vocab_files = {
  'general_tokens': 'general_tokens.tsv',
  'label_tokens': 'label_tokens.tsv',
}
compressed_vocab_files = {
  name: f'{path}.gz' for name, path in vocab_files.items()
}

class BooruCharsTokenRegistry(UnkTokenMixin, PadTokenMixin, TokenRegistry):
  data_dir: Path
  def __init__(
    self,
    data_dir: Path = Path(__file__).resolve().parent,
    extra_tokens: Iterable[str] = [],
    **kwargs
  ) -> None:
    self.data_dir = data_dir
    for name, path in vocab_files.items():
      self.ensure_decompressed(
        self.qualify_asset_path(compressed_vocab_files[name]),
        self.qualify_asset_path(path)
      )

    label_tokens: List[str] = self.vocab_file_lines(self.qualify_asset_path(vocab_files['label_tokens']))
    general_tokens: List[str] = self.vocab_file_lines(self.qualify_asset_path(vocab_files['general_tokens']))

    tokens = chain(label_tokens, general_tokens, extra_tokens)

    super().__init__(
      tokens=tokens,
      **kwargs
    )
  
  def qualify_asset_path(self, filename: str)-> str:
    return join(self.data_dir, filename)

  @staticmethod
  def ensure_decompressed(compressed_file_path: str, decompressed_file_path: str) -> None:
    assert exists(compressed_file_path) or exists(decompressed_file_path)
    if exists(compressed_file_path) and not exists(decompressed_file_path):
      with gzip.open(compressed_file_path, 'r') as f_in, open(decompressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
  
  @staticmethod
  def vocab_file_lines(file_path: str) -> List[str]:
    file: TextIOWrapper = open(file_path, 'r')
    lines: list[str] = [cleaned for cleaned in (line.rstrip('\n') for line in file.readlines()) if cleaned != '']
    return lines