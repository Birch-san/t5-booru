from dataclasses import dataclass
import datasets
# from datasets import GeneratorBasedBuilder, BuilderConfig
from datasets.info import DatasetInfo
from datasets.utils import Version
from datasets.features import Features
from datasets.data_files import DataFilesDict
from datasets.download.download_manager import DownloadManager
from datasets.splits import SplitGenerator
from datasets import Split, NamedSplit
from typing import Optional, List, Iterator, Iterable, Tuple, NamedTuple, TypedDict
from typing_extensions import TypeAlias
from sqlite3 import Connection, Cursor
# from contextlib import closing
from .db import create_connection
from .booru_db import get_file_ids, file_ids_to_dtos, get_tags, BooruFileId
from more_itertools import partition

# https://huggingface.co/docs/datasets/v1.2.0/add_dataset.html
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py

@dataclass
class BooruCharsCaptionsConfig(datasets.BuilderConfig):
  name = "booru_chars_captions"
  version = Version("0.0.0")
  description = "BOORU CHARS OPEN DATASET captions"
  sqlite_db_path: Optional[str] = None
  precomputed_validation_split_percentage: Optional[int] = 5

Caption: TypeAlias = List[str]

class CaptionRecord(TypedDict):
  tags: Caption

class CaptionExample(NamedTuple):
  key: str
  record: CaptionRecord

class BooruCharsCaptions(datasets.GeneratorBasedBuilder):
  config: BooruCharsCaptionsConfig
  BUILDER_CONFIG_CLASS = BooruCharsCaptionsConfig
  conn: Connection

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.conn = create_connection(self.config.sqlite_db_path)

  def _info(self) -> DatasetInfo:
    """Construct the DatasetInfo object. See `DatasetInfo` for details.

    Warning: This function is only called once and the result is cached for all
    following .info() calls.

    Returns:
        info: (DatasetInfo) The dataset information
    """
    return DatasetInfo(
      description="BOORU CHARS OPEN DATASET captions",
      homepage="https://www.kaggle.com/datasets/printcraft/booru-chars-2022",
      license="CC0",
      builder_name="booru_chars_captions",
      features=Features({
        "tags": [datasets.Value(dtype='string')]
      })
    )
  
  def _split_generators(self, dl_manager: DownloadManager) -> List[SplitGenerator]:
    """Specify feature dictionary generators and dataset splits.

    This function returns a list of `SplitGenerator`s defining how to generate
    data and what splits to use.

    Example::

        return [
                datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={'file': 'train_data.zip'},
                ),
                datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={'file': 'test_data.zip'},
                ),
        ]

    The above code will first call `_generate_examples(file='train_data.zip')`
    to write the train data, then `_generate_examples(file='test_data.zip')` to
    write the test data.

    Datasets are typically split into different subsets to be used at various
    stages of training and evaluation.

    Note that for datasets without a `VALIDATION` split, you can use a
    fraction of the `TRAIN` data for evaluation as you iterate on your model
    so as not to overfit to the `TEST` data.

    For downloads and extractions, use the given `download_manager`.
    Note that the `DownloadManager` caches downloads, so it is fine to have each
    generator attempt to download the source data.

    A good practice is to download all data in this function, and then
    distribute the relevant parts to each split with the `gen_kwargs` argument

    Args:
        dl_manager: (DownloadManager) Download manager to download the data

    Returns:
        `list<SplitGenerator>`.
    """
    cur: Cursor = self.conn.cursor()
    file_ids: Cursor = get_file_ids(cur)
    file_id_dtos = file_ids_to_dtos(file_ids)
    if self.config.precomputed_validation_split_percentage is None:
      return [
        SplitGenerator(
          name=Split.TRAIN,
          gen_kwargs={'file_ids': file_id_dtos}
        )
      ]
    retain = lambda enumeration: enumeration[0] % 100 < self.config.precomputed_validation_split_percentage
    validation, training = partition(retain, enumerate(file_id_dtos))
    return [
      SplitGenerator(
        name=Split.TRAIN,
        gen_kwargs={'file_ids': map(lambda enumeration: enumeration[1], training)}
      ),
      SplitGenerator(
        name=Split.VALIDATION,
        gen_kwargs={'file_ids': map(lambda enumeration: enumeration[1], validation)}
      )
    ]

  # def _prepare_split(self, split_generator: SplitGenerator, **kwargs):
  #   """Generate the examples and record them on disk.

  #   Args:
  #       split_generator: `SplitGenerator`, Split generator to process
  #       **kwargs: Additional kwargs forwarded from _download_and_prepare (ex:
  #           beam pipeline)
  #   """
  #   raise NotImplementedError()
  
  def _generate_examples(self, file_ids: Iterable[BooruFileId], **kwargs) -> Iterator[CaptionExample]:
    for file_id in file_ids:
      BOORU, FID = file_id
      print(f'file_ids for {BOORU}, {FID}:')
      cur: Cursor = self.conn.cursor()
      tags: List[str] = get_tags(cur, file_id)
      print(f'len: {len(tags)}')
      print(tags)
      caption_record = CaptionRecord(tags=tags)
      yield CaptionExample(key=f'{BOORU}_{FID}', record=caption_record)
    """Default function generating examples for each `SplitGenerator`.

    This function preprocess the examples from the raw data to the preprocessed
    dataset files.
    This function is called once for each `SplitGenerator` defined in
    `_split_generators`. The examples yielded here will be written on
    disk.

    Args:
        **kwargs (additional keyword arguments): Arguments forwarded from the SplitGenerator.gen_kwargs

    Yields:
        key: `str` or `int`, a unique deterministic example identification key.
            * Unique: An error will be raised if two examples are yield with the
                same key.
            * Deterministic: When generating the dataset twice, the same example
                should have the same key.
            Good keys can be the image id, or line number if examples are extracted
            from a text file.
            The key will be hashed and sorted to shuffle examples deterministically,
            such as generating the dataset multiple times keep examples in the
            same order.
        example: `dict<str feature_name, feature_value>`, a feature dictionary
            ready to be encoded and written to disk. The example will be
            encoded with `self.info.features.encode_example({...})`.
    """