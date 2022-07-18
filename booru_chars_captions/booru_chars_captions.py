from dataclasses import dataclass
import datasets
# from datasets import GeneratorBasedBuilder, BuilderConfig
from datasets.info import DatasetInfo
from datasets.utils import Version
from datasets.features import Features
from datasets.data_files import DataFilesDict
from datasets.download.download_manager import DownloadManager
from datasets.splits import SplitGenerator
from typing import Optional, Union

# https://huggingface.co/docs/datasets/v1.2.0/add_dataset.html
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py


@dataclass
class BooruCharsCaptionsConfig(datasets.BuilderConfig):
  name = "booru_chars_captions"
  version = Version("0.0.0")
  description = "BOORU CHARS OPEN DATASET captions"
  sqlite_db_path: Optional[str] = None

class BooruCharsCaptions(datasets.GeneratorBasedBuilder):
  config: BooruCharsCaptionsConfig
  BUILDER_CONFIG_CLASS = BooruCharsCaptionsConfig
  # def __init__(
  #   self,
  #   cache_dir: Optional[str] = None,
  #   config_name: Optional[str] = None,
  #   hash: Optional[str] = None,
  #   base_path: Optional[str] = None,
  #   info: Optional[DatasetInfo] = None,
  #   features: Optional[Features] = None,
  #   use_auth_token: Optional[Union[bool, str]] = None,
  #   repo_id: Optional[str] = None,
  #   data_files: Optional[Union[str, list, dict, DataFilesDict]] = None,
  #   data_dir: Optional[str] = None,
  #   name="deprecated",
  #   **config_kwargs,
  # ) -> None:
  #   super().__init__(
  #     cache_dir,
  #     config_name,
  #     hash,
  #     base_path,
  #     info,
  #     features,
  #     use_auth_token,
  #     repo_id,
  #     data_files,
  #     data_dir,
  #     name,
  #     **config_kwargs
  #   )

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
      builder_name="booru_chars_captions"
    )
  
  def _split_generators(self, dl_manager: DownloadManager):
    print(self.config.sqlite_db_path)
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
    raise NotImplementedError()

  # def _prepare_split(self, split_generator: SplitGenerator, **kwargs):
  #   """Generate the examples and record them on disk.

  #   Args:
  #       split_generator: `SplitGenerator`, Split generator to process
  #       **kwargs: Additional kwargs forwarded from _download_and_prepare (ex:
  #           beam pipeline)
  #   """
  #   raise NotImplementedError()
  
  def _generate_examples(self, **kwargs):
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
    raise NotImplementedError()