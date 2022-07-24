from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from .model import T5Booru
from argparse import ArgumentParser, Namespace
from os.path import join
from boorupiece.boorupiece import BooruPiece, VOCAB_FILES_NAMES
from booru_chars_captions_lightning.booru_chars_captions import Tokenize, BooruCharsCaptions, BooruCharsCaptionsDataset, BooruCharsCaptionsDatasetFactory
from transformers.models.t5.configuration_t5 import T5Config

def main(args: Namespace) -> None:
  wandb_logger = WandbLogger(
    project="t5-booru-lightning",
    entity="mahouko"
    )
  # tokenizer = AutoTokenizer.from_pretrained(
  #   cls='boorupiece',
  #   tokenizer_type='boorupiece',
  #   use_fast=False
  # )
  tokenizer = BooruPiece(
    compressed_general_tokens_file=join('boorupiece', VOCAB_FILES_NAMES['compressed_general_tokens_file']),
    compressed_label_tokens_file=join('boorupiece', VOCAB_FILES_NAMES['compressed_label_tokens_file']),
    extra_ids=0,
    )
  t5_config = T5Config.from_pretrained('t5-small', vocab_size=len(tokenizer))
  model = T5Booru(args, t5_config=t5_config)
  trainer: Trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

  # https://github.com/Lightning-AI/deep-learning-project-template
  # dataset = load_dataset(
  #   path='booru_chars_captions/booru_chars_captions.py',
  #   name='sqlite',
  #   precomputed_validation_split_percentage=5,
  #   sqlite_db_path=f"{environ['HOME']}/machine-learning/booru-chars/booru-chars.db"
  # )
  # train: IterableDataset = dataset['train']
  # validation: IterableDataset = dataset['validation']
  # trainer.fit(model, train, validation)
  # test: IterableDataset = dataset['test']
  # trainer.test(test_dataloaders=test)

  tokenize: Tokenize = lambda caption: tokenizer.encode(caption)
  dataset_factory: BooruCharsCaptionsDatasetFactory = lambda params: BooruCharsCaptionsDataset(**params, tokenize=tokenize)
  datamodule = BooruCharsCaptions(args, dataset_factory=dataset_factory)
  trainer.fit(model, datamodule=datamodule)
  # trainer.test(model, datamodule=datamodule)
  # TODO: actually run the tokenizer over these

if __name__ == "__main__":
  # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=add_model_specific_args#argparser-best-practices
  parser = ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=32)
  # parser.add_argument("--accelerator", default=None)
  parser = Trainer.add_argparse_args(parser)
  parser = BooruCharsCaptions.add_argparse_args(parser)
  parser = T5Booru.add_model_specific_args(parser)
  args = parser.parse_args()

  main(args)