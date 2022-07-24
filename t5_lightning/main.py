from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from .model import T5Booru
from argparse import ArgumentParser, Namespace
from datasets import IterableDataset, load_dataset
from os import environ

def main(args: Namespace) -> None:
  wandb_logger = WandbLogger(
    project="t5-booru-lightning",
    entity="mahouko"
    )
  model = T5Booru(args)
  trainer: Trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

  # https://github.com/Lightning-AI/deep-learning-project-template
  dataset = load_dataset(
    path='booru_chars_captions/booru_chars_captions.py',
    name='sqlite',
    precomputed_validation_split_percentage=5,
    sqlite_db_path=f"{environ['HOME']}/machine-learning/booru-chars/booru-chars.db"
  )
  train: IterableDataset = dataset['train']
  validation: IterableDataset = dataset['validation']
  # TODO: actually run the tokenizer over these
  trainer.fit(model, train, validation)
  # test: IterableDataset = dataset['test']
  # trainer.test(test_dataloaders=test)

if __name__ == "__main__":
  # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=add_model_specific_args#argparser-best-practices
  parser = ArgumentParser()
  # parser.add_argument("--accelerator", default=None)
  parser = Trainer.add_argparse_args(parser)
  parser = T5Booru.add_model_specific_args(parser)
  args = parser.parse_args()

  main(args)