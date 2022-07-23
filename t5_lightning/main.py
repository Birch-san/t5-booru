from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from t5_lightning.model import T5Booru
from argparse import ArgumentParser, Namespace

def main(args: Namespace) -> None:
  wandb_logger = WandbLogger(
    project="t5-booru-lightning",
    entity="mahouko"
    )
  model = T5Booru()
  trainer: Trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

  # https://github.com/Lightning-AI/deep-learning-project-template
  train, val, test = ([], [], [])
  trainer.fit(model, train, val)
  trainer.test(test_dataloaders=test)

if __name__ == "__main__":
  # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=add_model_specific_args#argparser-best-practices
  parser = ArgumentParser()
  parser.add_argument("--accelerator", default=None)
  parser = Trainer.add_argparse_args(parser)
  parser = T5Booru.add_model_specific_args(parser)
  args = parser.parse_args()

  main(args)