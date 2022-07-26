from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from .model import T5Booru
from argparse import ArgumentParser, Namespace
from boorupiece_simple.boorupiece import BooruPiece
from booru_chars_captions_lightning.booru_chars_captions import Tokenize, PadTokens, BooruCharsCaptions
from transformers.models.t5.configuration_t5 import T5Config

def main(args: Namespace) -> None:
  tokenizer = BooruPiece()
  t5_config = T5Config.from_pretrained('t5-small', vocab_size=tokenizer.token_registry.vocab_size)
  model = T5Booru(args, t5_config=t5_config)
  wandb_logger = WandbLogger(
    project="t5-booru-lightning",
    entity="mahouko"
    )
  trainer: Trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

  pad_tokens: PadTokens = tokenizer.pad_tokens
  tokenize: Tokenize = lambda labels: tokenizer.encode_tokens(tokenizer.tokenize_labels(labels))
  datamodule = BooruCharsCaptions(
    args,
    pad_tokens=pad_tokens,
    tokenize=tokenize,
  )
  trainer.fit(model, datamodule=datamodule)
  # trainer.test(model, datamodule=datamodule)

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