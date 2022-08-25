from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from .model import IsPadToken, T5Booru
from argparse import ArgumentParser, Namespace
from boorupiece_simple.boorupiece import BooruPiece
from booru_chars_captions_lightning.booru_chars_captions import PadTokens, BooruCharsCaptions, BooruCharsCaptionsDatasetFactory
from booru_chars_captions_lightning.booru_chars_caption_dataset import BooruCharsCaptionsDataset
from transformers.models.t5.configuration_t5 import T5Config

def main(args: Namespace) -> None:
  tokenizer = BooruPiece()
  t5_config = T5Config.from_pretrained(
    't5-small',
    vocab_size=tokenizer.token_registry.vocab_size,
    torch_dtype='bfloat16',
  )
  is_pad_token: IsPadToken = lambda token_id: token_id == tokenizer.token_registry.pad_token_id
  model = T5Booru.from_argparse_args(
    args,
    t5_config=t5_config,
    is_pad_token=is_pad_token,
    )
  wandb_logger = WandbLogger(
    project="t5-booru-lightning",
    entity="mahouko"
    )
  trainer: Trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

  pad_tokens: PadTokens = tokenizer.pad_tokens
  dataset_factory: BooruCharsCaptionsDatasetFactory = lambda params: BooruCharsCaptionsDataset.from_argparse_args(
    args,
    params=params,
  )
  datamodule = BooruCharsCaptions.from_argparse_args(
    args,
    pad_tokens=pad_tokens,
    dataset_factory=dataset_factory,
  )
  trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
  # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html?highlight=add_model_specific_args#argparser-best-practices
  parser = ArgumentParser()
  parser = Trainer.add_argparse_args(parser)
  parser = BooruCharsCaptions.add_argparse_args(parser)
  parser = BooruCharsCaptionsDataset.add_argparse_args(parser)
  parser = T5Booru.add_model_specific_args(parser)
  args = parser.parse_args()

  main(args)