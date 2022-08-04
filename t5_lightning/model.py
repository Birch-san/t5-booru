from argparse import ArgumentParser, Namespace
from pytorch_lightning import LightningModule
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import Adafactor, get_adafactor_schedule
from torch import Tensor, LongTensor
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Callable
from typing_extensions import TypeAlias
from booru_chars_captions_lightning.booru_chars_captions import Batch

TokenPredicate: TypeAlias = Callable[[int], bool]
IsPadToken: TypeAlias = TokenPredicate
IsNotPadToken: TypeAlias = TokenPredicate

class T5Booru(LightningModule):
  model: T5ForConditionalGeneration
  is_pad_token: IsPadToken
  is_not_pad_token: IsNotPadToken
  learning_rate: int
  captions_seen: int

  @staticmethod
  def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("T5Booru")
    parser.add_argument('--learning_rate', type=float, default=0.0, help='The initial learning rate for Adafactor')
    return parent_parser

  def __init__(
    self,
    args: Namespace,
    is_pad_token: IsPadToken,
    t5_config: T5Config,
    **kwargs,
  ) -> None:
    super().__init__()
    self.is_pad_token = is_pad_token
    self.is_not_pad_token = lambda token_id: ~is_pad_token(token_id)
    self.model = T5ForConditionalGeneration(config=t5_config)
    self.learning_rate = args.learning_rate
    self.captions_seen = 0
    self.save_hyperparameters()

  def forward(self, unmasked: LongTensor, masked: LongTensor) -> Tensor:
    attention_mask: LongTensor = self.is_not_pad_token(masked).long()

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    masked[self.is_pad_token(masked)] = -100

    # calculate loss
    output: Seq2SeqLMOutput = self.model.forward(
      input_ids=unmasked.to(self.device),
      attention_mask=attention_mask.to(self.device),
      labels=masked.to(self.device)
    )
    loss: Tensor = output.loss

    return loss

  def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
    loss: Tensor = self(unmasked=batch.unmasked, masked=batch.masked)
    batch_size = batch.masked.shape[0]
    self.captions_seen += batch_size
    self.log('train/loss', loss)
    self.log('captions_seen', self.captions_seen)
    return loss

  def configure_optimizers(self):
    params = list(filter(lambda p: p.requires_grad, self.parameters()))
    optimizer = Adafactor(params, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    scheduler = get_adafactor_schedule(optimizer, initial_lr=self.learning_rate)

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1,
      }
    }
