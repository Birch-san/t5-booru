from argparse import ArgumentParser, Namespace
from pytorch_lightning import LightningModule
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import Adafactor, get_adafactor_schedule
from torch import Tensor, LongTensor, FloatTensor
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Tuple, Optional#, Callable
#from typing_extensions import TypeAlias
from booru_chars_captions_lightning.booru_chars_captions import Batch

# IsPadToken: TypeAlias = Callable[[int], bool]

class T5Booru(LightningModule):
  model: T5ForConditionalGeneration
  # is_pad_token: IsPadToken
  pad_token_id: int
  learning_rate: int

  @staticmethod
  def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("T5Booru")
    parser.add_argument('--learning_rate', type=float, default=0.0, help='The initial learning rate for Adafactor')
    return parent_parser

  def __init__(
    self,
    args: Namespace,
    # is_pad_token: IsPadToken,
    pad_token_id: int,
    t5_config: T5Config,
    **kwargs,
  ) -> None:
    super().__init__()
    # self.is_pad_token = is_pad_token
    self.pad_token_id = pad_token_id
    self.model = T5ForConditionalGeneration(config=t5_config)
    self.learning_rate = args.learning_rate

  def forward(self, unmasked: LongTensor, masked: LongTensor) -> Tensor:
    attention_mask: LongTensor = (masked != self.pad_token_id).long()

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    # TODO: is there a way to replace this with a IsPadToken callback?
    masked[masked == self.pad_token_id] = -100

    # calculate loss
    output: Seq2SeqLMOutput = self.model.forward(
      input_ids=unmasked,
      attention_mask=attention_mask,
      labels=masked
    )
    loss: Tensor = output.loss

    return loss

  def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
    loss: Tensor = self(unmasked=batch.unmasked, masked=batch.masked)
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
