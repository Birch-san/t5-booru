from argparse import ArgumentParser, Namespace
from pytorch_lightning import LightningModule
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import Adafactor, get_adafactor_schedule
from torch import Tensor, LongTensor, FloatTensor
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Tuple, Optional, TypedDict, List
from booru_chars_captions_lightning.booru_chars_captions import Tokenized

# class Batch(TypedDict):
#   source: LongTensor
#   target: LongTensor

class T5Booru(LightningModule):
  model: T5ForConditionalGeneration
  learning_rate: int

  @staticmethod
  def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("T5Booru")
    parser.add_argument('--learning_rate', type=float, default=0.0, help='The initial learning rate for Adafactor')
    return parent_parser

  def __init__(
    self,
    args: Namespace,
    t5_config: T5Config,
    **kwargs,
  ) -> None:
    super().__init__()
    self.model = T5ForConditionalGeneration(config=t5_config)
    self.learning_rate = args.learning_rate
  
  def _encode(self, input_ids: LongTensor, output_attentions: bool) -> Tuple[LongTensor, Optional[FloatTensor]]:
    encoded: BaseModelOutput = self.model.encoder(input_ids=input_ids, output_attentions=output_attentions)
    if (output_attentions):
      attentions: Tuple[FloatTensor, ...] = encoded.attentions
      return encoded.last_hidden_state, attentions
    return encoded.last_hidden_state, None
  
  def _encodeWithAttention(self, input_ids: LongTensor) -> Tuple[LongTensor, FloatTensor]:
    return self._encode(input_ids=input_ids, output_attentions=True)
  
  def _encodeWithoutAttention(self, input_ids: LongTensor) -> LongTensor:
    last_hidden_state, _ = self._encode(input_ids=input_ids, output_attentions=False)
    return last_hidden_state

  def forward(self, masked: LongTensor, unmasked: LongTensor) -> Tensor:
    source, source_mask = self._encodeWithAttention(masked)
    target: LongTensor = self._encodeWithoutAttention(unmasked)

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    target[target == self.tokenizer.pad_token_id] = -100

    # calculate loss
    output: Seq2SeqLMOutput = self.model(input_ids=source, attention_mask=source_mask, labels=target)
    loss: Tensor = output.loss

    return loss

  def training_step(self, batch: List[Tokenized], batch_idx: int) -> Tensor:
    # TODO: dropout (first batch we saw was already 96 tokens long; we should consider getting this down to 32)
    # TODO: turn List[Tokenized] into LongTensor
    # TODO: it's probably wasteful to pad it as early as we do (i.e. in DataLoader);
    #       we could do the padding here, when the time comes to tensorize it
    #       (and when we know how much dropout is desired)
    unmasked: LongTensor = batch
    # TODO: some function over unmasked, to splice out ~8 tokens, and pad the end of the list
    masked: LongTensor = unmasked
    loss: Tensor = self(masked, unmasked)
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
