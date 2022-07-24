from argparse import ArgumentParser, Namespace
from pytorch_lightning import LightningModule
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import Adafactor, get_adafactor_schedule
from torch import Tensor, LongTensor, FloatTensor
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Tuple, Optional, TypedDict

from boorupiece.boorupiece import BooruPiece, VOCAB_FILES_NAMES

class Batch(TypedDict):
  source: LongTensor
  target: LongTensor

class T5Booru(LightningModule):
  model: T5ForConditionalGeneration
  tokenizer: BooruPiece
  @staticmethod
  def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("T5Booru")
    parser.add_argument('--learning_rate', type=float, default=0.0, help='The initial learning rate for Adafactor')
    return parent_parser

  def __init__(
    self,
    args: Namespace,
    **kwargs,
  ) -> None:
    super().__init__()
    # self.tokenizer = AutoTokenizer.from_pretrained(
    #   cls='boorupiece',
    #   tokenizer_type='boorupiece',
    #   use_fast=False
    # )
    # TODO we may not need this to be a property of our model; perhaps the dataset we receive could arrive already-tokenized
    self.tokenizer = BooruPiece(
      compressed_general_tokens_file=f"boorupiece/{VOCAB_FILES_NAMES['compressed_general_tokens_file']}",
      compressed_label_tokens_file=f"boorupiece/{VOCAB_FILES_NAMES['compressed_label_tokens_file']}",
      extra_ids=0,
    )
    config = T5Config.from_pretrained('t5-small', vocab_size=len(self.tokenizer))
    self.model = T5ForConditionalGeneration(config=config)
  
  def _encode(self, input_ids: LongTensor, output_attentions: bool) -> Tuple[LongTensor, Optional[FloatTensor]]:
    encoded: BaseModelOutput = self.model.encoder(input_ids=input_ids, output_attentions=output_attentions)
    if (output_attentions):
      attentions: Tuple[FloatTensor, ...] = encoded.attentions
      return encoded.last_hidden_state, attentions
    return encoded.last_hidden_state, None
  
  def encodeWithAttention(self, input_ids: LongTensor) -> Tuple[LongTensor, FloatTensor]:
    return self._encode(input_ids=input_ids, output_attentions=True)
  
  def encodeWithoutAttention(self, input_ids: LongTensor) -> LongTensor:
    last_hidden_state, _ = self._encode(input_ids=input_ids, output_attentions=False)
    return last_hidden_state

  def forward(self, batch: Batch) -> Tensor:
    source, source_mask = self.encodeWithAttention(batch['source'])
    target: LongTensor = self.encodeWithoutAttention(batch['target'])

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    target[target == self.tokenizer.pad_token_id] = -100

    # calculate loss
    output: Seq2SeqLMOutput = self.model(input_ids=source, attention_mask=source_mask, labels=target)
    loss: Tensor = output.loss

    return loss

  # TODO: figure out how to get from IterableDataset iterand to Batch
  #       notably, we haven't:
  #       - regularized the caption (see BooruPiece TODO)
  #       - tokenized the caption
  #       - this will give us 'target' but will not give us 'source'
  #       - make 'source' by splicing away 8 tokens (e.g. same as number of attention heads?)
  #       - pad all tokenized captions in source and target with pad_token_id, up to the length of the longest tokenized caption in target
  def training_step(self, batch, batch_idx: int) -> Tensor:
    loss: Tensor = self(batch)
    return loss

  def configure_optimizers(self):
    params = list(filter(lambda p: p.requires_grad, self.parameters()))
    optimizer = Adafactor(params, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    scheduler = get_adafactor_schedule(optimizer, initial_lr=self.hparams.learning_rate)

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1,
      }
    }
