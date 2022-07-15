from transformers import BatchEncoding, TensorType
from transformers.tokenization_utils_base import TruncationStrategy, TextInput, PreTokenizedInput
from transformers.utils import PaddingStrategy
from typing import Optional, Union, List

MAX_LENGTH = 256

def BooruPiece():
  def __init__(self) -> None:
    self.model_max_length = 1
    self.eos_token_id = 0
    return
  
  def batch_encode_plus(
    self,
    texts,
    return_tensors: Optional[Union[str, TensorType]] = 'pt',
    padding: Union[bool, str, PaddingStrategy] = 'longest',
    max_length: Optional[int] = MAX_LENGTH,
    truncation: Union[bool, str, TruncationStrategy] = True
  ) -> BatchEncoding:
    return
  
  def __len__(self) -> int:
    return
  
  def __call__(
    self,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
    return_attention_mask: Optional[bool] = False
    ) -> BatchEncoding:
    return
  
  # def save_pretrained(self, dir: str) -> None:

  # and ideally need T5Tokenizer.from_pretrained(name) to find it