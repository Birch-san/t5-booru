from run_mlm_no_trainer import main as trainer_main
from boorupiece import BooruPiece, BooruPieceConfig
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
import sys

from tokenizers.implementations.base_tokenizer import BaseTokenizer

def main():
  AutoConfig.register(BooruPieceConfig.model_type, BooruPieceConfig)
  AutoTokenizer.register(BooruPieceConfig, BooruPiece)
  TOKENIZER_MAPPING_NAMES.update({ f'{BooruPieceConfig.model_type}': (BooruPiece.__name__, None) })
  sys.modules[f'transformers.models.{BooruPieceConfig.model_type}'] = sys.modules[BooruPieceConfig.model_type]
  # TODO: tell T5Config to use bfloat16
  # TODO: https://github.com/Ki6an/fastT5
  trainer_main()



if __name__ == "__main__":
  main()