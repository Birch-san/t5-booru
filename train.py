from run_mlm_no_trainer import main as trainer_main
from boorupiece import BooruPiece, BooruPieceConfig
from transformers import AutoConfig, AutoTokenizer

def main():
  AutoConfig.register('boorupiece', BooruPieceConfig)
  AutoTokenizer.register(BooruPieceConfig, BooruPiece)
  trainer_main()

if __name__ == "__main__":
  main()