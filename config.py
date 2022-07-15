from transformers import T5Config
from tokenizer import BooruPieceTokenizer

def main() -> None:
  tokenizer = BooruPieceTokenizer()
  config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())

if __name__ == "__main__":
  main()