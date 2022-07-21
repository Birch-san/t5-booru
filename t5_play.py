from transformers import T5Tokenizer

small = T5Tokenizer.from_pretrained("google/t5-v1_1-small", from_slow=True)
# xxl = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

print(small.vocab_size)