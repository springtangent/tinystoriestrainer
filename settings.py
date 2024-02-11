from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
END_OF_TEXT = '<|endoftext|>'
MAX_LENGTH = 2048
