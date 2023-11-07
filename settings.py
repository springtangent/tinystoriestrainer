from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat')
END_OF_TEXT = '<|endoftext|>'
MAX_LENGTH = 2048