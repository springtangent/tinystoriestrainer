from transformers import GPTNeoForCausalLM, AutoTokenizer

TEMPERATURE = 0.8

model = GPTNeoForCausalLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")
tokenizer.pad_token = tokenizer.eos_token


def llm(prompt: str) -> str:
	# Now, let's generate a new text sequence
	input_ids = tokenizer.encode(prompt, return_tensors='pt')
	attention_mask = input_ids.ne(tokenizer.pad_token_id).int()

	# Generate text
	print('generating text')
	generated_text = model.generate(input_ids, attention_mask=attention_mask, max_length=2048, num_return_sequences=1, no_repeat_ngram_size=2, temperature=TEMPERATURE)

	# Decode the generated text
	return tokenizer.decode(generated_text[0], skip_special_tokens=True)


if __name__ == "__main__":
	prompt = 'Once upon a time'
	# prompt = """Sara and Ben fight a scary thing. They run to it. Sara throws a rock at it. Gwen pokes the scary thing with a fork, then"""
	print(llm(prompt))

