from datasets import load_dataset
from settings import tokenizer, MAX_LENGTH

dataset = load_dataset("json", data_files="autocoder.jsonl")

# take the "response" column from the dataset, and tokenize it.
dataset = dataset.map(
    lambda examples: tokenizer(
        examples['response'], 
        truncation=True, 
        padding='do_not_pad', 
        max_length=MAX_LENGTH
    ), 
    batched=True, 
    remove_columns=["prompt", "response", "source"]
)

dataset.save_to_disk("prepared_autocoder")
