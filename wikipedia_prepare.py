import argparse
from transformers import GPTNeoConfig, GPTNeoForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, Features, Value, Sequence, load_dataset
from tqdm import tqdm
from settings import END_OF_TEXT, MAX_LENGTH, tokenizer
# from apache_beam import DirectRunner



def process_wikipedia_dataset(dataset_split, output_name):
    # Load the Wikipedia Simple dataset
    dataset = load_dataset('wikipedia', '20220301.simple', beam_runner="DirectRunner")

    # Function to tokenize and chunk the text
    def tokenize_and_chunk(example):
        # Tokenize the text
        tokenized_text = tokenizer(example['text'], return_attention_mask=False)

        # Chunk tokenized text into segments of `chunk_size` tokens
        chunked_input_ids = [tokenized_text['input_ids'][i:i + MAX_LENGTH] for i in range(0, len(tokenized_text['input_ids']), MAX_LENGTH)]

        return {'input_ids': chunked_input_ids}

    # Tokenize and chunk the dataset
    dataset = dataset.map(tokenize_and_chunk, batched=True, remove_columns=dataset['train'].column_names)
    dataset = dataset['train'].train_test_split(test_size=0.05)

    dataset.save_to_disk("prepared_wikipedia_simple")

if __name__ == '__main__':
    process_wikipedia_dataset('20220301.simple', "prepared_wikipedia_simple")
    process_wikipedia_dataset('20220301.en', "prepared_wikipedia_en")
