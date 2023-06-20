import argparse
from transformers import GPTNeoConfig, GPTNeoForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from settings import END_OF_TEXT, MAX_LENGTH


TRAIN_PATH = 'TinyStoriesV2-GPT4-train.txt'
VALID_PATH = 'TinyStoriesV2-GPT4-valid.txt'
BATCH_SIZE = 4096


def load_stories(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        story = []
        for line in f:
            line = line.strip()
            if line == END_OF_TEXT:
                yield '\n'.join(story)
                story = []
            else:
                story.append(line)
        if story:  # handle last story in file
            yield '\n'.join(story)


def create_dataset(path: str, tokenizer: AutoTokenizer, padding_option):
    stories = list(tqdm(load_stories(path), desc="Loading Stories"))
    dataset = Dataset.from_dict({'text': stories})
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding=padding_option, max_length=MAX_LENGTH), batched=True, remove_columns=["text"])
    return tokenized_dataset


def load_data(train_path, valid_path, tokenizer, padding_option):
    return DatasetDict({
        'train': create_dataset(train_path, tokenizer, padding_option),
        'valid': create_dataset(valid_path, tokenizer, padding_option)
    })


def main(padding_option):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", eos_token=END_OF_TEXT, bos_token=END_OF_TEXT)
    if padding_option != 'do_not_pad':
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(TRAIN_PATH, VALID_PATH, tokenizer, padding_option)
    data.save_to_disk("train_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", help="padding option: 'max_length', 'longest', 'do_not_pad'", default='max_length')
    args = parser.parse_args()

    main(args.padding)
