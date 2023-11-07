import argparse
from transformers import GPTNeoConfig, GPTNeoForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm
from settings import END_OF_TEXT, MAX_LENGTH, tokenizer


TRAIN_PATH = 'TinyStoriesV2-GPT4-train.txt'
VALID_PATH = 'TinyStoriesV2-GPT4-valid.txt'


# Define the features of your dataset
features = Features({
    'input_ids': Sequence(feature=Value(dtype='int32')),
    'attention_mask': Sequence(feature=Value(dtype='int32')),
    # Define other features if you have them
})


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


def create_dataset(path: str, padding_option):
    stories = list(tqdm(load_stories(path), desc="Loading Stories"))
    dataset = Dataset.from_list([{'text': text} for text in stories])
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding=padding_option, max_length=MAX_LENGTH), batched=True, remove_columns=["text"])
    dataset.cast(features)
    return dataset


def load_data(train_path, valid_path, padding_option):
    return DatasetDict({
        'valid': create_dataset(valid_path, padding_option),
        'train': create_dataset(train_path, padding_option)
    })


def main(padding_option):
    if padding_option != 'do_not_pad':
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(TRAIN_PATH, VALID_PATH, padding_option)
    data.save_to_disk("prepared_tinystories2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", help="padding option: 'max_length', 'longest', 'do_not_pad'", default='do_not_pad')
    args = parser.parse_args()

    main(args.padding)
