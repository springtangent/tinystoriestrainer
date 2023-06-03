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

def tokenize_stories(stories: list[str], tokenizer: AutoTokenizer):
    for i in tqdm(range(0, len(stories), BATCH_SIZE), desc="Tokenizing"):
        batch = stories[i:i + BATCH_SIZE]
        tokenized = tokenizer(batch, truncation=True, padding='max_length', max_length=MAX_LENGTH)
        yield tokenized

def create_dataset(path: str, tokenizer: AutoTokenizer):
    stories = list(tqdm(load_stories(path), desc="Loading Stories"))
    dataset = Dataset.from_dict({'text': stories})
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True, remove_columns=["text"])
    return tokenized_dataset

def load_data(train_path, valid_path, tokenizer):
    return DatasetDict({
        'train': create_dataset(train_path, tokenizer),
        'valid': create_dataset(valid_path, tokenizer)
    })

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", eos_token=END_OF_TEXT, bos_token=END_OF_TEXT)
    tokenizer.pad_token = tokenizer.eos_token

    data = load_data(TRAIN_PATH, VALID_PATH, tokenizer)
    data.save_to_disk("train_dataset")

if __name__ == "__main__":
    main()
