from transformers import GPTNeoConfig, GPTNeoForCausalLM, TextDataset, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset


def load_data(train_path, test_path, tokenizer):
    print('loading dataset')
    data = load_dataset('text', data_files={"train": train_path, "test": test_path})

    # tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    data = data.map(tokenize_function, batched=True, remove_columns=["text"])

    return data


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", eos_token='<|endoftext|>', bos_token='<|endoftext|>')
    data = load_data('TinyStoriesV2-GPT4-train.txt', 'TinyStoriesV2-GPT4-valid.txt', tokenizer)
    data.save_to_disk("train_dataset")


if __name__ == "__main__":
    main()
