from datasets import Dataset, load_dataset
from tqdm import tqdm
from settings import END_OF_TEXT, MAX_LENGTH, tokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import math
import statistics
import matplotlib.pyplot as plt


def process_wikipedia_dataset(dataset_split, output_name):
    # Load the Wikipedia Simple dataset
    dataset = load_dataset('wikipedia', dataset_split, beam_runner="DirectRunner")
    results = []

    for example in tqdm(dataset['train']):
        # print(example)
        sentences = sent_tokenize(example['text'])
        tokenized_sentences = [tokenizer(sentence, add_special_tokens=False, return_attention_mask=False) for sentence in sentences]
        token_count = sum([len(s) for s in tokenized_sentences])
        target_tokens = int(MAX_LENGTH * 1.5)
        target_chunks = math.ceil(token_count / target_tokens)
        target_tokens = int(token_count/target_chunks)

        # delta = 0
        current_chunk = []
        chunks = []

        for sentence in tokenized_sentences:
            if len(sentence) + len(current_chunk) > target_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []


                if len(sentence) > target_tokens:
                    chunks += [sentence[i:i + target_tokens] for i in range(0, len(sentence), target_tokens)]
                else:
                    chunks.append(sentence)
            else:
                current_chunk += sentence

        chunks.append(current_chunk)
        results += [{'input_ids': tokenizer(chunk, return_attention_mask=False)} for chunk in chunks]

        # tokenized_text = tokenizer(example['text'], return_attention_mask=False)
        # results += [{'input_ids': tokenized_text['input_ids'][i:i + MAX_LENGTH]} for i in range(0, len(tokenized_text['input_ids']), MAX_LENGTH)]

    # result_lengths = [len(chunk['input_ids']) for chunk in results]
    # print('mean:', statistics.mean(result_lengths), 'median:', statistics.median(result_lengths), 'mode:', statistics.mode(result_lengths))

    # Create histogram
    """
    plt.hist(result_lengths, bins=10)
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Numbers')
    plt.show()
    """


    # Tokenize and chunk the dataset
    dataset = Dataset.from_list(results)
    dataset = dataset.train_test_split(test_size=0.05)

    dataset.save_to_disk(output_name)

if __name__ == '__main__':
    process_wikipedia_dataset('20220301.simple', "prepared_wikipedia_simple")
    process_wikipedia_dataset('20220301.en', "prepared_wikipedia_en")
