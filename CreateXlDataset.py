from datasets import load_dataset
from transformers import AutoTokenizer, DataCollator
import jsonlines
import numpy as np


def get_input_ids(tokenizer):
    """
    Get input ids from the bookcorpusopen, arxiv, and openwebtext datasets.
    :param tokenizer:
    :return:
    """
    book_data = load_dataset("bookcorpusopen", split="train")
    arxiv_data = load_dataset("ccdv/arxiv-summarization", split="train")
    owt_data = load_dataset("openwebtext", split="train[:15%]")

    def tokenization(example):
        test_data = tokenizer(example["text"])
        if type(test_data['input_ids'][0]) == list:
            for i in range(len(test_data['input_ids'])):
                test_data['input_ids'][i].append(tokenizer.eos_token_id)
        else:
            test_data['input_ids'].append(tokenizer.eos_token_id)
        return test_data

    def arxiv_tokenization(example):
        test_data = tokenizer(example["article"])
        if type(test_data['input_ids'][0]) == list:
            for i in range(len(test_data['input_ids'])):
                test_data['input_ids'][i].append(tokenizer.eos_token_id)
        else:
            test_data['input_ids'].append(tokenizer.eos_token_id)
        return test_data

    arxiv_tokenized = arxiv_data.map(arxiv_tokenization, batched=True)
    book_tokenized = book_data.map(tokenization, batched=True)
    owt_tokenized = owt_data.map(tokenization, batched=True)
    return sum(sorted(book_tokenized['input_ids'] + owt_tokenized['input_ids'] + arxiv_tokenized['input_ids'],
                  key=lambda x: len(x)))


def create_xl_dataset(num_batches=64):
    """
    Create dataset to support Transformer XL and other long context length models.
    The important thing here is to make sure that the data is formatted correctly.
    Since long context models need to be trained on many different sequence lengths,
    we should sort the data by length and then batch it into different length groups.
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
    entire_sequence = get_input_ids(tokenizer)
    batch_length = len(entire_sequence) // 64
    print("Batch length: ", batch_length)
    batched_sequence = np.array(
        [np.array(entire_sequence[i * batch_length:(i + 1) *batch_length]) for i in range(0, len(entire_sequence) // batch_length)])
    np.save("xl_dataset.npy", batched_sequence)


if __name__ == '__main__':
    create_xl_dataset()
