import json
import logging
import os

import pandas as pd
from datasets import load_from_disk
from torch.utils.data import TensorDataset


path = os.path.dirname(os.path.abspath(__file__)) # /opt/ml/colbert/colbert
logger = logging.getLogger(__name__)


def make_dataframe(dataset):
    dataset = pd.DataFrame(
        {"context": dataset["context"], "question": dataset["question"]}
    )
    dataset = dataset.reset_index(drop=True)

    return dataset


def tokenize_func(dataset, tokenizer, mode):
    if mode == "train" or mode == "valid":
        preprocessed_question = []
        preprocessed_context = []
        for question, context in zip(dataset["question"], dataset["context"]):
            preprocessed_question.append("[Q] " + question)
            preprocessed_context.append("[D] " + context)
        tokenized_question = tokenizer(preprocessed_question, return_tensors="pt", padding=True, truncation=True, max_length=128)
        tokenized_context = tokenizer(preprocessed_context, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # change tokenize_questions's [PAD] token to [MASK] token
        mask_token_id = tokenizer.mask_token_id # 4
        pad_token_id = tokenizer.pad_token_id # 0

        tokenized_question["input_ids"][tokenized_question["input_ids"] == pad_token_id] = mask_token_id

        return tokenized_question, tokenized_context

    else:
        raise NotImplementedError


def load_dataset(config, tokenizer, mode):
    if mode == "train":
        if os.path.exists(config.data.train_path):
            logger.info(f"Loading train dataset from {config.data.train_path}")
            dataset = load_from_disk(f"{path}/{config.data.train_path}") # features: ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']
            dataset = make_dataframe(dataset)
            
            logger.info("Tokenizing train dataset")
            train_question, train_context = tokenize_func(dataset, tokenizer, mode)

            train_dataset = TensorDataset(train_question["input_ids"], train_question["attention_mask"], train_question["token_type_ids"], train_context["input_ids"], train_context["attention_mask"], train_context["token_type_ids"])
            return train_dataset

    elif mode == "valid":
        if os.path.exists(config.data.valid_path):
            logger.info(f"Loading valid dataset from {config.data.valid_path}")
            dataset = load_from_disk(config.data.valid_path)
            dataset = make_dataframe(dataset)

            logger.info("Tokenizing valid dataset")
            valid_question, valid_context = tokenize_func(dataset, tokenizer, mode)

            valid_dataset = TensorDataset(valid_question["input_ids"], valid_question["attention_mask"], valid_question["token_type_ids"], valid_context["input_ids"], valid_context["attention_mask"], valid_context["token_type_ids"])
            return valid_dataset

    elif mode == "test":
        if os.path.exists(config.data.test_path):
            logger.info(f"Loading test dataset from {config.data.test_path}")
            dataset = load_from_disk(config.data.test_path)

    elif mode == "wiki":
        if os.path.exists(config.data.test_path):
            logger.info(f"Loading wiki dataset from {config.data.test_path}")
            with open(f"{path}/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
                wiki = json.load(f)
            context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
            tokenized_context = tokenizer(context, padding="max_length", truncation=True, max_length=config.tokenizer.context_maxlen, return_tensors="pt")
            
            wiki_context_dataset = TensorDataset(tokenized_context["input_ids"], tokenized_context["attention_mask"], tokenized_context["token_type_ids"])
            return wiki_context_dataset


    else:
        raise ValueError("Invalid mode, mode is only available on train and valid")
