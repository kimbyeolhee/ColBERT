import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments

from data_loader import load_dataset
from model.modeling_colbert import ColBERT
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed


path = os.path.dirname(os.path.abspath(__file__)) # /opt/ml/colbert/colbert
sys.path.append(path)

def main(args):
    config = OmegaConf.load(f"{path}/{args.config}")
    init_logger()
    set_seed(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(config)


    # load model
    model = ColBERT.from_pretrained(config.model.name_or_path, 
                                    question_maxlen=config.tokenizer.question_maxlen, 
                                    context_maxlen=config.tokenizer.context_maxlen,
                                    similarity_metric=config.model.similarity_metric,
                                    dim=config.model.dim,
                                    mask_punctuation=config.model.mask_punctuation)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.mode == "train":
        train_dataset = load_dataset(config, tokenizer, "train")
        valid_dataset = load_dataset(config, tokenizer, "valid")
        wiki_dataset =  load_dataset(config, tokenizer, "wiki")

        args = TrainingArguments(
            output_dir=config.trainer.output_dir,
            evaluation_strategy=config.trainer.evaluation_strategy,
            per_device_train_batch_size=config.trainer.per_device_train_batch_size,
            per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
            num_train_epochs=config.trainer.num_train_epochs,
            weight_decay=config.trainer.weight_decay,
            learning_rate=config.trainer.learning_rate,
        )

        trainer = Trainer(args, config, train_dataset=train_dataset, valid_dataset=valid_dataset, wiki_dataset=wiki_dataset ,model=model, device=device)
        trainer.train()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config.yaml")
    parser.add_argument("--mode", "-m", type=str, default="train", choices=["train", "test"])

    args = parser.parse_args()
    main(args)
