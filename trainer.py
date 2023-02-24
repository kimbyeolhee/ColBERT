import json
import logging
import os

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup


path = os.path.dirname(os.path.abspath(__file__)) # /opt/ml/colbert/colbert
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args, config, train_dataset=None, valid_dataset=None, wiki_dataset=None, test_dataset=None, model=None, device=None):
        self.args = args
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.wiki_dataset = wiki_dataset
        self.test_dataset = test_dataset
        self.device = device

        self.model = model

        with open("./data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)
        self.wiki_contexts = list(dict.fromkeys(w["text"] for w in wiki.values()))
        valid_data = load_from_disk("./data/train_dataset/validation")
        self.valid_contexts = valid_data["context"]      

    def train_per_epoch(self, epoch_iterator, optimizer, scheduler):
        batch_loss = 0

        for _, batch in enumerate(epoch_iterator):
            self.model.train()

            q_inputs = {"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device), "token_type_ids": batch[2].to(self.device)}
            c_inputs = {"input_ids": batch[3].to(self.device), "attention_mask": batch[4].to(self.device), "token_type_ids": batch[5].to(self.device)}

            outputs = self.model(q_inputs, c_inputs)

            # label: position of positive == diagonal of matrix
            labels = torch.arange(0, self.args.per_device_train_batch_size).long().to(self.device)

            loss = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(loss, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            self.model.zero_grad()
            
            batch_loss += loss

        return batch_loss / len(epoch_iterator)
            

    def train(self):
        best_loss = 1e9

        # set model to train mode
        self.model = self.model.to(self.device)
        self.model.train()

        # load dataloader
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.config.trainer.per_device_train_batch_size)
        DataLoader(self.wiki_dataset, batch_size=self.config.trainer.per_device_train_batch_size)
        DataLoader(self.valid_dataset, batch_size=self.config.trainer.per_device_train_batch_size)


        # load optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if all(nd not in n for nd in no_decay)
                ],
                "weight_decay": self.config.trainer.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        
        # start training
        self.model.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            # train per epoch
            train_loss = self.train_per_epoch(epoch_iterator, optimizer, scheduler)
            logger.info("Train loss: {}".format(train_loss))

            # save model if loss is lower than best loss
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model(f"{path}/saved_model/best_model_{epoch}.pt")
