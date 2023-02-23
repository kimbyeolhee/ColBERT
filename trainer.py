import json
import logging
import os
import pickle

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



    def valid_per_epoch(self, wiki_dataloader, valid_dataloader, epoch):
        logger.info("***** Running Validation *****")

        # make context_embeddings
        c_embs = []
        with torch.no_grad():
            epoch_iterator = tqdm(wiki_dataloader, desc="Iteration")
            self.model.eval()

            for _, batch in enumerate(epoch_iterator):
                c_inputs = {"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device), "token_type_ids": batch[2].to(self.device)}
                outputs = self.model.doc(**c_inputs)
                c_embs.extend(outputs.cpu().numpy())
            c_embs = torch.Tensor(c_embs) # (num_wiki, 768)

        if not os.path.exists(f"{path}/saved_models/context_embedding/"):
            os.makedirs(f"{path}/saved_models/context_embedding/")
        with open(f"{path}/saved_models/context_embedding/epoch_{epoch+1}.bin", "wb") as f:
            pickle.dump(c_embs, f)
            
        # validation with valid_dataloader
        top_10 = 0

        with torch.no_grad():
            epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
            self.model.eval()

            for _, batch in enumerate(epoch_iterator):
                q_inputs = {"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device), "token_type_ids": batch[2].to(self.device)}
                c_inputs = {"input_ids": batch[3].to(self.device), "attention_mask": batch[4].to(self.device), "token_type_ids": batch[5].to(self.device)}
                q_embs = self.model.query(**q_inputs)

                # calculate cosine similarity
                score = torch.matmul(q_embs, c_embs.transpose(0, 1)) # (num_valid, num_wiki)
                rank = torch.argsort(score, dim=1, descending=True).squeeze() # (num_wiki)

                top_10_passages = [self.wiki_contexts[i] for i in rank[:10]]
                
                # top_k accuracy with c_inputs and top_10_passages
                for i in range(len(c_inputs["input_ids"])):
                    if c_inputs["input_ids"][i] in top_10_passages:
                        top_10 += 1
        
        return top_10 / len(valid_dataloader)
            

    def train(self):
        # set model to train mode
        self.model = self.model.to(self.device)
        self.model.train()

        # load dataloader
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.config.trainer.per_device_train_batch_size)
        wiki_dataloader = DataLoader(self.wiki_dataset, batch_size=self.config.trainer.per_device_train_batch_size)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.config.trainer.per_device_train_batch_size)

        best_top_10_acc = 0

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

            # valid per epoch
            top_10_acc = self.valid_per_epoch(wiki_dataloader, valid_dataloader, epoch)
            logger.info("Top 10 accuracy: {}".format(top_10_acc))


            if top_10_acc > best_top_10_acc:
                best_top_10_acc = top_10_acc
                self.model.save_pretrained(f"{path}/saved_models/best_model")
                logger.info("Best model saved on epoch {}".format(epoch+1))

            
