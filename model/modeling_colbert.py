import string

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizerFast


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, question_maxlen, context_maxlen, similarity_metric, dim, mask_punctuation):
        super(ColBERT, self).__init__(config)

        self.question_maxlen = question_maxlen
        self.context_maxlen = context_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}
        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True for symbol in string.punctuation for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]] if type(w) != int} # {'!': True, '"': True, '#': True, '$': True ... }

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)

        self.init_weights()

    def forward(self, q_inputs, c_inputs):
        Q = self.query(**q_inputs)
        D = self.doc(**c_inputs)

        self.score(Q, D)

    def mask(self, input_ids):
        # punctuation is not considered as part of the document
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.tolist()]
        return mask

    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0] # (batch_size, question_maxlen, hidden_size)
        Q = self.linear(Q) # (batch_size, question_maxlen, dim)

        return torch.nn.functional.normalize(Q, p=2, dim=2)
    
    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0] # (batch_size, context_maxlen, hidden_size)
        D = self.linear(D) # (batch_size, context_maxlen, dim)

        mask = torch.tensor(self.mask(input_ids)).unsqueeze(2).float() # (batch_size, context_maxlen) -> (batch_size, context_maxlen, 1) 
        D = D * mask # (batch_size, context_maxlen, dim)

        D = torch.nn.functional.normalize(D, p=2, dim=2) 

        return D

    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            print("Q shape: ", Q.shape) # (batch_size, question_maxlen, dim)
            print("D shape: ", D.shape) # (batch_size, context_maxlen, dim)
            sim_scores = Q @ D.permute(0, 2, 1) # (batch_size, question_maxlen, context_maxlen)
            max_score = sim_scores.max(dim=2).values # (batch_size, question_maxlen)
            sum_max_score = max_score.sum(dim=1) # (batch_size)
            print("sim_scores shape: ", sim_scores.shape)
            print("max_score shape: ", max_score.shape)
            print("sum_max_score shape: ", sum_max_score.shape)

            return sum_max_score
        
        else:
            raise NotImplementedError

