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
        self.batch = 8

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

        score = self.score(Q, D)
        return score 

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

        # mask out padding and punctuation
        mask = torch.tensor(self.mask(input_ids), device=D.device, dtype=D.dtype).unsqueeze(2) # (batch_size, context_maxlen) -> (batch_size, context_maxlen, 1) 
        D = D * mask # (batch_size, context_maxlen, dim)

        D = torch.nn.functional.normalize(D, p=2, dim=2) 

        return D

    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            # Q : (batch_size, question_maxlen, dim)
            # D : (batch_size, context_maxlen, dim)
            Q = Q.view(self.batch, 1, -1, self.dim) # (batch_size, 1, question_maxlen, dim)
            D = D.transpose(1, 2) # (batch_size, dim, context_maxlen)
            scores = torch.matmul(Q, D) # (batch_size, batch_size, question_maxlen, context_maxlen)
            max_scores = torch.max(scores, dim=3)[0] # (batch_size, batch_size, question_maxlen)
            sum_max_score = torch.sum(max_scores, dim=2) # (batch_size, batch_size)
            return sum_max_score
        
        else:
            raise NotImplementedError

