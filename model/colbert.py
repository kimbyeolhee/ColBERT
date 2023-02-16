from transformers import BertModel, BertPreTrainedModel


class ColbertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert_model = BertModel(config)
    
    def forward(self):
        pass
