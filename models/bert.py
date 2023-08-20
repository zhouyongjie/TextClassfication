import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from configs import config


class BertClassification(nn.Module):
    def __init__(self):
        super(BertClassification, self).__init__()
        self.model_name = 'bert-base-chinese'
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden_outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = hidden_outputs.pooler_output
        output = self.fc(outputs)
        return output



