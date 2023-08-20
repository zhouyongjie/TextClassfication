from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch

device = torch.device('cuda:0')


class BertClassification(nn.Module):
    def __init__(self):
        super(BertClassification, self).__init__()
        self.model_name = 'bert-base-chinese'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, 6)  # 768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters

    # def forward(self, x):  # 这里的输入是一个list
    #     batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
    #                                                        max_length=128,
    #                                                        pad_to_max_length=True)  # tokenize、add special token、pad
    #     input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
    #     attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
    #     hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
    #     outputs = hiden_outputs[0][:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
    #     output = self.fc(outputs)
    #     return output
    def forward(self, input_ids, attention_mask):  # 这里的输入是一个list
        # batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
        #                                                    max_length=128,
        #                                                    pad_to_max_length=True)  # tokenize、add special token、pad
        # input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        # attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = hiden_outputs[0][:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(outputs)
        return output

