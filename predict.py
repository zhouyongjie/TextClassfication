import torch
from transformers import BertTokenizer

from models.bert import BertClassification
from utils.data_utils import idx2label


def predict(text):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertClassification()
    model.load_state_dict(torch.load('./result/model_epoch_1.pt'))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    text_tokenized = tokenizer.encode_plus(text, return_tensors='pt').to(device)

    outputs = model(**text_tokenized)

    _, preds = torch.max(outputs, dim=1)
    print(f"{text}的类别为：【{idx2label[preds.item()]}】")


input_text = '去日本游玩多出一天行程，可以去哪里？'
predict(input_text)

