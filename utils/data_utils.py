import pandas as pd
from torch.utils.data import Dataset
from configs import config


idx2label = {value: key for key, value in config.label2idx.items()}


class TouTiaoDataset(Dataset):
    def __init__(self, file_path, is_train=True):
        df = pd.read_csv(file_path, encoding='utf-8')
        split_num = 2000 if is_train else 500
        self.text_list = df['Text'][:split_num]
        self.label_list = df['Label'][:split_num]

    def __getitem__(self, item):
        label = config.label2idx.get(self.label_list[item])

        return self.text_list[item], label

    def __len__(self):
        return len(self.label_list)


