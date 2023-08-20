import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from transformers import AutoTokenizer
from utils.data_utils import TouTiaoDataset
import numpy as np
import os
from tqdm import tqdm

from models.bert import BertClassification
from configs import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model: BertClassification, dev_loader, tokenizer, loss_function):
    model = model.eval()
    losses = []

    acc_metric = MulticlassAccuracy(num_classes=config.num_classes, device=device)
    with torch.no_grad():
        for data in dev_loader:

            inputs, labels = data
            batch_tokenized = tokenizer.batch_encode_plus(inputs,
                                                          add_special_tokens=True,
                                                          max_length=config.max_length,
                                                          # pad_to_max_length=True,
                                                          padding="max_length",
                                                          truncation="longest_first",
                                                          return_tensors='pt').to(device)
            outputs = model(**batch_tokenized)
            labels = torch.as_tensor(labels).to(device)
            loss = loss_function(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            acc_metric.update(labels, preds)
            losses.append(loss.item())
        acc = acc_metric.compute().item()
    return acc, np.mean(losses)


def trainer(train_loader, dev_loader, model, tokenizer, optimizer, loss_function):
    for epoch in range(config.epoch):
        print_avg_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data
            batch_tokenized = tokenizer.batch_encode_plus(inputs,
                                                          add_special_tokens=True,
                                                          max_length=config.max_length,
                                                          # pad_to_max_length=True,
                                                          padding="max_length",
                                                          truncation="longest_first",
                                                          return_tensors='pt').to(device)
            outputs = model(**batch_tokenized)
            labels = torch.as_tensor(labels).to(device)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_avg_loss += loss.item()

        accurate, dev_loss = eval_model(dev_loader=dev_loader,
                                        model=model,
                                        tokenizer=tokenizer,
                                        loss_function=loss_function)
        print("epoch: %d, train_loss:%.4f, dev_loss: %.4f, Accurate: %.4f" %
              (epoch, print_avg_loss, dev_loss, accurate))

        torch.save(model.state_dict(), os.path.join(config.save_model_path, f'model_epoch_{epoch}.pt'))


def main():

    print("----> 加载数据")
    train_dataset = TouTiaoDataset(config.train_data_path, is_train=True)
    dev_dataset = TouTiaoDataset(config.dev_data_path, is_train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=config.batch_size, shuffle=True)

    model = BertClassification().to(device)
    # for param_tensor in model.state_dict():
    #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    optimizer = Adam(model.parameters(), lr=config.lr)
    cross_loss = nn.CrossEntropyLoss()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    print("----> 开始训练")
    trainer(train_loader=train_loader,
            dev_loader=dev_loader,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss_function=cross_loss)


if __name__ == "__main__":
    main()
