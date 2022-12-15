# -*- encoding: utf-8 -*-
'''
@create_time: 2022/12/13 11:33:08
@author: lichunyu
'''
import os
import time

import numpy as np
import polars as pl
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import T_co
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer


MAX_LENGTH = 20


class LR(nn.Module):

    def __init__(self, num_labels=1855603) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.type_embedding = nn.Embedding(4, 100)  # 0: clicks, 1: carts, 2: orders, 3: mask
        self.fc = nn.Linear(100, self.num_labels)

    def forward(self, feature, type_ids, labels=None):
        type_embedding = self.type_embedding(type_ids)
        embedding = feature + type_embedding
        feature = torch.mean(embedding, dim=-1)
        logits = self.fc(feature)
        if self.training and labels:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}


class LRDataset(Dataset):

    def __init__(self, word2vec, data) -> None:
        super().__init__()
        self.word2vec = word2vec
        self.sentences = data["sample_sentences"]
        self.type_ids = data["sample_type_ids"]
        self.label = data["clicks"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index) -> T_co:
        feature = torch.from_numpy(np.array([self.word2vec.wv[i] for i in self.sentences[index]]))
        type_ids = torch.from_numpy(np.array(self.type_ids[index]))
        label = torch.from_numpy(np.array(self.label[index][:1]))
        return {
            "feature": feature,
            "type_ids": type_ids,
            "labels": label
        }


def collate_func(batch):
    # torch.nn.functional.pad(batch[0]["feature"], (0,0,0,20,0,0))

    feature = torch.cat([F.pad(i["feature"].unsqueeze(0), (0, 0, 0, MAX_LENGTH-i["feature"].shape[-2])) for i in batch], dim=0)
    type_ids = torch.cat([F.pad(i["type_ids"], (0, MAX_LENGTH-i["type_ids"].shape[-1]), value=3).unsqueeze(0) for i in batch], dim=0)
    label = torch.cat([i["labels"] for i in batch], dim=0)
    return {
        "feature": feature,
        "type_ids": type_ids,
        "labels": label
    }


def main():
    word2vec = joblib.load("../notebook/word2vec.m")
    bt = time.time()
    # train_data = pd.read_pickle("../notebook/train_data_sample.pkl")
    dev_data = pd.read_pickle("../notebook/dev_data_sample.pkl")
    print(f"load data cost {time.time()-bt}")
    dev_dataset = LRDataset(word2vec=word2vec, data=dev_data)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=16, collate_fn=collate_func)
    for i in dev_dataloader:
        ...
    ...


if __name__ == "__main__":
    main()
    ...