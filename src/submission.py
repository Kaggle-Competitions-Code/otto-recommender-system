# -*- encoding: utf-8 -*-
'''
@create_time: 2022/12/20 13:37:53
@author: lichunyu
'''
import os
import time
import sys
import datetime
import logging

from tqdm import tqdm
import numpy as np
import polars as pl
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.dataloader import T_co
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, set_seed
from sklearn.metrics import f1_score, accuracy_score, classification_report, cohen_kappa_score


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAX_LENGTH = 20

set_seed(42)
 

class LR(nn.Module):

    def __init__(self, num_labels=1855603) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.type_embedding = nn.Embedding(4, 100)  # 0: clicks, 1: carts, 2: orders, 3: mask
        self.fc = nn.Linear(100, self.num_labels)

    def forward(self, feature, type_ids, labels=None):
        type_embedding = self.type_embedding(type_ids)
        embedding = feature + type_embedding
        feature = torch.mean(embedding, dim=-2)
        logits = self.fc(feature)
        if self.training and labels is not None:
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
        # self.label = data["clicks"]

        self.session = data["session"]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index) -> T_co:
        feature = torch.from_numpy(np.array([self.word2vec.wv[i] for i in self.sentences[index]]))
        type_ids = torch.from_numpy(np.array(self.type_ids[index])).to(torch.int64)
        # label = torch.from_numpy(np.array(self.label[index][:1])).to(torch.int64)
        session = torch.from_numpy(np.array([self.session[index]]))
        return {
            "feature": feature,
            "type_ids": type_ids,
            "session": session
        }


def collate_func(batch):
    # torch.nn.functional.pad(batch[0]["feature"], (0,0,0,20,0,0))

    feature = torch.cat([F.pad(i["feature"].unsqueeze(0), (0, 0, 0, MAX_LENGTH-i["feature"].shape[-2])) for i in batch], dim=0)
    type_ids = torch.cat([F.pad(i["type_ids"], (0, MAX_LENGTH-i["type_ids"].shape[-1]), value=3).unsqueeze(0) for i in batch], dim=0)
    session = torch.cat([i["session"] for i in batch], dim=0)
    return {
        "feature": feature,
        "type_ids": type_ids,
        "session": session
    }


def ensemble(preds:pd.DataFrame):
    submission = pd.read_csv("../data/submission-lb576.csv")
    submission["is_click"] = submission["session_type"].apply(lambda x: 1 if "_clicks" in x else 0)
    submission = submission[submission["is_click"]==0].reset_index(drop=True)[["session_type", "labels"]]
    submission = pd.concat([preds, submission]).reset_index(drop=True)
    submission.to_csv("submission_lr.csv", index=False)
    ...


def main():
    model: LR = torch.load("lr-2022-12-20-16-14-38-f1_1.pth")
    # model: LR = None
    test_data = pd.read_pickle("../notebook/test_data_sample.pkl")
    word2vec = joblib.load("../notebook/word2vec.m")
    test_dataset = LRDataset(word2vec=word2vec, data=test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=collate_func)
    data4df = {
        "session_type": [],
        "labels": []
    }
    model.eval()
    logits_list = []
    session_list = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        feature = batch["feature"].cuda()
        type_ids = batch["type_ids"].cuda()
        session = batch["session"].numpy()
        with torch.no_grad():
            logits = model(feature=feature, type_ids=type_ids)["logits"]
            logits = logits.cpu().detach().clone().numpy()
            # session_list.append(session)
            # logits_list.append(logits)
        data4df["session_type"].extend([str(i)+"_clicks" for i in session])
        data4df["labels"].extend(logits.argmax(-1).tolist())
    preds = pd.DataFrame(data4df)
    ensemble(preds=preds)
    ...



if __name__ == "__main__":
    main()
    # ensemble()
    ...