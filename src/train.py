# -*- encoding: utf-8 -*-
'''
@create_time: 2022/12/13 11:33:08
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
        self.label = data["clicks"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index) -> T_co:
        feature = torch.from_numpy(np.array([self.word2vec.wv[i] for i in self.sentences[index]]))
        type_ids = torch.from_numpy(np.array(self.type_ids[index])).to(torch.int64)
        label = torch.from_numpy(np.array(self.label[index][:1])).to(torch.int64)
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


def flat_f1(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='micro')



def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.info("start")
    word2vec = joblib.load("../notebook/word2vec.m")
    bt = time.time()
    train_data = pd.read_pickle("../notebook/train_data_sample.pkl")
    dev_data = pd.read_pickle("../notebook/dev_data_sample.pkl")
    dev_dataset = LRDataset(word2vec=word2vec, data=dev_data)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=160, collate_fn=collate_func)
    train_dataset = LRDataset(word2vec=word2vec, data=train_data)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=160, collate_fn=collate_func)
    print(f"load data cost {time.time()-bt}")

    epoch = 5
    model = LR()
    # optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=0)
    model.cuda()

    for e in range(epoch):  # Epoch
        logger.info('============= Epoch {:} / {:} =============='.format(e + 1, epoch))
        model.train()

        n_step_train_loss = 0
        total_train_loss = 0

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if step % 50 == 0 and step != 0:
                logger.info(f"loss: {n_step_train_loss/50}")
                n_step_train_loss = 0
            feature = batch["feature"].cuda()
            type_ids = batch["type_ids"].cuda()
            labels = batch["labels"].cuda()
            model.zero_grad()
            output = model(feature=feature, type_ids=type_ids, labels=labels)
            logits = output["logits"]
            loss = output["loss"]
            n_step_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        total_eval_f1 = 0
        total_eval_acc = 0
        total_eval_p = []
        total_eval_l = []

        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            if step >= 1000:
                break
            feature = batch["feature"].cuda()
            type_ids = batch["type_ids"].cuda()
            labels = batch["labels"].cuda()
            with torch.no_grad():
                output = model(feature=feature, type_ids=type_ids, labels=labels)
                logits = output["logits"].cpu().numpy()
                labels = labels.cpu().numpy()
                total_eval_f1 += flat_f1(logits, labels)

        avg_val_f1 = total_eval_f1 / len(dev_dataloader)
        logger.info('F1: {0:.2f}'.format(avg_val_f1))
        current_ckpt = 'lr-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-f1_' + str(int(avg_val_f1*100)) + '.pth'
        torch.save(model, current_ckpt)
    ...


if __name__ == "__main__":
    main()
    ...