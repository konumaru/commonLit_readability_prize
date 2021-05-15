import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, RobertaTokenizer

from dataset import CommonLitDataModule
from model import CommonLitBertModel, CommonLitModel, CommonLitRoBERTaModel


class CommonLitTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=200):
        self.excerpt = data[["excerpt"]].to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, idx):
        text = str(self.excerpt[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # token_type_ids = inputs["token_type_ids"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


def get_test_dataloader():
    test = pd.read_csv("../data/raw/test.csv")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    test_dataset = CommonLitTestDataset(test, tokenizer, 200)
    return DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )


def predict():
    dataloader = get_test_dataloader()

    checkpoints = [
        "../tb_logs/Baseline/version_0/checkpoints/epoch=28-loss=0.0000-val_loss=0.5166.ckpt",
        "../tb_logs/Baseline/version_1/checkpoints/epoch=14-loss=0.0000-val_loss=0.5212.ckpt",
        "../tb_logs/Baseline/version_2/checkpoints/epoch=26-loss=0.0000-val_loss=0.5245.ckpt",
    ]

    pred_test = []
    for _, ckpt in enumerate(checkpoints):
        print(f"Predicted by {ckpt}")

        model = CommonLitModel.load_from_checkpoint(ckpt)
        model.eval()
        model.freeze()

        pred_ckpt = []
        for batch in dataloader:
            z = model(batch)
            pred_ckpt.append(z)

        pred_ckpt = torch.cat(pred_ckpt, dim=0).detach().numpy().copy()
        pred_test.append(pred_ckpt)

    pred_test = np.mean(pred_test, axis=0)
    return pred_test


def main():
    test = pd.read_csv("../data/raw/test.csv")
    submit = test[["id"]].copy()
    submit["target"] = predict()

    print(submit.head())
    submit.to_csv("../data/submit/submission.csv", index=False)


if __name__ == "__main__":
    main()
