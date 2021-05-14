import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class CommonLitDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=100):
        self.target = data[["target"]].to_numpy()
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
        token_type_ids = inputs["token_type_ids"]

        target = self.target[idx]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
        }


class CommonLitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, tokenizer, batch_size: int = 32):
        super(CommonLitDataModule, self).__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train = pd.read_pickle(self.data_dir / "train.pkl")
        self.valid = pd.read_pickle(self.data_dir / "valid.pkl")

    def train_dataloader(self):
        dataset = CommonLitDataset(self.train, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = CommonLitDataset(self.valid, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        dataset = CommonLitDataset(self.valid, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )
