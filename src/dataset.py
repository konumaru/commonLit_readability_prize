import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class CommonLitDataset(Dataset):
    def __init__(self, target, excerpt, tokenizer, max_len):
        self.target = target
        self.excerpt = excerpt
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])
        inputs = self.tokenizer(
            text, max_length=self.max_len, padding="max_length", truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        target = self.target[item]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.double),
        }


class CommonLitDataModule(pl.LightningDataModule):
    def __init__(self):
        super(CommonLitDataModule, self).__init__()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
