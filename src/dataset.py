import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class CommonLitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int = 256,
        is_test: int = False,
    ):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.excerpt = data[["excerpt"]].to_numpy()

        if is_test:
            self.target = np.zeros((len(data), 1))
            self.textstat = np.zeros((len(data), 1))
        else:
            self.target = data[["target"]].to_numpy()
            textstat = data.drop(["excerpt", "target"], axis=1)
            self.textstat = textstat.to_numpy()

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, idx):
        text = str(self.excerpt[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        textstat = self.textstat[idx]
        target = self.target[idx]

        return {
            "inputs": {
                "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    inputs["attention_mask"], dtype=torch.long
                ),
                "token_type_ids": torch.tensor(
                    inputs["token_type_ids"], dtype=torch.long
                ),
            },
            "textstat": torch.tensor(textstat, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


class CommonLitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, tokenizer, batch_size: int = 32):
        super(CommonLitDataModule, self).__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train = pd.read_pickle(self.data_dir / "train.pkl")
        self.valid = pd.read_pickle(self.data_dir / "valid.pkl")

    def train_dataloader(self):
        dataset = CommonLitDataset(self.train, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = CommonLitDataset(self.valid, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )


if __name__ == "__main__":
    data = pd.read_pickle("../data/split/fold_0/train.pkl")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = CommonLitDataset(data, tokenizer)

    datamodule = CommonLitDataModule(
        data_dir="../data/split/fold_0/",
        tokenizer=tokenizer,
        batch_size=4,
    )

    batch = iter(datamodule.train_dataloader()).next()
    print(batch)
