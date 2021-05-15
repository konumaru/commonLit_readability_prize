import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, RobertaTokenizer

from dataset import CommonLitDataModule
from model import CommonLitBertModel, CommonLitModel, CommonLitRoBERTaModel


def main():
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    for n_fold in range(5):
        datamodule = CommonLitDataModule(f"../data/split/fold_{n_fold}/", tokenizer, 32)
        datamodule.setup()

        train_dataloader_len = len(datamodule.train_dataloader())

        tb_logger = TensorBoardLogger(
            save_dir="../tb_logs",
            name="Baseline",
        )

        model = CommonLitModel(
            base_model=CommonLitRoBERTaModel(),
            num_epoch=100,
            train_dataloader_len=train_dataloader_len,
        )
        trainer = Trainer(
            max_epochs=100,
            gpus=1,
            accelerator="dp",
            # fast_dev_run=1,
            logger=tb_logger,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
