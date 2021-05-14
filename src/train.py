import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from dataset import CommonLitDataModule
from model import CommonLitBertModel, CommonLitModel


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    for n_fold in range(5):
        datamodule = CommonLitDataModule(f"../data/split/fold_{n_fold}/", tokenizer, 32)
        datamodule.setup()

        tb_logger = TensorBoardLogger(
            save_dir="../tb_logs",
            name="Debug",
        )

        model = CommonLitModel(base_model=CommonLitBertModel())
        trainer = Trainer(
            max_epochs=10,
            fast_dev_run=1,
            logger=tb_logger,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

        break


if __name__ == "__main__":
    main()
