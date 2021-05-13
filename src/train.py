import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from dataset import CommonLitDataModule
from model import CommonLitModel


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    datamodule = CommonLitDataModule("../data/split/fold_0", tokenizer, 32)
    datamodule.setup()

    model = CommonLitModel()
    trainer = Trainer(
        max_epochs=3,
        fast_dev_run=1,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
