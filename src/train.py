import argparse
import os
import pathlib
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer

from dataset import CommonLitDataModule
from models import CommonLitModel
from utils.common import load_pickle, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="is use debug mode")

    args = parser.parse_args()
    return args


def train():
    seed = 42
    debug = True
    num_fold = 5

    lr = 2e-5
    num_epochs = 1
    batch_size = 16
    model_name_or_path = "roberta-base"

    work_dir = pathlib.Path(f"../data/working/seed{seed}")
    os.makedirs(work_dir / "models", exist_ok=True)

    data = pd.read_csv("../data/raw/train.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    best_checkpoints = []
    oof = np.zeros(data.shape[0])
    for fold in range(num_fold):
        datamodule = CommonLitDataModule(
            work_dir / f"split/fold_{fold}", tokenizer, batch_size
        )
        model = CommonLitModel(
            lr=lr,
            num_epochs=num_epochs,
            lr_scheduler="cosine",
            lr_interval="step",
            lr_warmup_step=int(len(datamodule.train_dataloader()) * 0.06),
            roberta_model_name_or_path=model_name_or_path,
            train_dataloader_len=len(datamodule.train_dataloader()),
        )
        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{val_metric:.4f}",
        )

        if debug:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            trainer = pl.Trainer(
                accelerator=None,
                max_epochs=1,
                limit_train_batches=0.01,
                limit_val_batches=0.05,
                callbacks=[lr_monitor, checkpoint],
            )
            trainer.fit(model=model, datamodule=datamodule)
        else:
            trainer = pl.Trainer(
                accelerator="dp",
                gpus=1,
                callbacks=[lr_monitor, checkpoint],
                max_epochs=num_epochs,
                stochastic_weight_avg=True,
                val_check_interval=0.1,
                limit_train_batches=0.9,
                limit_val_batches=0.9,
                fast_dev_run=debug,
            )
            trainer.fit(model=model, datamodule=datamodule)

        print(f"Fold-{fold} Best Checkpoint:\n", checkpoint.best_model_path)
        best_checkpoints.append(checkpoint.best_model_path)

        # Save best weight mdoel as pytroch format.
        best_model = CommonLitModel.load_from_checkpoint(checkpoint.best_model_path)
        torch.save(
            best_model.roberta_model.state_dict(),
            work_dir / f"models/fold_{fold}.pth",
        )

        # Predict oof
        if debug:
            pred = np.random.rand(len(datamodule.valid.index))
        else:
            pred = trainer.predict(dataloaders=datamodule.val_dataloader())
            pred = np.concatenate(pred, axis=0).ravel()

        oof[datamodule.valid.index] = pred

        break

    np.save(work_dir / "oof.npy", oof)

    metric = mean_squared_error(data["target"].values, oof, squared=False)
    with open(work_dir / f"metric={metric:.6f}.txt", "w") as f:
        f.write("")


def main():
    args = parse_args()

    train()


if __name__ == "__main__":
    main()
