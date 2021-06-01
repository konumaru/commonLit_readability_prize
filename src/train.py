import os
import pathlib
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dataset import CommonLitDataModule
from models import CommonLitModel


def calc_average_loss(ckeckpoints):
    metrics = []
    for ckpt in ckeckpoints:
        metric = float(re.findall(r"val_metric=(\d+\.\d+)", ckpt)[0])
        metrics.append(metric)

    return metrics


def dump_best_checkpoints(best_checkpoints, exp_name, metric_avg, metric_std):
    os.makedirs(f"../data/versions/{exp_name}", exist_ok=True)

    with open(
        f"../data/versions/{exp_name}/best_checkpoints_{metric_avg:.6f}±{metric_std:.4f}.txt",
        "w",
    ) as f:
        txt = "\n".join(best_checkpoints)
        f.write(txt)


def main():
    DEBUG = 0
    NUM_FOLD = 15 if DEBUG == 0 else 1

    exp_name = "RoBERTa-Baseline"
    num_epoch = 15
    batch_size = 8
    lr = 5e-5

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    best_checkpoints = []
    for n_fold in range(NUM_FOLD):
        datamodule = CommonLitDataModule(
            f"../data/split/fold_{n_fold}/", tokenizer, batch_size
        )

        # Logger
        tb_logger = TensorBoardLogger(
            save_dir="../tb_logs",
            name=exp_name,
        )
        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop = EarlyStopping(
            mode="min",
            patience=5,
            verbose=False,
            monitor="val_loss",
            min_delta=0.005,
        )
        checkpoint = ModelCheckpoint(
            filename="{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{val_metric:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )

        model = CommonLitModel(
            lr=lr,
            num_epoch=num_epoch,
            lr_scheduler="cosine",
            lr_interval="epoch",
            lr_warmup_step=0,
        )
        trainer = Trainer(
            accelerator="dp",
            gpus=1,
            logger=tb_logger,
            callbacks=[lr_monitor, checkpoint],
            max_epochs=num_epoch,
            stochastic_weight_avg=True,
            val_check_interval=0.2,
            limit_train_batches=0.9,
            limit_val_batches=1.0,
            fast_dev_run=DEBUG,
        )
        trainer.fit(model=model, datamodule=datamodule)

        print(f"{n_fold}-Fold Best Checkpoint:\n", checkpoint.best_model_path)
        best_checkpoints.append(checkpoint.best_model_path)

    metrics = calc_average_loss(best_checkpoints)
    metric_avg = np.mean(metrics)
    metric_std = np.std(metrics)

    dump_best_checkpoints(best_checkpoints, exp_name, metric_avg, metric_std)
    print(best_checkpoints)

    print(f"Average Validation Loss: {metric_avg:.6f} ± {metric_std:.4f}")


if __name__ == "__main__":
    main()
