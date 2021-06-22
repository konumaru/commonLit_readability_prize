import argparse
import datetime
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from kaggle.api.kaggle_api_extended import KaggleApi
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error
from transformers import AutoModel, AutoTokenizer

from dataset import CommonLitDataModule
from models import CommonLitModel, RMSELoss
from utils.common import load_pickle, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="is use debug mode")
    parser.add_argument(
        "--dump_dir", type=str, default="../data/models", help="learning rate"
    )
    # Below is model paramerters.
    parser.add_argument("--seed", type=int, default=42, help="fix random seed")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="learning rate")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-base",
        help="pretrained model name or path",
    )

    args = parser.parse_args()
    return args


def remove_glob(filepath, recursive=True):
    for p in glob.glob(filepath, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)


def train(
    seed: int = 42,
    dump_dir: str = "../data/models",
    debug: bool = True,
    num_fold: int = 5,
    lr: float = 2e-5,
    num_epochs: int = 15,
    batch_size: int = 16,
    model_name_or_path: str = "roberta-base",
):
    work_dir = pathlib.Path(os.path.join(dump_dir, f"{model_name_or_path}/seed{seed}"))
    os.makedirs(work_dir, exist_ok=True)

    remove_glob(str(work_dir / "models"))
    os.makedirs(work_dir / "models", exist_ok=True)

    data = pd.read_csv("../data/raw/train.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    best_checkpoints = []
    oof = np.zeros(data.shape[0])
    for fold in range(num_fold):
        datamodule = CommonLitDataModule(
            f"../data/working/seed{seed}/split/fold_{fold}", tokenizer, batch_size
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
                val_check_interval=0.05,
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
            pred = torch.cat(pred, dim=0).detach().cpu().numpy().ravel()
            # pred = np.concatenate(pred, axis=0).ravel()

        oof[datamodule.valid.index] = pred

    np.save(work_dir / "oof.npy", oof)

    metric = mean_squared_error(data["target"].values, oof, squared=False)
    with open(work_dir / f"metric_{metric:.6f}.txt", "w") as f:
        f.write("")


def upload_to_kaggle_dataset(
    user_id: str,
    dataset_title: str,
    upload_dir: str,
    message: str,
):
    dataset_metadata = {}
    dataset_metadata["id"] = f"{user_id}/{dataset_title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = dataset_title

    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    api = KaggleApi()
    api.authenticate()

    if dataset_metadata["id"] not in [
        str(d) for d in api.dataset_list(user=user_id, search=dataset_title)
    ]:
        # If dataset is not exist, run below.
        api.dataset_create_new(
            folder=upload_dir,
            convert_to_csv=False,
            dir_mode="skip",
        )
    else:
        api.dataset_create_version(
            folder=upload_dir,
            version_notes=message,
            convert_to_csv=False,
            delete_old_versions=True,
            dir_mode="zip",
        )


def main():
    args = parse_args()

    # SEEDS = [42, 422, 12, 123, 1234]
    # for seed in SEEDS:
    train(
        seed=args.seed,
        dump_dir=args.dump_dir,
        debug=args.debug,
        num_fold=5,
        # Below is model paramerters.
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        model_name_or_path=args.model_name_or_path,
    )

    upload_to_kaggle_dataset(
        user_id="konumaru",
        dataset_title="commonlit-fine-tuned-roberta-base",
        upload_dir=f"../data/models/{args.model_name_or_path}",
        message=datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
    )


if __name__ == "__main__":
    main()
