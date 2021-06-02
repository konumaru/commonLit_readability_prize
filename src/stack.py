import os
import pathlib
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import xgboost
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from xgboost import XGBRegressor

from dataset import CommonLitDataset
from models import CommonLitModel, CommonLitRoBERTaModel, RMSELoss
from utils.common import load_pickle
from utils.train_function import train_cross_validate, train_svr, train_xbg


def load_data() -> pd.DataFrame:
    dump_dir = pathlib.Path("../data/split")
    data = pd.read_csv("../data/raw/train.csv")

    textstat = load_pickle("../data/features/textstats.pkl")

    data = pd.concat([data, textstat], axis=1)
    data.drop(["id", "url_legal", "license", "standard_error"], axis=1, inplace=True)
    return data


def get_dataloader():
    train = load_data()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = CommonLitDataset(train, tokenizer, 256)
    return DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )


def predict_by_ckpt(checkpoints, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred = []
    for _, ckpt in enumerate(checkpoints):
        print(f"Predicted by {ckpt}")

        model = CommonLitModel().load_from_checkpoint(ckpt)
        model = model.to(device)
        model.eval()
        model.freeze()

        pred_ckpt = []
        for batch in dataloader:
            batch["inputs"]["input_ids"] = batch["inputs"]["input_ids"].to(device)
            batch["inputs"]["attention_mask"] = batch["inputs"]["attention_mask"].to(
                device
            )
            batch["inputs"]["token_type_ids"] = batch["inputs"]["token_type_ids"].to(
                device
            )
            batch["textstat"] = batch["textstat"].to(device)

            z = model(batch)
            pred_ckpt.append(z)

        pred_ckpt = torch.cat(pred_ckpt, dim=0).detach().cpu().numpy().copy()
        pred.append(pred_ckpt)

    return pred


def get_ckpt_path(checkpoint_path: str) -> List:
    with open(checkpoint_path, "r") as f:
        txt = f.readlines()

    model_version = "RoBERTa-Baseline"
    dir_path = "../data/models/roberta/"
    checkpoints = [t.strip() for t in txt]
    checkpoints = [ckpt.replace("../tb_logs/", dir_path) for ckpt in checkpoints]
    return checkpoints


def main():
    # Predict by RoBERTa
    ckpt_path = "../data/models/roberta/best_checkpoints.txt"
    checkpoints = get_ckpt_path(ckpt_path)
    dataloader = get_dataloader()

    pred = predict_by_ckpt(checkpoints, dataloader)

    train = pd.read_csv("../data/raw/train.csv")[["id", "target"]]
    train[[f"pred_{i}" for i in range(len(checkpoints))]] = pred

    X = train[[f"pred_{i}" for i in range(len(checkpoints))]].copy()
    y = train["target"]

    cv = model_selection.StratifiedKFold(n_splits=5)
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_cv = pd.cut(y, bins=num_bins, labels=False)
    train_cross_validate(X, y, cv, train_svr, save_dir="../data/models/svr/", y_cv=y_cv)


if __name__ == "__main__":
    main()
