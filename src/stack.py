import os
import pathlib
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import model_selection
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CommonLitDataset
from models import CommonLitModel, CommonLitRoBERTaModel, RMSELoss
from utils.common import load_pickle
from utils.train_function import train_cross_validate, train_svr, train_xbg


def load_data() -> pd.DataFrame:
    dump_dir = pathlib.Path("../data/split")
    data = pd.read_csv("../data/raw/train.csv")

    textstat = load_pickle("../data/features/textstats.pkl", verbose=False)

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


def predict_by_ckpt(checkpoints):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()

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
    ckpt_path = "../data/models/roberta/best_checkpoints_0.496413±0.0162.txt"
    checkpoints = get_ckpt_path(ckpt_path)

    pred = predict_by_ckpt(checkpoints)

    train = pd.read_csv("../data/raw/train.csv")[["id", "target"]]
    train[[f"pred_{i}" for i in range(len(checkpoints))]] = pred

    X = train[[f"pred_{i}" for i in range(len(checkpoints))]].copy().to_numpy()
    y = train["target"].to_numpy()

    cv = model_selection.StratifiedKFold(n_splits=5)
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_cv = pd.cut(y, bins=num_bins, labels=False)
    train_cross_validate(X, y, cv, train_svr, save_dir="../data/models/svr/", y_cv=y_cv)


if __name__ == "__main__":
    main()
