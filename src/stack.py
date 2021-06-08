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
from models import CommonLitRoBERTaModel, RMSELoss
from utils.common import load_pickle
from utils.train_function import train_cross_validate, train_ridge, train_svr, train_xbg


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


def predict_by_ckpt(
    num_fold: int = 15,
    model_name: str = "roberta-base",
) -> List[np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()

    pred = []
    for i, ckpt in enumerate(range(num_fold)):
        print(f"Predicted by {i}-fold model.")

        model = CommonLitRoBERTaModel(model_name_or_path=model_name).to(device)
        model.load_state_dict(torch.load(f"../data/models/{model_name}/{i}-fold.pth"))
        model.eval()  # Ignore dropout and bn layers.

        pred_ckpt = []
        with torch.no_grad():  # Skip gradient calculation
            for batch in dataloader:
                batch["inputs"]["input_ids"] = batch["inputs"]["input_ids"].to(device)
                batch["inputs"]["attention_mask"] = batch["inputs"][
                    "attention_mask"
                ].to(device)
                batch["inputs"]["token_type_ids"] = batch["inputs"][
                    "token_type_ids"
                ].to(device)
                batch["textstat"] = batch["textstat"].to(device)

                z = model(batch)
                pred_ckpt.append(z)

        pred_ckpt = torch.cat(pred_ckpt, dim=0).detach().cpu().numpy().copy()
        pred.append(pred_ckpt)

    return pred


def main():
    num_fold = 15
    # model_name = "further-trained-roberta"  # roberta-base, further-trained-roberta
    model_name = "distilroberta-base"

    pred = predict_by_ckpt(num_fold, model_name)

    train = pd.read_csv("../data/raw/train.csv")[["id", "target"]]
    train[[f"pred_{i}" for i in range(num_fold)]] = pred

    textstat = load_pickle("../data/features/textstats.pkl", verbose=False).to_numpy()
    X = train[[f"pred_{i}" for i in range(num_fold)]].copy().to_numpy()
    # X = np.concatenate([X, textstat], axis=1)  # Not improved
    y = train["target"].to_numpy()

    cv = model_selection.RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=42
    )
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_cv = pd.cut(y, bins=num_bins, labels=False)

    train_cross_validate(
        X, y, cv, train_svr, save_dir=f"../data/models/{model_name}/svr/", y_cv=y_cv
    )
    train_cross_validate(
        X, y, cv, train_xbg, save_dir=f"../data/models/{model_name}/xgb/", y_cv=y_cv
    )
    train_cross_validate(
        X, y, cv, train_ridge, save_dir=f"../data/models/{model_name}/ridge/", y_cv=y_cv
    )


if __name__ == "__main__":
    main()
