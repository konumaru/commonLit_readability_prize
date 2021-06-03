"""
NOTE:

- ここでは src 内の module を import した実行を許す
- 最終の実装は notebook 形式で行う
- notebook, dataset を kaggle api を使って upload する
- そのあと、kaggle code 上で実行し、submission

kaggle api は makefile に実装できると良さそう
"""
import os
import pickle
import re
from typing import AnyStr, List, Optional

import nltk
import numpy as np
import pandas as pd
import textstat
import torch
from nltk import pos_tag
from nltk.corpus import stopwords
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CommonLitDataset
from models import CommonLitModel

pandarallel.initialize(progress_bar=True)


def get_preprocessed_excerpt(src_data: pd.DataFrame) -> pd.DataFrame:
    def preprocess_excerpt(text: AnyStr):
        text = re.sub("[^a-zA-Z]", " ", text).lower()
        text = nltk.word_tokenize(text)  # NOTE: 英文を単語分割する
        text = [word for word in text if word not in set(stopwords.words("english"))]

        lemma = nltk.WordNetLemmatizer()  # NOTE: 複数形の単語を単数形に変換する
        text = " ".join([lemma.lemmatize(word) for word in text])
        return text

    dst_data = src_data["excerpt"].parallel_apply(preprocess_excerpt)
    return dst_data


def get_textstat(src_data: pd.DataFrame) -> pd.DataFrame:
    dst_data = pd.DataFrame()

    dst_data = dst_data.assign(
        excerpt_len=src_data["preprocessed_excerpt"].str.len(),
        avg_word_len=(
            src_data["preprocessed_excerpt"]
            .apply(lambda x: [len(s) for s in x.split()])
            .map(np.mean)
        ),
        char_count=src_data["excerpt"].map(textstat.char_count),
        word_count=src_data["preprocessed_excerpt"].map(textstat.lexicon_count),
        sentence_count=src_data["excerpt"].map(textstat.sentence_count),
        syllable_count=src_data["excerpt"].apply(textstat.syllable_count),
        smog_index=src_data["excerpt"].apply(textstat.smog_index),
        automated_readability_index=src_data["excerpt"].apply(
            textstat.automated_readability_index
        ),
        coleman_liau_index=src_data["excerpt"].apply(textstat.coleman_liau_index),
        linsear_write_formula=src_data["excerpt"].apply(textstat.linsear_write_formula),
    )

    scaler = StandardScaler()
    feat_cols = dst_data.columns.tolist()
    dst_data[feat_cols] = scaler.fit_transform(dst_data)
    return dst_data


def get_dataloader(data: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = CommonLitDataset(data, tokenizer, 256, is_test=True)
    return DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )


def predict_by_ckpt(data, checkpoints):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(data)

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


def predict(data: pd.DataFrame, model_dir: str, n_splits: int) -> np.ndarray:
    pred = np.zeros(data.shape[0])
    for n_fold in range(n_splits):
        with open(os.path.join(model_dir, f"{n_fold}-fold.pkl"), mode="rb") as file:
            model = pickle.load(file)

        pred += model.predict(data) / n_splits

    return pred


def main():
    test = pd.read_csv("../data/raw/test.csv", usecols=["id", "excerpt"])
    test["preprocessed_excerpt"] = get_preprocessed_excerpt(test)
    textstat_feat = get_textstat(test)

    test = pd.concat([test, textstat_feat], axis=1)

    # Predict by RoBERTa
    ckpt_path = "../data/models/roberta/best_checkpoints_0.496413±0.0162.txt"
    checkpoints = get_ckpt_path(ckpt_path)

    pred = predict_by_ckpt(test, checkpoints)
    test[[f"pred_{i}" for i in range(len(checkpoints))]] = pred

    X_pred = test[[f"pred_{i}" for i in range(len(checkpoints))]]

    model_dir = "../data/models/svr/"
    # model_dir = "../data/models/xgb/"

    submission = test[["id"]].copy()
    submission["target"] = predict(X_pred, model_dir, 5)

    # submission.to_csv("submission.csv", index=False)

    print(submission.head())


if __name__ == "__main__":
    main()
