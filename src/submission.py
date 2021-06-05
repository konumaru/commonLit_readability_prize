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
from models import CommonLitRoBERTaModel

pandarallel.initialize()


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


def predict_by_ckpt(
    data: pd.DataFrame,
    num_fold: int = 15,
    model_name: str = "roberta-base",
) -> List[np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(data)

    pred = []
    for i, ckpt in enumerate(range(num_fold)):
        print(f"Predicted by {i}-fold model.")

        model = CommonLitRoBERTaModel().to(device)
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
    num_fold = 15
    pred = predict_by_ckpt(test)
    test[[f"pred_{i}" for i in range(num_fold)]] = pred

    X_pred = test[[f"pred_{i}" for i in range(num_fold)]]

    model_dir = "../data/models/svr/"
    # model_dir = "../data/models/xgb/"

    num_fold = 15
    submission = test[["id"]].copy()
    submission["target"] = predict(X_pred, model_dir, num_fold)

    # submission.to_csv("submission.csv", index=False)

    print(submission.head())


if __name__ == "__main__":
    main()
