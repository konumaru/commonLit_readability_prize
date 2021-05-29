import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from utils.common import load_pickle


def main():
    dump_dir = pathlib.Path("../data/split")
    data = pd.read_csv("../data/raw/train.csv")

    textstat = load_pickle("../data/features/textstats.pkl")

    data = pd.concat([data, textstat], axis=1)
    data.drop(["id", "url_legal", "license", "standard_error"], axis=1, inplace=True)

    print(data.head())

    # Ref: https://www.kaggle.com/abhishek/step-1-create-folds
    num_bins = int(np.floor(1 + np.log2(len(data))))
    target_bins = pd.cut(data["target"], bins=num_bins, labels=False)

    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # NOTE: Local cv is local cv is 0.517480 ± 0.0216
    # cv = RepeatedKFold(n_splits=5,  n_repeats=3, random_state=42)
    # NOTE: Local cv is 0.514975 ± 0.0203
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(data, target_bins)):

        train = data.loc[train_idx, :]
        valid = data.loc[valid_idx, :]

        fold_dump_dir = dump_dir / f"fold_{n_fold}"
        fold_dump_dir.mkdir(exist_ok=True)

        train.to_pickle(fold_dump_dir / "train.pkl")
        valid.to_pickle(fold_dump_dir / "valid.pkl")

        print("Fold:", n_fold)
        print(
            f"\tTrain Target Average: {train.target.mean():.06f}"
            + f"\tTrain Size={train.shape[0]}"
        )
        print(
            f"\tValid Target Average: {valid.target.mean():.06f}"
            + f"\tValid Size={valid.shape[0]}"
        )


if __name__ == "__main__":
    main()
