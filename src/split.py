import os
import pathlib

import numpy as np
import pandas as pd
from sklearn import model_selection

from utils.common import load_pickle, seed_everything

seed_everything()


def split_fold(data: pd.DataFrame, seed: int = 42):
    dump_dir = pathlib.Path(f"../data/working/seed{seed}/split/")
    os.makedirs(dump_dir, exist_ok=True)
    # Ref: https://www.kaggle.com/abhishek/step-1-create-folds
    num_bins = int(np.floor(1 + np.log2(len(data))))
    target_bins = pd.cut(data["target"], bins=num_bins, labels=False)

    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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


def main():
    SEEDS = [42, 422, 12, 123, 1234]
    data = pd.read_csv("../data/raw/train.csv")

    for seed in SEEDS:
        print(f"\n=== Seed{seed} ===")
        split_fold(data, seed)


if __name__ == "__main__":
    main()
