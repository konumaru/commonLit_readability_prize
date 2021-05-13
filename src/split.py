import pathlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    dump_dir = pathlib.Path("../data/split")
    data = pd.read_csv("../data/raw/train.csv")
    target_bins = pd.cut(data["target"], bins=20, labels=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(data, target_bins)):

        train = data.loc[train_idx, ["excerpt", "target"]]
        valid = data.loc[valid_idx, ["excerpt", "target"]]

        fold_dump_dir = dump_dir / f"fold_{n_fold}"
        fold_dump_dir.mkdir(exist_ok=True)

        train.to_pickle(fold_dump_dir / "train.pkl")
        valid.to_pickle(fold_dump_dir / "valid.pkl")

        print("Fold:", n_fold)
        print(f"\tTrain Target Average: {train.target.mean():.06f}")
        print(f"\tValid Target Average: {valid.target.mean():.06f}")


if __name__ == "__main__":
    main()
