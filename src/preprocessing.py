import pathlib
import re
from typing import AnyStr

import nltk
import numpy as np
import pandas as pd
import textstat
from nltk import pos_tag
from nltk.corpus import stopwords
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils.common import save_cache, seed_everything, timer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

pandarallel.initialize(progress_bar=True)

DST_DIR = "../data/features/"

seed_everything()


@save_cache(filepath=f"{DST_DIR}preprocessed_excerpt.pkl", use_cache=True)
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


@save_cache(filepath=f"{DST_DIR}textstats.pkl", use_cache=False)
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


def main():
    train = pd.read_csv("../data/raw/train.csv", usecols=["id", "excerpt", "target"])
    train["preprocessed_excerpt"] = get_preprocessed_excerpt(train)

    _ = get_textstat(train)


if __name__ == "__main__":
    with timer("Preprocessing"):
        main()
