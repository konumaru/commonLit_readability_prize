"""
NOTE:

- ここでは src 内の module を import した実行を許す
- 最終の実装は notebook 形式で行う
- notebook, dataset を kaggle api を使って upload する
- そのあと、kaggle code 上で実行し、submission

kaggle api は makefile に実装できると良さそう
"""
import re
from typing import AnyStr

import nltk
import numpy as np
import pandas as pd
import textstat
from nltk import pos_tag
from nltk.corpus import stopwords
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler

from dataset import CommonLitDataset

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


def main():
    test = pd.read_csv("../data/raw/test.csv", usecols=["id", "excerpt"])
    test["preprocessed_excerpt"] = get_preprocessed_excerpt(test)
    textstat_feat = get_textstat(test)

    test = pd.concat([test, textstat_feat], axis=1)
    print(test.head())


if __name__ == "__main__":
    main()
