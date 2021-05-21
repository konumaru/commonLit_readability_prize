# Competition Overview

## 概要

### Description

- 学業において適切なレベルの難易度である魅力的な文章に触れることで、生徒は自然にリーディングスキルを身につけることができる
- 既存の手法で Flesch-Kincaid Grade Level が存在するが、文章の構成要素や理論的妥当性の観点が欠けている
  - もう少し精密な Lexile というのも存在するがコストが高く、計算式が公開されておらず、透明性に欠ける
- 今回 3 年生から 12 年生のクラスで扱う読み物の複雑さを評価するモデルを構築する
  - 様々な分野から集められた文章

### 評価指標

- RMSE
  - 外れ値の影響を受けやすい
  - 正規分布を仮定

### 期日

- ８月３日

### データについて

- 学習データが 2,800 行と少ないのが特徴的
- Test Data には Train Data よりも現代のテキストが若干多い
- 公開 Test Data にはライセンス情報が含まれているが、非公開 Test Data には license, url_legal の情報は含まれていない
- テストデータは 2,000 弱
  - https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236335#1292356

## EDA of Public Notebooks

- Train data が 2834 records と少なめ
- target は平均 -1 あたりの正規分布っぽい形をしている
  - range(-3.676268, 1.71139)
- std_error は平均 0.5 あたりの少し偏った左に分布をしてる

  - target のサンプル数が少ない分布に合わせて変化している様子

    ![img](https://user-images.githubusercontent.com/17187586/117521366-62375900-afe8-11eb-8838-d6ad5c5ea84d.png)

- Target は比較的既存の Flesch-Kincaid Grade Level で定められているルールに相関を持っていそう

### Libraries

- `nltk.pos_tag(morph)` で品詞を取得できる
  - [https://qiita.com/m\_\_k/items/ffd3b7774f2fde1083fa#品詞の取得](https://qiita.com/m__k/items/ffd3b7774f2fde1083fa#%E5%93%81%E8%A9%9E%E3%81%AE%E5%8F%96%E5%BE%97)

### Models

- （謎の）前処理をした sentence を Tfidf → LinearRegression
  - CV=0.59 くらい
  - [https://www.kaggle.com/ruchi798/commonlit-readability-prize-eda-baseline](https://www.kaggle.com/ruchi798/commonlit-readability-prize-eda-baseline)
- BERT の fine tune で予測
  - [https://www.kaggle.com/jeongyoonlee/tf-keras-bert-baseline-training-inference](https://www.kaggle.com/jeongyoonlee/tf-keras-bert-baseline-training-inference)
- Roberta with pytorch
  - Public LB 0.511
  - [https://www.kaggle.com/hannes82/commonlit-readability-roberta-inference](https://www.kaggle.com/hannes82/commonlit-readability-roberta-inference)
  - ものすごい過学習してるらしい
    - [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236465#1293407](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236465#1293407)

TF-idf で変換したベクトルを線形モデルで予測する

or

訓練済み BERT 系モデルから予測する

がベースラインっぽい

### コンペの難しいところ

- 学習データが少ない
- fine-tuning の不安定性
- Submission が３時間以内

## 疑問

- target はどのようにして決まっている？
- 今回はアンサンブル大会になるのか？

## 論文まとめ

一つ一つ読んで Issue にまとめるとよさそう

[CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236307)

[CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236307#1292355)

## 参考ライブラリ

- [https://pypi.org/project/readability/](https://pypi.org/project/readability/)
  - 参照 discussion: [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236321](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236321)
