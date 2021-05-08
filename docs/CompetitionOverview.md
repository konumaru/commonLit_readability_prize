## コンペについて

### Description

- 学業において適切なレベルの難易度である魅力的な文章に触れることで、生徒は自然にリーディングスキルを身につけることができる
- 既存の手法で Flesch-Kincaid Grade Level が存在するが、文章の構成要素や理論的妥当性の観点が欠けている
    - もう少し精密な Lexile というのも存在するがコストが高く、計算式が公開されておらず、透明性に欠ける
- 今回3年生から12年生のクラスで扱う読み物の複雑さを評価するモデルを構築する
    - 様々な分野から集められた文章

### 評価指標

- RMSE
    - 外れ値の影響を受けやすい
    - 正規分布を仮定

### 期日

- ８月３日

### データについて

- Test Data には Train Data よりも現代のテキストが若干多い
- 公開 Test Data にはライセンス情報が含まれているが、非公開 Test Data には license, url_legal の情報は含まれていない

## EDA of Public Notebooks

- Train data が 2834 records と少なめ
- target は平均 -1 あたりの正規分布っぽい形をしている
    - range(-3.676268, 1.71139)
- std_error は平均 0.5 あたりの少し偏った左に分布をしてる
    - target のサンプル数が少ない分布に合わせて変化している様子

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/85265b31-1aae-4264-8d45-388aa48af8fd/_2021-05-07_23.03.57.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/85265b31-1aae-4264-8d45-388aa48af8fd/_2021-05-07_23.03.57.png)

- Target は比較的既存の Flesch-Kincaid Grade Level で定められているルールに相関を持っていそう

### Libraries

- `nltk.pos_tag(morph)` で品詞を取得できる
    - [https://qiita.com/m__k/items/ffd3b7774f2fde1083fa#品詞の取得](https://qiita.com/m__k/items/ffd3b7774f2fde1083fa#%E5%93%81%E8%A9%9E%E3%81%AE%E5%8F%96%E5%BE%97)

### Models

- （謎の）前処理をしたsentenceを Tfidf → LinearRegression
    - CV=0.59くらい
    - [https://www.kaggle.com/ruchi798/commonlit-readability-prize-eda-baseline](https://www.kaggle.com/ruchi798/commonlit-readability-prize-eda-baseline)
- BERT の fine tune で予測
    - [https://www.kaggle.com/jeongyoonlee/tf-keras-bert-baseline-training-inference](https://www.kaggle.com/jeongyoonlee/tf-keras-bert-baseline-training-inference)
- Roberta with pytorch
    - Public LB 0.511
    - [https://www.kaggle.com/hannes82/commonlit-readability-roberta-inference](https://www.kaggle.com/hannes82/commonlit-readability-roberta-inference)
    - ものすごい過学習してるらしい
        - [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236465#1293407](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236465#1293407)

TF-idfで変換したベクトルを線形モデルで予測する

or

訓練済みBERT系モデルから予測する

がベースラインっぽい

## 疑問

- target はどのようにして決まっている？
- 今回はアンサンブル大会になるかなあ？

## 論文まとめ

一つ一つ読んでIssueにまとめるとよさそう

[CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236307)

[CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236307#1292355)

## 参考ライブラリ

- [https://pypi.org/project/readability/](https://pypi.org/project/readability/)
    - 参照discussion: [https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236321](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236321)
