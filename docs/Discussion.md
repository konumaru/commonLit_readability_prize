# Catch Up Discussion

## 2021-07-06

この時点で特に読むものがなくなってきている感覚がある。
コンペ締め切り１ヶ月前。

## 2021-06-09

- [Target の Noise に対する対処案](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/244431)
  - target は専門家が直接アノテーションしたものではない
    - Bradley-Terry 分布を使用した加工がされている
  - 意見
    - 少ないデータに対して連続値で学習すると過学習する可能性が考えられるため、分類問題にするのがやはりいいのかもしれない
- [Transformer の FNN 層を 1dCNN にしてみた](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/244421)
  - [FastSpeech](https://arxiv.org/pdf/1905.09263.pdf) という既存の取り組みみたい
  - 上記の元ネタは音声データへのアプローチ
  - アイデアを取り込むなら、RoBERTa の pooler_output を 1dCNN が学習するとかから検証してみようかな。
- [コンピュータビジョンのアプローチ](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/244306)
  - CV のアプローチを使えるか？という問題提議
  - 文章を木構造に変換して、その画像をそのまま resnet で学習するというもの
  - ベースラインを全体の平均値としたとき、改善した。
  - こういうアプローチもあるんだな、というくらいのもの。

## 2021-05-22

- 学習済みモデルがたくさん参照されている notebook
  - https://www.kaggle.com/leighplt/transformers-cv-train-inference-pytorch

## 2021-05-20

- Google 翻訳で２重翻訳することで Data Augmentation をする
  - https://www.kaggle.com/c/commonlitreadabilityprize/discussion/237182
  - お気持ち
    - そのまま予測するのは少し抵抗があるけど、stack するなら中間のモデルがいい感じに吸収してくれるかも（？）
- CV vs LB
  - https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236645
  - [roberta-large のスコアが高い](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236645)
    - batch_size=8 で学習してるらしい
  - Seed Averaging するだけでもスコア改善しそう
    - https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236645
  - spaCy を使った特徴量エンジニアリング
    - https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236645
- [Some Ideas](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/239434)
  - Binning して分類問題として CrossEntropy で解く
    - → 　過学習が避けられる？スタッキングを前提にするならいいのかもしれない？
  - いくつかの Checkpoint でアンサンブルする
