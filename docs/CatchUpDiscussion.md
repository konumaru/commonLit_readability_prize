# Catch Up Discussion

## 2021-06-05

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
