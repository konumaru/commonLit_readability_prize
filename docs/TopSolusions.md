# Top Solusions

## 全体の傾向

- roberta-large を中心とした bert 亜種のモデルをアンサンブル

## ToC

- [Top Solusions](#top-solusions)
  - [全体の傾向](#全体の傾向)
  - [ToC](#toc)
  - [Solusions](#solusions)
    - [1 位](#1-位)
    - [2 位](#2-位)
    - [3 位](#3-位)
    - [4 位](#4-位)
  - [What to use next](#what-to-use-next)
  - [Reference](#reference)

## Solusions

### [1 位](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257844)

- [外部データ](https://github.com/UKPLab/sentence-transformers)の利用
- 複数モデルをリッジ回帰でアンサンブル
  - albert-xxlarge, deberta-large, roberta-large, electra-large
- 6 fold cross validation
- evaluation with oof

### [2 位](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258328)

- 5 fold cross validation
- 19 models ensemble
- post processing
  - 予測値の範囲によって倍率を調整

### [3 位](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258095)

- モデルのアンサンブル
  - roberta-large, deberta-large
- 学習時のパラメータなど
  - 減衰学習率と層ごとの割当
    - Regression Head には別途学習率を割当
  - Multi Smaple Dropout
  - Weighted average of hidden layers

### [4 位](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258148)

- 大量のモデルのアンサンブル
- Loss は MSE
- No dropout

## What to use next

- roberta 以外のモデルでの学習をうまくいくまで試す

## Reference

- https://upura.hatenablog.com/entry/2021/08/08/182756
