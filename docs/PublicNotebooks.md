# Chatch Up Public Notebooks

## 2021-06-23

### Ensemble Public Notebooks

https://www.kaggle.com/mobassir/commonlit-readability-tensorflow-torch-ensemble

## 2021-05-20

### RoBerta + SVM, LB = 0.478

https://www.kaggle.com/maunish/clrp-roberta-svm?scriptVersionId=63301342

- RoBerta + SVM の stacking
  - RoBerta + LGBMh では LB = 0.483 ~ 0.512
- Cross Validation
  - Fold の nbin を np.floor(1 + np.log2(len(train_data))) にしている。意図はわからない
    - np.floor(1 + np.log2(2000)) = 11 になる
- １層目: RobertaForSequenceClassification
  - batch_size=128, max_len=256
  - num_labels=1 にして回帰予測
- ２層目: SVM
  - １層目の予測値を学習
  - CV は StratifiedKFold, bin を使う
- raw_data の train を再度２層目で使っているのが印象的
  - よく考えたら GBDT もそういう学習してるか。。。
- tokenizer では truncate=True にしたほうがよさそう

### RoBerta + XGB, LB = 0.488

https://www.kaggle.com/jollibobert/commonlit-readability-roberta-xgb-baseline

- RoBerta + XGB の stacking
  - １層目の model は fold １つ分だけで検証している
  - CV は KFold
- １層目: RobertaForSequenceClassification
- ２層目: XGBoost
  - lr=0.05, max_depth=3, early_stopping_rounds=20

### RoBerta + CNN, LB = 0.489

https://www.kaggle.com/sourabhy/commonlit-roberta-cnn?scriptVersionId=63343981

- １層目: RobertaForSequenceClassification
- ２層目: 1D CNN

```python
# こうやって書くと token_type_ids があるとき、ないときのどちらも対応できるんだな？
inputs = {key:val.reshape(val.shape[0], -1).to(device) for key, val in inputs.items()}
outputs = model(**inputs)
```

### BERT + textstats, LB = 0.505

https://www.kaggle.com/simakov/lama-bert-inference?scriptVersionId=62711466

- textstats
  - word_count, sentence_len, sentence_count, comm_count など
- LAMA モデルっていうのが何かわからない
- textstats を追加することで 0.53X -> 0.505 に上昇
