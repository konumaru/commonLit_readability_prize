# CommonLit Readability Prize

![background](https://user-images.githubusercontent.com/17187586/117848267-1c1b1780-b2be-11eb-8035-6fbd4e081d09.png)

## Summary

The RMSE of the final result is 0.4607133200425228 with 0.7 \* ensemble + 0.3 \* stacking

![img](https://user-images.githubusercontent.com/17187586/127723772-b0c4e68a-65e4-4082-b53e-88cad7908205.png)

## Evaluation

- The evaluation is RMSE

### Cross Validation Strategy

- StratifiedKFold

```python
num_bins = int(np.floor(1 + np.log2(len(data))))
target_bins = pd.cut(data["target"], bins=num_bins, labels=False)
```

## Models

### Ensemble single models

|                                            |   RMSE   |
| :----------------------------------------: | :------: |
| roberta-base + attention head + layer norm | 0.473694 |
|       roberta-base + attention head        | 0.470910 |
|    roberta-base-squad2 + attention head    | 0.477740 |
|       roberta-large + attention head       | 0.473006 |
|   roberta-large-squad2 + attention head    | 0.471116 |
|       roberta-large + mean pool head       | 0.474779 |

The RMSE that averages all of the above is 0.46214926662874833

### Stacking

|               |   RMSE   |
| :-----------: | :------: |
|     Ridge     | 0.462588 |
| Baysian Ridge | 0.462392 |
|      MLP      | 0.508576 |
|      SVR      | 0.468852 |
|      XGB      | 0.463275 |
| Random Forest | 00.48840 |

The RMSE that averages all of the above is 0.46127848800757043

## Feature Engineering

### RoBERTa

- Only excerpt

### Text features

Text features were created based on this [notebook](notebook/create-text-features.ipynb).

The above features were selected using the Stepwise method.
I removed features to account for overfit.

## Not improve experiments

- Some custom heads
  - LSTM head
  - GRU head
  - 1DCNN head
  - roberta-base + mean pool head
- Concat last 2 hidden state layers
- SWA
- Weight initialize of custom heads and regression layer

## Development Enviroment

- Local
- GCP
- Google Colab
- Kaggle
