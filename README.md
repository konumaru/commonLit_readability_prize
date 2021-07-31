# CommonLit Readability Prize

![background](https://user-images.githubusercontent.com/17187586/117848267-1c1b1780-b2be-11eb-8035-6fbd4e081d09.png)

## Summary

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

The RMSE of the final result is 0.4607133200425228 with 0.7 \* ensemble + 0.3 \* stacking

### Ensemble single models

- roberta-base + attention head + layer norm
  - RMSE: 0.4736949670144576
- roberta-base + attention head
  - RMSE: RMSE: 0.4709106082952253
- roberta-base-squad2 + attention head
  - RMSE: RMSE: 0.4777408091672718
- roberta-large + attention head
  - RMSE: 0.4730063474053594
- roberta-large-squad2 + attention head
  - RMSE: 0.4711161440071689
- roberta-large + mean pool head
  - RMSE: 0.47477908722085643

The RMSE that averages all of the above is 0.46214926662874833

### Stacking

- Ridge
  - RMSE: 0.462588524073919
- Baysian Ridge
  - RMSE: 0.46239263410931475
- MLP
  - RMSE: 0.5085765790075847
- SVR
  - RMSE: 0.4688521101657648
- XGB
  - RMSE: 0.4632751688391198
- Random Forest
  - RMSE: 0.4884084753993871

The RMSE that averages all of the above is 0.46127848800757043

## Feature Engineering

### RoBERTa

- Only excerpt

### Text features

Text features were created based on this [notebook](notebook/create-text-features.ipynb).
The above features were selected using the Stepwise method.

I arbitrarily removed features to account for overlearning tendencies.

## Not improve experiments

- Some custom heads
  - LSTM head
  - GRU head
  - 1DCNN head
  - roberta-base + mean pool head
- Concat last 2 hidden state layers
- SWA
- Weight initialize of custom heads and regression layer
