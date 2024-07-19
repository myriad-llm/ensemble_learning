# machine-learning

## data

```shell
mkdir -p data/raw
```

`./data/raw` used to store raw data.

`./data/processed` used to store processed data.

## use

`ALL`: `ALL=True` means use all labeled data to train model. `ALL=False` means use part of labeled data to train model, and the rest of labeled data is used to test model.

`SMOTE`: `SMOTE=True` means use SMOTE to balance data. `SMOTE=False` means not use SMOTE to balance data. What should be noted is that data augmentation is also used to balance data and after that, the `SMOTE` is used.