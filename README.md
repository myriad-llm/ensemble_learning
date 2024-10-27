# ensemble-learning

## requirements

- [Installing AutoGluon - AutoGluon 1.1.1 documentation](https://auto.gluon.ai/stable/install.html)
- scikit-learn
- imbalanced-learn

## data

`./self_data/processed` used to cache processed data. After augmenting data and train-test spliting, the processed data will be saved in this folder. If you want to re-augment data, you must delete this folder and rerun your code.

## use

`ALL`: `ALL=True` means use all labeled data to train model. `ALL=False` means use part of labeled data to train model, and the rest of labeled data is used to test model.

`NO_SMOTE`: `NO_SMOTE=False` means use SMOTE to balance data.

`TEST_RAITO`: `TEST_RAITO=0.2` means 20% original labeled data is used to test model.

## dev

`*.ipynb` in `./code` folder without cell output.

In `./best_archive` folder:

1. `*.ipynb`  with cell output to store the best model.
2. `AutogluonModels` to store the best model.
3. `*.csv` to store the best model's prediction.
4. all code files in `./code` folder.