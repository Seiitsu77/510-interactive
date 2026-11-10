# Iris classification with logistic regression

A simple ML pipeline that predicts iris species from four flower measurements. The script loads the Iris dataset bundled with scikit-learn, so no dataset download is needed.

## Run

```sh
python -m pip install -r requirements.txt
python train.py
```

## Pipeline

1. Split the data into 120 training samples and 30 test samples using stratification and random_state=42.
2. Preprocess missing values using training-set medians with SimpleImputer. Iris has no missing values, so this step leaves its values unchanged.
3. Engineer features using PolynomialFeatures(degree=2, include_bias=False). This expands the four measurements into 14 features including the original values, their squares, and pairwise products such as petal length * petal width.
4. Standardize the engineered features using StandardScaler.
5. Train a LogisticRegression model and evaluate it on the held-out test set.

All preprocessing, feature engineering, and model training are combined in one scikit-learn Pipeline. The pipeline is fitted only on the training set to prevent data leakage.

## Result

`accuracy=0.9667` (29 of 30 test flowers classified correctly).

Measured with scikit-learn 1.6.1 on the fixed split above. This is a single hold-out result. The test split was also evaluated during development of the earlier version.
