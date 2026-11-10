# Iris classification with logistic regression

This Python script predicts iris species from four flower measurements using the Iris dataset bundled with scikit-learn.

## Run

```sh
python -m pip install -r requirements.txt
python train.py
```

## Result

`accuracy=0.9333` (28 of 30 test flowers classified correctly).

The script uses a stratified split with 120 flowers for training and 30 for testing. A fixed random seed of 42 makes the split reproducible. The pipeline fits the feature scaler and logistic regression model on the training set, then measures accuracy on the held-out test set.
