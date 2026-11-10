from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)
    print(f"accuracy={model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
