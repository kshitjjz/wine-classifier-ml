"""Structured classification pipeline: wine cultivar prediction.

Dataset: scikit-learn's bundled Wine dataset (178 samples, 13 chemical
features, 3 cultivars). It ships with scikit-learn, so the project runs
offline with no downloads and no synthetic data.

Pipeline steps, each one explainable on its own:
  1. Load and inspect the data (shapes, class balance, missing values).
  2. Preprocess: stratified train/test split, standard scaling fitted
     on the training set only (fitting on the full data would leak
     test information into the scaler).
  3. Train three models: logistic regression, random forest, SVM.
  4. Evaluate on the held-out test set: accuracy, classification
     report, confusion matrix.
  5. Save the best model + scaler for predict.py.

Run:  python main.py
"""

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
TEST_SIZE = 0.25
RANDOM_STATE = 42


def load_data():
    data = load_wine()
    print(f"Samples: {data.data.shape[0]}, features: {data.data.shape[1]}")
    print(f"Classes: {list(data.target_names)}")
    unique, counts = np.unique(data.target, return_counts=True)
    print(f"Class balance: {dict(zip(data.target_names[unique], counts))}")
    print(f"Missing values: {int(np.isnan(data.data).sum())}")
    return data.data, data.target, data.feature_names, data.target_names


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=5000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
        "svm": SVC(kernel="rbf"),
    }


def main():
    X, y, feature_names, target_names = load_data()

    # Stratified split keeps the class ratio identical in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Scale AFTER splitting; the scaler learns only from training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, model in build_models().items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = (acc, model)
        print(f"\n{name}: accuracy = {acc:.4f}")

    best_name = max(results, key=lambda n: results[n][0])
    best_acc, best_model = results[best_name]
    print(f"\nBest model: {best_name} (accuracy {best_acc:.4f})")
    print("\nClassification report (best model):")
    print(classification_report(
        y_test, best_model.predict(X_test_scaled), target_names=target_names))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_test, best_model.predict(X_test_scaled)))

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        ranked = sorted(zip(feature_names, importances),
                        key=lambda pair: pair[1], reverse=True)
        print("\nTop features by importance:")
        for feature, importance in ranked[:5]:
            print(f"  {feature}: {importance:.3f}")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved best model to {MODEL_PATH} and scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
