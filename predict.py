"""Predict a wine cultivar from 13 chemical measurements.

Usage:
    python main.py                       # train + evaluate first
    python predict.py 14.2 1.7 2.3 ...  # 13 numbers, in dataset order

Feature order (sklearn Wine dataset):
  alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
  total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
  color_intensity, hue, od280/od315_of_diluted_wines, proline
"""

import sys

import joblib
from sklearn.datasets import load_wine

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"


def main(argv) -> int:
    if len(argv) != 14:
        print(__doc__)
        print("Error: expected exactly 13 feature values.", file=sys.stderr)
        return 1
    try:
        features = [float(v) for v in argv[1:]]
    except ValueError:
        print("Error: all 13 values must be numbers.", file=sys.stderr)
        return 1
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Error: no saved model. Run `python main.py` first.",
              file=sys.stderr)
        return 1
    prediction = model.predict(scaler.transform([features]))[0]
    target_names = load_wine().target_names
    print(f"Predicted cultivar: {target_names[prediction]} (class {prediction})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
