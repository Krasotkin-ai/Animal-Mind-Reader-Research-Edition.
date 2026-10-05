import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def features(X):
    # Trial-level firing rate over the available window.
    return X.mean(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = features(d["X"])
    y = d["y"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced")
    )
    model.fit(Xtr, ytr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "test_X": Xte,
            "test_y": yte,
            "labels": d["labels"],
        },
        out,
    )
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
