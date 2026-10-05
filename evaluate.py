import argparse
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    bundle = joblib.load(args.model)
    X = d["X"].mean(axis=1)
    y = d["y"]

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    Xte, yte = X[te], y[te]

    pred = bundle["model"].predict(Xte)
    labels = d["labels"]

    print("=== HELD-OUT TEST ===")
    print("Accuracy:", round(accuracy_score(yte, pred), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(yte, pred), 4))
    print(classification_report(yte, pred, target_names=labels, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(yte)
    chance = balanced_accuracy_score(yte, shuffled)
    print("\nShuffled-label sanity baseline:", round(chance, 4))
    print("IMPORTANT: this is a sanity check, not a substitute for cross-session validation.")

if __name__ == "__main__":
    main()
