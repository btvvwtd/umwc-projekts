import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Czytam dane z: {args.data}")

    df = pd.read_csv(args.data)

    X = df.drop("label", axis=1)
    y = df["label"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)

    acc = accuracy_score(yte, model.predict(Xte))
    print(f"Accuracy: {acc}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(
        json.dumps({"accuracy": acc}, indent=2)
    )

if __name__ == "__main__":
    main()
