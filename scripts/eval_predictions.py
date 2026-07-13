"""Score inference.py outputs against the ground-truth column.

Usage:
    python scripts/eval_predictions.py LABEL:path/to/pred.csv [LABEL:another.csv ...]

Each CSV must contain the truth column (default `Score`) and the prediction
column written by inference.py (`pred_log10_kcat`). Prints one row per file.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred):
    return {
        "PCC": pearsonr(y_true, y_pred)[0],
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    p = argparse.ArgumentParser(description="Score inference.py outputs")
    p.add_argument("inputs", nargs="+", help="LABEL:path.csv pairs")
    p.add_argument("--label_col", default="Score")
    p.add_argument("--pred_col", default="pred_log10_kcat")
    args = p.parse_args()

    rows = []
    for item in args.inputs:
        label, _, path = item.partition(":")
        df = pd.read_csv(path)
        for col in (args.label_col, args.pred_col):
            if col not in df.columns:
                raise KeyError(f"{path}: missing column '{col}'")
        m = metrics(df[args.label_col].to_numpy(), df[args.pred_col].to_numpy())
        m["n"] = len(df)
        m["model"] = label
        rows.append(m)

    out = pd.DataFrame(rows)[["model", "n", "PCC", "RMSE", "MAE", "R2"]]
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
