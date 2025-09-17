import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def read_output_file(filepath: Path) -> np.ndarray:
    with open(filepath, "r") as f:
        data = [int(line.strip()) for line in f if line.strip() != ""]
    return np.asarray(data, dtype=int)


def evaluate_models(prediction_dirs, baseline_dir):
    """
    Evaluate multiple model prediction directories against a baseline truth directory.
    """

    results = []
    # Iterate over each prediction directory
    for pred_dir in prediction_dirs:
        # Containers for all predictions and truths for this model
        all_preds = []
        all_truths = []

        # Match prediction files with truth files based on the ID before the first underscore
        for pred_file in pred_dir.iterdir():
            if pred_file.is_file():
                # Extract the identifier from the prediction filename
                pred_id = pred_file.name.split('_', 1)[0]

                # Find the corresponding truth file in the baseline directory
                truth_candidates = [
                    f for f in baseline_dir.iterdir()
                    if f.is_file() and f.name.split('_', 1)[0] == pred_id
                ]

                if not truth_candidates:
                    raise FileNotFoundError(
                        f"Truth file with ID '{pred_id}' not found for prediction {pred_file}"
                    )

                # Use the first matching truth file (there should be exactly one per ID)
                truth_file = truth_candidates[0]

                preds = read_output_file(pred_file)
                truths = read_output_file(truth_file)

                # Ensure both arrays have the same length
                if preds.shape != truths.shape:
                    raise ValueError(
                        f"Shape mismatch between predictions ({preds.shape}) "
                        f"and truths ({truths.shape}) in file {pred_file.name}"
                    )

                all_preds.append(preds)
                all_truths.append(truths)

        if not all_preds:
            raise ValueError(f"No prediction files found in directory {pred_dir}")

        # Concatenate all arrays to compute metrics over the whole dataset
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_truths)

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        results.append(
            {
                "model": f"{pred_dir.parent.name}/{pred_dir.name}",
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    base_dir = Path("ESADA_4percent")

    prediction_dirs = [
        base_dir / "1layer" / "plaintext_binaryoutput",
        base_dir / "2bit" / "hhe_binaryoutput",
        base_dir / "2layer" / "plaintext_binaryoutput",
        base_dir / "3bit" / "hhe_binaryoutput",
    ]
    baseline_dir = base_dir / "baseline_binaryoutput"

    df_results = evaluate_models(prediction_dirs, baseline_dir)
    print(df_results)
