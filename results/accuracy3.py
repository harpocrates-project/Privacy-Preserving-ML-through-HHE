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
    Only predictions whose IDs appear in *all* prediction directories are included
    in the metric calculations.
    """

    # Determine the set of prediction IDs that are present in every prediction directory
    common_ids = None
    for pred_dir in prediction_dirs:
        ids_in_dir = {
            f.name.split('_', 1)[0]
            for f in pred_dir.iterdir()
            if f.is_file()
        }
        if common_ids is None:
            common_ids = ids_in_dir
        else:
            common_ids &= ids_in_dir

    if not common_ids:
        raise ValueError("No common prediction IDs found across all prediction directories.")

    results = []
    # Iterate over each prediction directory
    for pred_dir in prediction_dirs:
        # Containers for all predictions and truths for this model
        all_preds = []
        all_truths = []

        # Match prediction files with truth files based on the ID before the first underscore
        for pred_file in pred_dir.iterdir():
            if not pred_file.is_file():
                continue

            # Extract the identifier from the prediction filename
            pred_id = pred_file.name.split('_', 1)[0]

            # Skip predictions that are not present in every dataset
            if pred_id not in common_ids:
                continue

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
            raise ValueError(f"No prediction files found in directory {pred_dir} after filtering.")

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
        base_dir / "plaintext" / "1layer" / "plaintext_binaryoutput",
        base_dir / "plaintext" / "2layer" / "plaintext_binaryoutput",
        base_dir / "hhe" / "2bit" / "hhe_binaryoutput",
        base_dir / "hhe" / "3bit" / "hhe_binaryoutput",
        base_dir / "hhe" / "4bit" / "hhe_binaryoutput",
    ]
    baseline_dir = base_dir / "baseline_binaryoutput"

    df_results = evaluate_models(prediction_dirs, baseline_dir)
    print(df_results)

# Plot the evaluation results as a bar graph
import matplotlib.pyplot as plt

# Prepare the DataFrame for plotting
df_plot = df_results.set_index("model")

# Create a grouped bar chart
ax = df_plot.plot(kind="bar", figsize=(12, 7))

# Add value labels on each bar
for container in ax.containers:
    ax.bar_label(container, fmt="{:.2f}", padding=3)

# Customize the plot
ax.set_title("Model Performance Metrics")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Metric")
plt.tight_layout()

# Display the plot
plt.show()
