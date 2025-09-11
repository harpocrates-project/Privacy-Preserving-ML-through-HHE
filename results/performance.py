import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

def load_time_files(directory: str) -> pd.DataFrame:
    """
    Load all ``.time`` files from ``directory`` into a pandas DataFrame.

    The expected filename pattern is ``{plaintext_id}.time`` (any ``_`` in the
    stem is considered part of the ``plaintext_id``). Each file should contain
    lines of the form ``<label> <seconds>`` (e.g. ``real 0.10``). The function
    returns a DataFrame where each row corresponds to one file and the time
    labels become columns.

    Parameters
    ----------
    directory:
        Path to the folder that contains the ``.time`` files.

    Returns
    -------
    pandas.DataFrame
        Columns: ``plaintext_id`` plus one column for each distinct time label
        found across the files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"'{directory}' is not a valid directory.")

    records: List[Dict[str, Any]] = []
    # Keep track of all encountered time labels so we can ensure a consistent column order
    all_labels: set = set()

    for file_path in dir_path.glob("*.time"):
        # Parse filename (everything before the first underscore, if any, is part of the ID)
        plaintext_id, _ = file_path.stem.split('_')

        # Read the file content
        time_data: Dict[str, float] = {}
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    # Skip malformed lines but keep processing the file
                    continue
                label, value_str = parts
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                time_data[label] = value
                all_labels.add(label)

        # Combine parsed info into a single record
        record: Dict[str, Any] = {"plaintext_id": plaintext_id}
        record.update(time_data)
        records.append(record)

    # Build the DataFrame; ensure missing columns are filled with NaN
    df = pd.DataFrame(records)

    # Optional: order columns (plaintext_id, then sorted time labels)
    ordered_cols = ["plaintext_id"] + sorted(all_labels)
    df = df.reindex(columns=ordered_cols)

    return df

def load_size_file(file_path: str) -> pd.DataFrame:
    """
    Load a whitespace‑separated text file containing two columns:
    ``size_kb`` and ``file_name``.  An additional column ``plaintext_id``
    is derived from ``file_name`` (the substring before the first
    underscore).

    Parameters
    ----------
    file_path : str
        Path to the file to read.

    Returns
    -------
    pandas.DataFrame
        Columns: ``size_kb``, ``file_name``, ``plaintext_id``.
    """
    # Read the file; assume any whitespace delimiter and a header row may be absent
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["size_kb", "file_name"])
    except Exception as e:
        raise ValueError(f"Unable to read '{file_path}': {e}")

    # Extract plaintext_id from file_name
    def _extract_id(name: str) -> str:
        # Split on the first underscore; if none, return the whole name
        return name.split("_", 1)[0]

    df["plaintext_id"] = df["file_name"].apply(_extract_id)

    # Ensure size_kb is numeric
    df["size_kb"] = pd.to_numeric(df["size_kb"], errors="coerce")

    return df


def compute_speed_df(sizes_df: pd.DataFrame, times_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transfer speed (kilobytes per second) for each ``plaintext_id``.

    Parameters
    ----------
    sizes_df : pd.DataFrame
        DataFrame returned by ``load_size_file``. Must contain the columns
        ``plaintext_id`` and ``size_kb``.
    times_df : pd.DataFrame
        DataFrame returned by ``load_time_files``. Must contain the column
        ``plaintext_id`` and at least one numeric time column. If a ``real``
        column is present it is used; otherwise the first numeric column
        (excluding ``plaintext_id``) is used.

    Returns
    -------
    pd.DataFrame
        Columns ``plaintext_id`` and ``speed_kb_per_s`` (kilobytes per second).
    """
    import pandas as pd

    # Determine which time column to use
    if "real" in times_df.columns:
        time_col = "real"
    else:
        # Pick the first numeric column that is not the identifier
        numeric_cols = times_df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "plaintext_id"]
        if not numeric_cols:
            raise ValueError("No numeric time column found in times_df")
        time_col = numeric_cols[0]

    # Average the chosen time column per plaintext_id
    avg_times = (
        times_df.groupby("plaintext_id")[time_col]
        .mean()
        .reset_index()
        .rename(columns={time_col: "mean_time"})
    )

    # Merge with size information
    merged = pd.merge(
        sizes_df[["plaintext_id", "size_kb"]], avg_times, on="plaintext_id", how="inner"
    )

    # Compute speed (kb / second)
    merged["speed_kb_per_s"] = merged["size_kb"] / merged["mean_time"]

    return merged[["plaintext_id", "speed_kb_per_s"]]


def plot_time_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    title: str = "Time Comparison",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``real`` time of two
    upload‑time DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames produced by ``load_time_files``.  They must contain a column
        named ``real`` or, if that column is absent, any numeric column (excluding
        ``plaintext_id``) will be used as a fallback.
    title : str, optional
        Title of the plot.
    label1, label2 : str, optional
        Labels for the two bars (default ``"Dataset 1"`` and ``"Dataset 2"``).

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import pandas as pd

    def _choose_time_column(df: pd.DataFrame) -> str:
        """Return the column name to use for averaging."""
        if "real" in df.columns:
            return "real"
        # Fallback: first numeric column that is not the identifier
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "plaintext_id"]
        if not numeric_cols:
            raise ValueError("No suitable numeric time column found in the DataFrame.")
        return numeric_cols[0]

    time_col1 = _choose_time_column(df1)
    time_col2 = _choose_time_column(df2)

    avg1 = df1[time_col1].mean()
    avg2 = df2[time_col2].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([0, 1], [avg1, avg2], color=["tab:blue", "tab:orange"], tick_label=[label1, label2])

    # Annotate bars with the average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, (avg1, avg2)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Average Time (seconds)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg1, avg2) * 1.2)  # give some headroom for the annotations
    plt.tight_layout()
    plt.show()


def plot_speed_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    title: str = "Speed Comparison",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``speed_kb_per_s`` of two
    speed DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames that contain a ``speed_kb_per_s`` column.  The function
        computes the mean of this column for each DataFrame and displays the
        results as two bars.

    title : str, optional
        Title of the plot.
    label1, label2 : str, optional
        Labels for the two bars (default ``"Dataset 1"`` and ``"Dataset 2"``).

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    # Ensure the required column exists
    if "speed_kb_per_s" not in df1.columns or "speed_kb_per_s" not in df2.columns:
        raise ValueError("Both DataFrames must contain a 'speed_kb_per_s' column.")

    # Compute the average speed for each DataFrame
    avg1 = df1["speed_kb_per_s"].mean()
    avg2 = df2["speed_kb_per_s"].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([0, 1], [avg1, avg2], color=["tab:green", "tab:red"], tick_label=[label1, label2])

    # Annotate each bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, (avg1, avg2)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Average Speed (KB/s)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg1, avg2) * 1.2)  # add headroom for annotations
    plt.tight_layout()
    plt.show()

def plot_storage_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    title: str = "Storage Comparison",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``size_kb`` of two
    size DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames produced by ``load_size_file``.  They must contain a
        ``size_kb`` column.
    title : str, optional
        Title of the plot.
    label1, label2 : str, optional
        Labels for the two bars (default ``"Dataset 1"`` and ``"Dataset 2"``).

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    # Validate input DataFrames
    if "size_kb" not in df1.columns or "size_kb" not in df2.columns:
        raise ValueError("Both DataFrames must contain a 'size_kb' column.")

    # Compute average sizes
    avg1 = df1["size_kb"].mean()
    avg2 = df2["size_kb"].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [0, 1],
        [avg1, avg2],
        color=["tab:purple", "tab:brown"],
        tick_label=[label1, label2],
    )

    # bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, (avg1, avg2)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Average Size (KB)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg1, avg2) * 1.2)  # add headroom for annotations
    plt.tight_layout()
    plt.show()

# Alias to match the typo used later in the script
plot_speed_comparision = plot_speed_comparison

hhe_sizes = load_size_file('ESADA_4percent/hhe_data.txt')
plaintext_sizes = load_size_file('ESADA_4percent/plaintext_data.txt')

hhe_upload_times = load_time_files('ESADA_4percent/hhe_upload_time')
plaintext_upload_times = load_time_files('ESADA_4percent/plaintext_upload_time')

hhe_evalulate_time = load_time_files('ESADA_4percent/hhe_evaluate_time')
plaintext_evalulate_time = load_time_files('ESADA_4percent/plaintext_evaluate_time')

plaintext_upload_speed = compute_speed_df(plaintext_sizes, plaintext_upload_times)
hhe_upload_speed = compute_speed_df(plaintext_sizes, hhe_upload_times)

plot_storage_comparison(hhe_sizes, plaintext_sizes, title="Storage Cost", label1="HHE", label2="Plaintext")
plot_speed_comparision(hhe_upload_speed, plaintext_upload_speed, title="Upload Performance", label1="HHE", label2="Plaintext")
plot_time_comparison(hhe_evalulate_time, plaintext_evalulate_time, title="Evaluation Performance", label1="HHE", label2="Plaintext")
