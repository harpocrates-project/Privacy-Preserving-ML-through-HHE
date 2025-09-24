import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

# Increase default font size for all plot elements
plt.rcParams.update({'font.size': 16})

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
    *dfs: pd.DataFrame,
    title: str = "Time Comparison",
    labels: list[str] | None = None,
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average time of an arbitrary
    number of upload‑time DataFrames.

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or more DataFrames produced by ``load_time_files``.  Each must
        contain a column named ``real`` or, if that column is absent, any numeric
        column (excluding ``plaintext_id``) will be used as a fallback.
    title : str, optional
        Title of the plot.
    labels : list[str], optional
        Human‑readable labels for the bars.  If omitted, default labels
        ``["Dataset 1", "Dataset 2", ...]`` are generated.  The length of
        ``labels`` must match the number of supplied DataFrames.

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    if not dfs:
        raise ValueError("At least one DataFrame must be provided.")

    def _choose_time_column(df: pd.DataFrame) -> str:
        """Return the column name to use for averaging."""
        if "real" in df.columns:
            return "real"
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "plaintext_id"]
        if not numeric_cols:
            raise ValueError("No suitable numeric time column found in a DataFrame.")
        return numeric_cols[0]

    # Determine the column to average for each DataFrame and compute the averages
    avgs: list[float] = []
    for df in dfs:
        time_col = _choose_time_column(df)
        avgs.append(df[time_col].mean())

    # Prepare labels
    if labels is None:
        labels = [f"Dataset {i}" for i in range(1, len(dfs) + 1)]
    else:
        if len(labels) != len(dfs):
            raise ValueError(
                "The number of labels must match the number of DataFrames provided."
            )

    # Choose colors (default matplotlib tab10 palette)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(dfs))]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    x_positions = range(len(dfs))
    bars = ax.bar(
        x_positions,
        avgs,
        color=colors,
        tick_label=labels,
    )

    # Annotate each bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, avgs):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_ylabel("Average Time (seconds)")
    ax.set_title(title)
    ax.set_ylim(0, max(avgs) * 1.2 if avgs else 1)  # headroom for annotations
    plt.tight_layout()
    plt.show()


def plot_speed_comparison(
    *dfs: pd.DataFrame,
    title: str = "Speed Comparison",
    labels: list[str] | None = None,
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``speed_kb_per_s``
    of an arbitrary number of speed DataFrames.

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or more DataFrames that contain a ``speed_kb_per_s`` column.
    title : str, optional
        Title of the plot.
    labels : list[str], optional
        Human‑readable labels for the bars.  If omitted, default labels
        ``["Dataset 1", "Dataset 2", ...]`` are generated.  The length of
        ``labels`` must match the number of supplied DataFrames.

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    if not dfs:
        raise ValueError("At least one DataFrame must be provided.")

    # Validate that every DataFrame contains a ``speed_kb_per_s`` column
    for i, df in enumerate(dfs, start=1):
        if "speed_kb_per_s" not in df.columns:
            raise ValueError(f"DataFrame #{i} does not contain a 'speed_kb_per_s' column.")

    # Compute the average speed for each DataFrame (convert to plain float)
    avgs: list[float] = [float(df["speed_kb_per_s"].mean()) for df in dfs]

    # Prepare labels
    if labels is None:
        labels = [f"Dataset {i}" for i in range(1, len(dfs) + 1)]
    else:
        if len(labels) != len(dfs):
            raise ValueError(
                "The number of labels must match the number of DataFrames provided."
            )

    # Choose a color cycle (use the default matplotlib tab10 palette)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(dfs))]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    x_positions = range(len(dfs))
    bars = ax.bar(
        x_positions,
        avgs,
        color=colors,
        tick_label=labels,
    )

    # Annotate each bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, avgs):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_ylabel("Average Speed (KB/s)")
    ax.set_title(title)
    ax.set_ylim(0, max(avgs) * 1.2 if avgs else 1)  # add headroom for annotations
    plt.tight_layout()
    plt.show()


def plot_storage_comparison(
    *dfs: pd.DataFrame,
    title: str = "Storage Comparison",
    labels: list[str] | None = None,
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``size_kb`` of an
    arbitrary number of size DataFrames.

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or more DataFrames produced by ``load_size_file``.  Each must
        contain a ``size_kb`` column.
    title : str, optional
        Title of the plot.
    labels : list[str], optional
        Human‑readable labels for the bars.  If omitted, default labels
        ``["Dataset 1", "Dataset 2", ...]`` are generated.  The length of
        ``labels`` must match the number of supplied DataFrames.

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    if not dfs:
        raise ValueError("At least one DataFrame must be provided.")

    # Validate that every DataFrame contains a ``size_kb`` column
    for i, df in enumerate(dfs, start=1):
        if "size_kb" not in df.columns:
            raise ValueError(f"DataFrame #{i} does not contain a 'size_kb' column.")

    # Compute the average size for each DataFrame
    avgs: list[float] = [df["size_kb"].mean() for df in dfs]

    # Prepare labels
    if labels is None:
        labels = [f"Dataset {i}" for i in range(1, len(dfs) + 1)]
    else:
        if len(labels) != len(dfs):
            raise ValueError(
                "The number of labels must match the number of DataFrames provided."
            )

    # Choose a color cycle (use the default matplotlib tab10 palette)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(dfs))]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    x_positions = range(len(dfs))
    bars = ax.bar(
        x_positions,
        avgs,
        color=colors,
        tick_label=labels,
    )

    # Annotate each bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, avgs):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_ylabel("Average Size (KB)")
    ax.set_title(title)
    # Add some headroom for the annotation text
    ax.set_ylim(0, max(avgs) * 1.2 if avgs else 1)

    plt.tight_layout()
    plt.show()


def filter_dataframes_to_common_ids(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Return a list of DataFrames containing only the rows whose ``plaintext_id``
    appears in **all** supplied DataFrames.

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or more DataFrames that each contain a ``plaintext_id`` column.

    Returns
    -------
    list[pd.DataFrame]
        A list with the same length as ``dfs``.  Each element is a copy of the
        corresponding input DataFrame, filtered to the intersection of
        ``plaintext_id`` values across all inputs.

    Raises
    ------
    ValueError
        If any supplied DataFrame does not contain a ``plaintext_id`` column.
    """
    if not dfs:
        raise ValueError("At least one DataFrame must be provided.")

    # Verify that every DataFrame has the required column
    for i, df in enumerate(dfs, start=1):
        if "plaintext_id" not in df.columns:
            raise ValueError(f"DataFrame #{i} does not contain a 'plaintext_id' column.")

    # Compute the intersection of all plaintext_id sets
    common_ids = set(dfs[0]["plaintext_id"].dropna())
    for df in dfs[1:]:
        common_ids.intersection_update(df["plaintext_id"].dropna())

    # If there is no common id, return empty DataFrames with the same columns
    if not common_ids:
        return [df.iloc[0:0].copy() for df in dfs]

    # Filter each DataFrame to only rows with ids in the intersection
    filtered_dfs = [
        df[df["plaintext_id"].isin(common_ids)].reset_index(drop=True).copy()
        for df in dfs
    ]

    return filtered_dfs

hhe_2bit_sizes = load_size_file('ESADA_4percent/hhe/2bit/hhe_data.txt')
hhe_4bit_sizes = load_size_file('ESADA_4percent/hhe/4bit/hhe_data.txt')
plaintext_sizes = load_size_file('ESADA_4percent/plaintext/plaintext_data.txt')

hhe_2bit_upload_times = load_time_files('ESADA_4percent/hhe/2bit/hhe_upload_time')
hhe_3bit_upload_times = load_time_files('ESADA_4percent/hhe/3bit/hhe_upload_time')
hhe_4bit_upload_times = load_time_files('ESADA_4percent/hhe/4bit/hhe_upload_time')
plaintext_upload_times = load_time_files('ESADA_4percent/plaintext/2layer/plaintext_upload_time')

hhe_2bit_evaluate_time = load_time_files('ESADA_4percent/hhe/2bit/hhe_evaluate_time')
hhe_3bit_evaluate_time = load_time_files('ESADA_4percent/hhe/3bit/hhe_evaluate_time')
hhe_4bit_evaluate_time = load_time_files('ESADA_4percent/hhe/4bit/hhe_evaluate_time')
plaintext_evaluate_time = load_time_files('ESADA_4percent/plaintext/2layer/plaintext_evaluate_time')

(
    hhe_2bit_sizes,
    hhe_4bit_sizes,
    plaintext_sizes,
    hhe_2bit_upload_times,
    hhe_3bit_upload_times,
    hhe_4bit_upload_times,
    plaintext_upload_times,
    hhe_2bit_evaluate_time,
    hhe_3bit_evaluate_time,
    hhe_4bit_evaluate_time,
    plaintext_evaluate_time,
) = filter_dataframes_to_common_ids(
    hhe_2bit_sizes,
    hhe_4bit_sizes,
    plaintext_sizes,
    hhe_2bit_upload_times,
    hhe_3bit_upload_times,
    hhe_4bit_upload_times,
    plaintext_upload_times,
    hhe_2bit_evaluate_time,
    hhe_3bit_evaluate_time,
    hhe_4bit_evaluate_time,
    plaintext_evaluate_time,
)

hhe_2bit_upload_speed = compute_speed_df(plaintext_sizes, hhe_2bit_upload_times)
hhe_3bit_upload_speed = compute_speed_df(plaintext_sizes, hhe_3bit_upload_times)
hhe_4bit_upload_speed = compute_speed_df(plaintext_sizes, hhe_4bit_upload_times)
plaintext_upload_speed = compute_speed_df(plaintext_sizes, plaintext_upload_times)



plot_storage_comparison(
    hhe_2bit_sizes,
    hhe_4bit_sizes,
    plaintext_sizes,
    title="Average Storage Cost per File",
    labels=["hhe/1layer/2bit", "hhe/1layer/4bit", "plaintext"],
)

plot_time_comparison(
    hhe_2bit_upload_times,
    hhe_3bit_upload_times,
    hhe_4bit_upload_times,
    plaintext_upload_times,
    title="Average Upload Time per File",
    labels=["hhe/1layer/2bit", "hhe/1layer/3bit", "hhe/1layer/4bit", "plaintext/*/*"],
)

# Plot upload speed comparison
plot_speed_comparison(
    hhe_2bit_upload_speed,
    hhe_3bit_upload_speed,
    hhe_4bit_upload_speed,
    plaintext_upload_speed,
    title="Average Upload Speed",
    labels=["hhe/1layer/2bit", "hhe/1layer/3bit", "hhe/1layer/4bit", "plaintext/*/*"],
)

plot_time_comparison(
    hhe_2bit_evaluate_time,
    hhe_3bit_evaluate_time,
    hhe_4bit_evaluate_time,
    plaintext_evaluate_time,
    title="Average Model Evalution Time per File",
    labels=["hhe/1layer/2bit", "hhe/1layer/3bit", "hhe/1layer/4bit", "plaintext/2layer/float"],
)
