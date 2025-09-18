#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def scale_to_spo2(value):
    """
    Scale numeric input(s) from the range 0‑30 to the range 70‑100.

    Parameters
    ----------
    value : int, float, list, tuple, or numpy.ndarray
        Value(s) to be scaled. Values outside the 0‑30 interval are clipped
        to the nearest bound before scaling.

    Returns
    -------
    numpy.ndarray or scalar
        Scaled value(s) in the range 70‑100. The return type matches the
        input type: a scalar if a scalar was provided, otherwise a NumPy
        array.
    """
    # Convert to NumPy array for vectorised operations; keep scalars as‑is
    is_scalar = np.isscalar(value)
    arr = np.array(value, dtype=float) if not is_scalar else value

    # Clip to the expected input range
    if not is_scalar:
        np.clip(arr, 0, 30, out=arr)
    else:
        arr = max(0, min(30, arr))

    # Linear scaling: 0 → 70, 30 → 100
    scaled = 70 + (arr / 30.0) * 30

    return scaled.item() if is_scalar else scaled


def plot_spo2_data(file_path):
    """Read SpO2 data from a file where each line represents a 5‑minute segment.
    Each line is a comma‑separated list of integer samples recorded at 1 Hz.
    The function concatenates all segments and plots the signal versus time (hours)."""
    # Collect all samples from the file
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            # Convert comma‑separated values to integers
            segment = [int(val) for val in line.split(',') if val]
            samples.extend(segment)

    if not samples:
        raise ValueError("No SpO2 data found in the provided file.")

    data = np.array(samples)
    data = scale_to_spo2(data)
    # Convert sample indices (seconds) to hours for the x‑axis
    time_hours = np.arange(len(data)) / 3600.0  # 1 Hz sampling → one sample per second

    plt.figure(figsize=(10, 4))
    plt.plot(time_hours, data, label='SpO2')
    plt.xlabel('Time (hours)')
    plt.ylabel('SpO2 (%)')
    plt.title('Blood Oxygen Saturation (1Hz)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    axes = plt.axes((0.9, 0.01, 0.06, 0.06))
    btn_ok = Button(axes, "OK")
    btn_ok.on_clicked(lambda _: plt.close())
    plt.show()

    return plt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot SpO2 data from a file.')
    parser.add_argument('file', help='Path to the data file')
    args = parser.parse_args()

    plot_spo2_data(args.file)
