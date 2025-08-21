import numpy as np
import matplotlib.pyplot as plt


def plot_spo2_data(file_path):
    """Read SpO2 data from a file where each line represents a 5‑minute segment.
    Each line is a comma‑separated list of integer samples recorded at 1 Hz.
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
    # Convert sample indices (seconds) to hours for the x‑axis
    time_hours = np.arange(len(data)) / 3600.0  # 1 Hz sampling → one sample per second

    plt.figure(figsize=(10, 4))
    plt.plot(time_hours, data, label='SpO2')
    plt.xlabel('Time (hours)')
    plt.ylabel('SpO2 (%)')
    plt.title('SpO2 Signal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return plt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot SpO2 data from a file.')
    parser.add_argument('file', help='Path to the data file')
    args = parser.parse_args()

    plot_spo2_data(args.file)
