import matplotlib.pyplot as plt


def plot_binaryoutput(file_path):
    """
    Reads a binary output file where each line corresponds to a 5‑minute segment.
    0 indicates no desaturation event, 1 indicates an event.
    Plots the detection signal versus time.
    """
    # Read the file and convert each line to an integer (skip empty lines)
    with open(file_path, 'r') as f:
        values = [int(line.strip()) for line in f if line.strip() != '']

    if not values:
        raise ValueError("The input file contains no data.")

    # Create a time axis in minutes (or hours) – each entry is 5 minutes apart
    segment_length_min = 5
    time_minutes = [i * segment_length_min for i in range(len(values))]
    # Optionally convert to hours for a nicer x‑axis
    time_hours = [t / 60.0 for t in time_minutes]

    # Plot
    plt.figure(figsize=(10, 3))
    plt.step(time_hours, values, where='post', label='Desaturation event')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['No event', 'Event'])
    plt.xlabel('Time (hours)')
    plt.title('Desaturation Event Detection')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return plt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot desaturation detection from a binary output file."
    )
    parser.add_argument(
        "file",
        help="Path to the binary output text file containing 0/1 values per 5‑minute segment.",
    )
    args = parser.parse_args()

    plot_binaryoutput(args.file)
