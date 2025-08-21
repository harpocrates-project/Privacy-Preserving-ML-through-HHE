import subprocess
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class UserExecutableGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("User Executable Wrapper")
        self.geometry("1100x800")
        self.resizable(True, True)

        # Path to the executable (configurable) with default value
        self.executable_path = tk.StringVar()
        self.executable_path.set("./build/user")
        self.plot_canvas = None          # Will hold the Matplotlib canvas
        self.plot_frame = None           # Frame that contains the canvas
        self.create_widgets()

    def create_widgets(self):
        row = 0

        # Executable path selector
        tk.Label(self, text="Executable Path:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        exe_entry = tk.Entry(self, textvariable=self.executable_path, width=40, state="readonly")
        exe_entry.grid(row=row, column=1, sticky="w", padx=5)
        tk.Button(self, text="Browse...", command=self.browse_executable).grid(row=row, column=2, padx=5)
        row += 1

        # Analyst URL
        tk.Label(self, text="Analyst URL:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.analyst_url = tk.Entry(self, width=45)
        self.analyst_url.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self.analyst_url.insert(0, "localhost:50051")  # default value
        row += 1

        # CSP URL
        tk.Label(self, text="CSP URL:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.csp_url = tk.Entry(self, width=45)
        self.csp_url.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self.csp_url.insert(0, "localhost:50052")  # default value
        row += 1

        # SpO2 Data file selector
        tk.Label(self, text="SpO2 Data File:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.spo2_path = tk.StringVar()
        spo2_entry = tk.Entry(self, textvariable=self.spo2_path, width=40, state="readonly")
        spo2_entry.grid(row=row, column=1, sticky="w", padx=5)
        tk.Button(self, text="Browse...", command=self.browse_spo2_data).grid(row=row, column=2, padx=5)
        row += 1

        # Records (auto-populated)
        tk.Label(self, text="Records:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        self.records_var = tk.StringVar()
        records_entry = tk.Entry(self, textvariable=self.records_var, width=45, state="readonly")
        records_entry.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        row += 1

        # Plot area (will be placed above the Run button)
        self.plot_frame = tk.Frame(self)
        self.plot_frame.grid(row=row, column=0, columnspan=3, pady=10, sticky="nsew")
        row += 1

        # Run button
        tk.Button(self, text="Run", command=self.run_executable, width=10).grid(row=row, column=1, pady=15)

    def browse_executable(self):
        path = filedialog.askopenfilename(
            title="Select 'user' Executable",
            filetypes=[("All Files", "*")],
        )
        if path:
            self.executable_path.set(path)

    def browse_spo2_data(self):
        path = filedialog.askopenfilename(
            title="Select SpO2 Data File",
            filetypes=[("All Files", "*.*")],
        )
        if path:
            self.spo2_path.set(path)
            self.update_record_count(path)
            self.show_spo2_plot(path)

    def update_record_count(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                line_count = len(f.readlines())
            self.records_var.set(str(line_count))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to count lines in file:\n{e}")
            self.records_var.set("")

    def show_spo2_plot(self, file_path):
        """Generate a Matplotlib figure from the SpO2 file and embed it in the GUI."""
        try:
            fig = plot_spo2_data(file_path)

            # Remove previous canvas if it exists
            if self.plot_canvas is not None:
                self.plot_canvas.get_tk_widget().destroy()
                self.plot_canvas = None

            self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not generate plot:\n{e}")

    def run_executable(self):
        exe = self.executable_path.get()
        analyst = self.analyst_url.get().strip()
        csp = self.csp_url.get().strip()
        spo2 = self.spo2_path.get()
        records = self.records_var.get()

        # Basic validation
        if not all([exe, analyst, csp, spo2, records]):
            messagebox.showwarning("Missing Information", "Please fill in all fields before running.")
            return

        # Create a new window for real‑time output
        output_win = tk.Toplevel(self)
        output_win.title("Executable Output")
        output_win.geometry("800x600")

        txt = tk.Text(output_win, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        y_scroll = tk.Scrollbar(output_win, command=txt.yview)
        y_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        txt.configure(yscrollcommand=y_scroll.set)  # Queue to safely transfer lines from threads to the Tkinter mainloop
        out_queue = queue.Queue()

        def enqueue_output(pipe, tag):
            """Read lines from pipe and put them in the queue with an optional tag."""
            for line in iter(pipe.readline, ""):
                out_queue.put((tag, line))
            pipe.close()

        # Start the subprocess
        try:
            cmd = [exe, analyst, csp, spo2, records]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            messagebox.showerror("Execution Error", f"An error occurred while starting the executable:\n{e}")
            output_win.destroy()
            return

        # Launch threads to read stdout and stderr
        threading.Thread(target=enqueue_output, args=(proc.stdout, "STDOUT"), daemon=True).start()
        threading.Thread(target=enqueue_output, args=(proc.stderr, "STDERR"), daemon=True).start()

        def poll_queue():
            """Insert any queued lines into the Text widget."""
            try:
                while True:
                    tag, line = out_queue.get_nowait()
                    if tag == "STDERR":
                        txt.insert(tk.END, line, "stderr")
                    else:
                        txt.insert(tk.END, line)
                    txt.see(tk.END)
            except queue.Empty:
                pass

            if proc.poll() is None:
                # Process still running; continue polling
                self.after(100, poll_queue)
            else:
                # Process finished; show return code
                retcode = proc.returncode
                txt.insert(tk.END, f"\nProcess finished with return code {retcode}\n")
                txt.see(tk.END)

        # Tag styling for stderr (optional)
        txt.tag_configure("stderr", foreground="red")

        # Start polling the queue
        self.after(100, poll_queue)


def plot_spo2_data(file_path):
    """Read SpO2 data from a file where each line represents a 5‑minute segment.
    Each line is a comma‑separated list of integer samples recorded at 1 Hz.
    The function concatenates all segments and returns a Matplotlib Figure."""
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

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_hours, data, label='SpO2')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('SpO2 (%)')
    ax.set_title('SpO2 Signal')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    app = UserExecutableGUI()
    app.mainloop()
