import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

from base import CommandLineWrapper


class UserWrapper(CommandLineWrapper):
    def __init__(self, master=None):
        self.analyst = tk.StringVar()
        self.csp = tk.StringVar()
        self.spo2_data = tk.StringVar()
        self.num_records = tk.IntVar()
        super().__init__(master)
        # Update number of records whenever the SpO2 data file changes
        self.spo2_data.trace_add("write", lambda *args: self.update_num_records())

        @property
        def terminal_cmd(self):
            exe_path = self.executable_path.get().strip()
            analyst = self.analyst.get().strip()
            csp = self.csp.get().strip()
            spo2 = self.spo2_data.get().strip()
            num = self.num_records.get()

            cmd = exe_path
            if analyst:
                cmd += f" {analyst}"
            if csp:
                cmd += f" {csp}"
            if spo2:
                cmd += f" {spo2}"
            if num:
                cmd += f" {num}"

            return [part.format(cmd=cmd) for part in self.terminal_cmd_template]

    def browse_spo2_data(self):
        file_path = filedialog.askopenfilename(
            title="Select SpO2 Data file",
            filetypes=[("All files", "*")],
        )
        if file_path:
            self.spo2_data.set(file_path)

    def update_num_records(self):
        path = self.spo2_data.get()
        if path and os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    # Simple line count (ignoring empty lines)
                    count = sum(1 for line in f if line.strip())
                self.num_records.set(count)
            except Exception:
                self.num_records.set(0)
        else:
            self.num_records.set(0)

    def create_widgets(self):
        # Parameters frame (Analyst, CSP, SpO2 Data, Number of Records)
        param_frame = tk.Frame(self)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Analyst
        tk.Label(param_frame, text="Analyst:").pack(side=tk.LEFT)
        tk.Entry(param_frame, textvariable=self.analyst, width=15).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # CSP
        tk.Label(param_frame, text="CSP:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(param_frame, textvariable=self.csp, width=15).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # SpO2 Data
        tk.Label(param_frame, text="SpO2 Data:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(param_frame, textvariable=self.spo2_data, width=25).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        tk.Button(
            param_frame, text="Browse...", command=self.browse_spo2_data
        ).pack(side=tk.LEFT, padx=5)

        # Number of Records (read‑only)
        tk.Label(param_frame, text="Number of Records:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(
            param_frame,
            textvariable=self.num_records,
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Add the standard executable/browse/run widgets
        super().create_widgets()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("User")
    app = UserWrapper(master=root)
    root.mainloop()
