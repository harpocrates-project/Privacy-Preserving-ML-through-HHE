import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox


class CommandLineWrapper(tk.Frame):
    """
    Base class for a Tkinter GUI that wraps a command‑line executable.
    It provides a file selector for the executable and a button to launch it
    inside the system's default terminal emulator (Linux only).
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.executable_path = tk.StringVar()
        self.terminal_cmd_template = ["x-terminal-emulator", "-e", "{cmd}"]
        self.create_widgets()
        self.pack()

    @property
    def terminal_cmd(self):
        exe_path = self.executable_path.get().strip()
        return [part.format(cmd=exe_path) for part in self.terminal_cmd_template]

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("All files", "*")],
        )
        return file_path

    def browse_executable(self):
        file_path = self.browse_file()
        if file_path:
            self.executable_path.set(file_path)

    def create_widgets(self):
        path_frame = tk.Frame(self)
        path_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(path_frame, text="Executable:").pack(side=tk.LEFT)

        entry = tk.Entry(path_frame, textvariable=self.executable_path, width=50)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        browse_btn = tk.Button(
            path_frame, text="Browse...", command=self.browse_executable
        )
        browse_btn.pack(side=tk.LEFT, padx=5)

        run_btn = tk.Button(
            path_frame, text="Run", command=self.run_executable
        )
        run_btn.pack(side=tk.LEFT, padx=5)

    def run_executable(self):
        try:
            subprocess.Popen(self.terminal_cmd)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to launch the terminal emulator:\n{e}",
            )
