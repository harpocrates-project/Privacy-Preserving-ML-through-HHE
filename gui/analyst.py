import tkinter as tk

from base import CommandLineWrapper


class AnalystWrapper(CommandLineWrapper):
    def __init__(self, master=None, default_exe=None, default_host=None, default_csp=None):
        self.host = tk.StringVar(value=default_host)
        self.csp = tk.StringVar(value=default_csp)
        super().__init__(master, default_exe)

    @property
    def terminal_cmd(self):
        exe_path = self.executable_path.get().strip()
        host = self.host.get().strip()
        csp = self.csp.get().strip()
        cmd = exe_path
        if host:
            cmd += f" {host}"
        if csp:
            cmd += f" {csp}"
        return [part.format(cmd=cmd) for part in self.terminal_cmd_template]

    def create_widgets(self):
        # Parameters frame (Host, CSP)
        param_frame = tk.Frame(self)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Host
        tk.Label(param_frame, text="Host:").pack(side=tk.LEFT)
        tk.Entry(param_frame, textvariable=self.host, width=20).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # CSP
        tk.Label(param_frame, text="CSP:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(param_frame, textvariable=self.csp, width=20).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        super().create_widgets()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Analyst")
    app = AnalystWrapper(
        master=root,
        default_exe="gui/analyst.sh",
        default_host="localhost:50051",
        default_csp="localhost:50052",
    )
    root.mainloop()
