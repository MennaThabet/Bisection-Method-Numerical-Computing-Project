# GUI:
import threading
import traceback
from typing import Any, Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math

# import backend and utils
from bisection import bisection
import utils

# ----------------- GUI -----------------
class BisectionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Bisection Method — Numerical Project")
        root.geometry("900x640")
        root.minsize(820, 560)

        # style
        style = ttk.Style()
        style.theme_use("clam")  # modern-ish builtin theme
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("TButton", padding=6)
        style.configure("Small.TButton", padding=4)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        # top frame: inputs
        top = ttk.Frame(root, padding=(12,10))
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Function f(x):", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.func_var = tk.StringVar(value="x^3 - x - 2")
        self.func_entry = ttk.Entry(top, textvariable=self.func_var, font=("Segoe UI", 11))
        self.func_entry.grid(row=0, column=1, columnspan=5, sticky="we", padx=(8,0))

        ttk.Label(top, text="Interval [a, b]:").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.a_var = tk.StringVar(value="1.0")
        self.b_var = tk.StringVar(value="2.0")
        self.a_entry = ttk.Entry(top, width=12, textvariable=self.a_var)
        self.b_entry = ttk.Entry(top, width=12, textvariable=self.b_var)
        self.a_entry.grid(row=1, column=1, sticky="w", padx=(8,2), pady=(8,0))
        self.b_entry.grid(row=1, column=2, sticky="w", padx=(4,2), pady=(8,0))

        ttk.Label(top, text="Digits (d):").grid(row=1, column=3, sticky="w", pady=(8,0))
        self.d_var = tk.StringVar(value="6")
        self.d_entry = ttk.Entry(top, width=6, textvariable=self.d_var)
        self.d_entry.grid(row=1, column=4, sticky="w", pady=(8,0))

        # sample functions dropdown
        ttk.Label(top, text="Samples:").grid(row=2, column=0, sticky="w", pady=(8,0))
        samples = ["x^3 - x - 2", "x - 1", "x*sin(x) - 1", "e^x - 3", "1/(x-2)", "sin(x)", "1/sin(x)", "sqrt(x)", "ln(x)",  "(x+1)(x-1)"]
        self.sample_var = tk.StringVar(value=samples[0])
        self.sample_combo = ttk.Combobox(top, values=samples, textvariable=self.sample_var, state="readonly")
        self.sample_combo.grid(row=2, column=1, columnspan=2, sticky="we", padx=(8,0), pady=(8,0))
        ttk.Button(top, text="Use sample", command=self._use_sample, style="Small.TButton").grid(row=2, column=3, padx=(6,0), pady=(8,0))

        # Buttons: Run / Export
        button_frame = ttk.Frame(top)
        button_frame.grid(row=0, column=6, rowspan=3, padx=(12,0), sticky="n")

        self.run_btn = ttk.Button(button_frame, text="Run Bisection", command=self.on_run, width=16)
        self.run_btn.pack(pady=(0,8))
        self.export_btn = ttk.Button(button_frame, text="Export CSV", command=self.on_export, width=16, state="disabled")
        self.export_btn.pack(pady=(0,8))
        ttk.Button(button_frame, text="Clear Table", command=self.on_clear, width=16).pack(pady=(0,8))

        # progress bar / status
        self.status_var = tk.StringVar(value="Ready")
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.status = ttk.Label(root, textvariable=self.status_var)
        self.progress.pack(side="top", fill="x", padx=12, pady=(6,0))
        self.status.pack(side="top", anchor="w", padx=12, pady=(4,8))

        # Results: summary + treeview table
        results_frame = ttk.Frame(root, padding=(12,8))
        results_frame.pack(side="top", fill="both", expand=True)

        # Make results_frame grid expand nicely
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        # Summary area with scrollbar (vertical) on the right
        summary_frame = ttk.Frame(results_frame)
        summary_frame.grid(row=0, column=0, sticky="we")
        summary_frame.columnconfigure(0, weight=1)

        ttk.Label(summary_frame, text="Summary:", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        # text + scrollbar in same row
        text_container = ttk.Frame(summary_frame)
        text_container.grid(row=1, column=0, sticky="we", pady=(6,0))
        text_container.columnconfigure(0, weight=1)

        self.summary_text = tk.Text(text_container, height=6, wrap="word", font=("Segoe UI", 10))
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        self.summary_text.configure(state="disabled")

        summary_vsb = ttk.Scrollbar(text_container, orient="vertical", command=self.summary_text.yview)
        summary_vsb.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=summary_vsb.set)

        # Treeview for iterations
        tree_frame = ttk.Frame(results_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(8,0))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        cols = ("n","a","b","p","f(p)","error")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=110)
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        # keep last iterations for export
        self._last_iterations: Optional[List[Dict[str, Any]]] = None

    def _use_sample(self):
        self.func_var.set(self.sample_var.get())

    def on_clear(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")
        self._last_iterations = None
        self.export_btn.configure(state="disabled")
        self.status_var.set("Cleared")

    def on_export(self):
        if not self._last_iterations:
            messagebox.showinfo("Nothing to export", "No iterations to export.")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files","*.*")],
            title="Save iterations as CSV"
        )
        if not fname:
            return
        try:
            utils.ensure_dir_for_file(fname)
            utils.save_iterations_to_csv(self._last_iterations, fname)
            messagebox.showinfo("Saved", f"Saved iterations to:\n{fname}")
        except Exception as exc:
            messagebox.showerror("Save error", f"Failed to save CSV:\n{exc}")

    def on_run(self):
        func = self.func_var.get().strip()
        try:
            a = float(self.a_var.get().strip())
            b = float(self.b_var.get().strip())
            d = int(self.d_var.get().strip())
        except Exception:
            messagebox.showerror("Input error", "Please enter valid numeric values for a, b and integer d.")
            return
        if a >= b:
            messagebox.showerror("Interval error", "Left endpoint a must be less than right endpoint b.")
            return
        # disable run button & start progress
        self.run_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Running bisection...")
        # run backend in background thread
        thread = threading.Thread(target=self._run_bisection_thread, args=(func,a,b,d), daemon=True)
        thread.start()

    def _run_bisection_thread(self, func: str, a: float, b: float, d: int):
        try:
            res = bisection(func, a, b, d)
        except Exception as exc:
            res = {"error": f"Unhandled exception in backend: {exc}\n{traceback.format_exc()}"}
        # schedule UI update on main thread
        self.root.after(50, lambda: self._on_bisection_done(res))

    def _on_bisection_done(self, res: Dict[str, Any]):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        if "error" in res:
            self.status_var.set("Error")
            message = res.get("error", "Unknown error")
            # if workability info exists, include
            work = res.get("workability")
            if work:
                message += f"\n\nDetails: {work}"
            messagebox.showerror("Bisection Error", message)
            return

        # fill summary
        self._last_iterations = res.get("iterations", [])
        root = res.get("root")
        N = res.get("N_required")
        tol = res.get("tol")
        conv_by_tol = res.get("converged_by_tol", False)
        workability = res.get("workability")
        conv_text = res.get("convergence_text", "")
        rate_text = res.get("rate_text", "")
        summary_lines = [
            f"Root approx: {root}",
            f"N (required): {N}",
            f"Tolerance (tol): {tol}",
            f"Converged by tol during N iterations: {conv_by_tol}",
            f"Workability: {workability}",
            "",
            conv_text,
            "",
            rate_text
        ]
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(str(x) for x in summary_lines if x is not None))
        self.summary_text.configure(state="disabled")
        # ensure scrollbar is at top
        self.summary_text.yview_moveto(0.0)

        # populate treeview
        for r in self.tree.get_children():
            self.tree.delete(r)
        for row in self._last_iterations:
            # format numbers reasonably
            vals = (
                row.get("n"),
                utils.pretty_format_number(row.get("a", ""), digits=8),
                utils.pretty_format_number(row.get("b", ""), digits=8),
                utils.pretty_format_number(row.get("p", ""), digits=8),
                utils.pretty_format_number(row.get("f(p)", ""), digits=8),
                utils.pretty_format_number(row.get("error", "") if row.get("error") is not None else "", digits=8)
            )
            self.tree.insert("", "end", values=vals)

        # enable export when iterations exist
        if self._last_iterations:
            self.export_btn.configure(state="normal")
            self.status_var.set(f"Done — produced {len(self._last_iterations)} iterations.")
        else:
            self.status_var.set("Done — no iterations produced.")

def run_app():
    root = tk.Tk()
    app = BisectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()
