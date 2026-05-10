"""main.py — Main entry point. Settings GUI, pipeline runner, and query interface."""
import sys
import os
import re
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

CONFIG_FILE   = os.path.join(os.path.dirname(__file__), "config.py")
TOTAL_PAPERS  = 288_368   # total unique papers in the dataset (used for % conversion)

# Field group keys: which fields belong in which sub-tab
GROUP_TRAINING = {
    "TRAIN_SIZE", "VOCAB_SIZE", "SEED", "ALPHA", "LR_RATE",
    "N_ITER", "LAMBDA", "MONTHS_PER_PAPER",
}
GROUP_COST = {
    "COST_FRACTION", "RESOURCE_COST", "SALARY_MULTIPLIER",
    "SALARY_PROFESSOR",     "DIST_PROFESSOR",
    "SALARY_GRAD",          "DIST_GRAD",
    "SALARY_MASTERS",       "DIST_MASTERS",
    "SALARY_OTHER_ACADEMIC","DIST_OTHER",
    "SALARY_UNDERGRAD",     "DIST_UNDERGRAD",
}
GROUP_PATHS = {"DATA_PATH", "INPUT_DIR", "OUTPUT_DIR"}

# Field definitions: (config_key, label, type, description)
FIELDS = [
    ("TRAIN_SIZE",               "Train Size",               int,   "% of dataset for NLP training — 100% = all 288K papers"),
    ("VOCAB_SIZE",               "Max Features (TF-IDF)",    int,   "Top-N most frequent tokens. Paper uses 10,000 (covers 94.7% of occurrences). Full corpus has 266,760 unique tokens — increase to include more rare terms."),
    ("SEED",                     "Random Seed",              int,   "Reproducibility seed"),
    ("ALPHA",                    "NB Alpha (smoothing)",     float, "Laplace smoothing for Naive Bayes"),
    ("LR_RATE",                  "LR Learning Rate",         float, "Logistic Regression step size"),
    ("N_ITER",                   "LR Iterations",            int,   "Logistic Regression training iterations"),
    ("LAMBDA",                   "LR Lambda (L2)",           float, "L2 regularization strength"),
    ("COST_FRACTION",            "Cost Fraction",            float, "Fraction of annual salary per paper (0.25 = 25%)"),
    ("RESOURCE_COST",             "Resource Cost ($)",        int,   "Flat compute + equipment cost per paper. $0 = theory only, $500 = median ArXiv experiment, $5000 = large training run, $50000+ = foundation model"),
    ("SALARY_MULTIPLIER",        "Global Scaler",            float, "Scales ALL salaries proportionally. 1.0 = published medians, 0.7 = 30% lower, 1.3 = 30% higher. Override individual rows below for fine control."),
    # Salary tiers + their share of the author population (co-dependent — must sum to 1.0)
    ("SALARY_PROFESSOR",         "Professor — $",            int,   "PI / advisor — median salary"),
    ("DIST_PROFESSOR",           "Professor — share",        float, "Fraction who are professors (default 0.15)"),
    ("SALARY_GRAD",              "Graduate Researcher — $",  int,   "Combined PhD + postdoc median — average of both"),
    ("DIST_GRAD",                "Graduate Researcher — share", float, "Fraction graduate researchers/postdocs (default 0.70)"),
    ("SALARY_MASTERS",           "Master Student — $",     int,   "Master student — unknown, set below PhD level; adjust as needed"),
    ("DIST_MASTERS",             "Master Student — share", float, "Fraction master students (default 0.05)"),
    ("SALARY_OTHER_ACADEMIC",    "Other/Unknown — $",        int,   "Visiting researchers, RAs, unknown role — also used as default fallback"),
    ("DIST_OTHER",               "Other/Unknown — share",    float, "Fraction other/unknown academic (default 0.05) — all five shares must sum to 1.0"),
    ("SALARY_UNDERGRAD",         "Undergrad — $",            int,   "Typically unpaid / course credit ($0)"),
    ("DIST_UNDERGRAD",           "Undergrad — share",        float, "Fraction undergrads (default 0.05) — all five shares must sum to 1.0"),
    ("MONTHS_PER_PAPER",         "Months per Paper",         int,   "Assumed production time per paper"),
    ("DATA_PATH",                "Data Path",                str,   "Folder containing the JSONL files"),
    ("INPUT_DIR",                "Input Directory",          str,   "Folder for intermediate data (roster, affiliations, pickles, salary sources)"),
    ("OUTPUT_DIR",               "Output Directory",         str,   "Folder for final deliverables (figures, model_results.csv)"),
]

SCRIPTS = [
    ("01_cost.py",  "1. Cost Computation"),
    ("02_nlp.py", "2. NLP Pipeline"),
    ("03_analysis.py",     "3. Analysis & Figures"),
]


def read_config():
    """Parse config.py and return a dict of {key: raw_string_value}."""
    values = {}
    with open(CONFIG_FILE, encoding="utf-8") as f:
        text = f.read()
    for key, _, typ, _ in FIELDS:
        m = re.search(rf'^{key}\s*=\s*(.+)$', text, re.MULTILINE)
        if m:
            raw = re.sub(r'\s*#.*$', '', m.group(1)).strip()
            if typ is str:
                raw = raw.strip('"').strip("'")
            else:
                raw = raw.replace("_", "")  # handle 15_000 → 15000
            values[key] = raw
    return values


def write_config(new_values):
    """Write updated values back into config.py."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        text = f.read()

    for key, _, typ, _ in FIELDS:
        if key not in new_values:
            continue
        val = new_values[key]
        # Format the replacement value
        if typ is str:
            formatted = f'"{val}"'
        elif typ is int:
            v = int(val)
            # restore underscore formatting for large numbers
            formatted = f"{v:_}" if v >= 1000 else str(v)
        else:
            formatted = str(float(val))

        text = re.sub(
            rf'^({key}\s*=\s*)(.+?)(\s*(?:#.*)?$)',
            lambda m, fmt=formatted: m.group(1) + fmt + (m.group(3) if m.group(3).strip().startswith('#') else ''),
            text,
            flags=re.MULTILINE,
        )

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(text)


import threading
import queue as _queue
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_active_proc   = [None]
_all_run_btns  = []          # all script buttons — disabled while one runs
_MAX_QUEUE     = 5_000       # drop lines beyond this to prevent memory blowup


def _kill_tree(proc):
    """Kill process and all its children (worker threads, etc.)."""
    if _HAS_PSUTIL:
        try:
            parent = _psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try: child.kill()
                except _psutil.NoSuchProcess: pass
            parent.kill()
        except _psutil.NoSuchProcess:
            pass
    else:
        proc.kill()          # fallback: kill only the parent


def run_script(script_name, log_widget, btn, original_label):
    """
    Run a script in a background thread so the UI stays responsive.
    Edge cases handled:
      - Full process tree killed on cancel (psutil)
      - All other script buttons disabled while running (no double-launch)
      - Double-click guard on the active button
      - Queue capped to prevent memory blowup on verbose output
    """
    # Guard: don't start if already running
    if _active_proc[0] is not None:
        return

    log_widget.config(state="normal")
    log_widget.delete("1.0", tk.END)
    log_widget.insert(tk.END, f"Running {script_name}...\n\n")
    log_widget.config(state="disabled")

    cancelled_flag = [False]
    line_queue     = _queue.Queue(maxsize=_MAX_QUEUE)

    def _cancel():
        if _active_proc[0]:
            cancelled_flag[0] = True
            _kill_tree(_active_proc[0])

    # Active button → Cancel; all other buttons → disabled
    for b, lbl in _all_run_btns:
        if b is btn:
            b.config(text="⛔  Cancel", command=_cancel)
        else:
            b.config(state="disabled")

    def restore():
        _active_proc[0] = None
        cancelled_flag[0] = False
        for b, lbl in _all_run_btns:
            b.config(state="normal", text=lbl,
                     command=lambda s=script_name, bw=b, lb=lbl:
                         run_script(s, log_widget, bw, lb))
        # Restore correct commands for each button
        for b, lbl in _all_run_btns:
            script = next((s for s, l in SCRIPTS if l == lbl), None)
            if script:
                b.config(command=lambda s=script, bw=b, lb=lbl:
                             run_script(s, log_widget, bw, lb))

    def _worker():
        proc = subprocess.Popen(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        _active_proc[0] = proc
        dropped = 0
        for line in proc.stdout:
            if not line_queue.full():
                line_queue.put(line)
            else:
                dropped += 1
        proc.wait()
        if dropped:
            line_queue.put(f"\n[{dropped:,} lines dropped — output too large]\n")
        line_queue.put(("__DONE__", proc.returncode))

    def _poll():
        try:
            while True:
                item = line_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    rc = item[1]
                    status = ("⛔ Cancelled" if cancelled_flag[0]
                              else "✓ Done"  if rc == 0
                              else f"✗ Failed (exit {rc})")
                    log_widget.config(state="normal")
                    log_widget.insert(tk.END, f"\n{status}\n")
                    log_widget.config(state="disabled")
                    restore()
                    return
                else:
                    log_widget.config(state="normal")
                    log_widget.insert(tk.END, item)
                    log_widget.see(tk.END)
                    log_widget.config(state="disabled")
        except _queue.Empty:
            pass
        log_widget.after(50, _poll)

    threading.Thread(target=_worker, daemon=True).start()
    log_widget.after(50, _poll)


def build_ui():
    root = tk.Tk()
    root.title("ArXiv Research Cost — Settings")
    root.resizable(True, True)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    current = read_config()
    entries = {}

    # ── Sub-tab helper ────────────────────────────────────────────────────────
    def build_field_tab(parent, fields_subset, label):
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text=f"  {label}  ")

        ttk.Label(frame, text="Setting", font=("", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(frame, text="Value", font=("", 10, "bold")).grid(
            row=0, column=1, sticky="w", pady=(0, 6))
        ttk.Label(frame, text="Description", font=("", 10, "bold")).grid(
            row=0, column=2, sticky="w", pady=(0, 6), padx=(12, 0))
        ttk.Separator(frame, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        return frame

    # Build the three sub-tabs
    training_frame = build_field_tab(notebook,
        [f for f in FIELDS if f[0] in GROUP_TRAINING], "Training & NLP")
    cost_frame     = build_field_tab(notebook,
        [f for f in FIELDS if f[0] in GROUP_COST], "Cost & Salary")
    paths_frame    = build_field_tab(notebook,
        [f for f in FIELDS if f[0] in GROUP_PATHS], "Paths")

    # Map each field to its frame
    field_to_frame = {}
    for key, *_ in FIELDS:
        if key in GROUP_TRAINING:   field_to_frame[key] = training_frame
        elif key in GROUP_COST:     field_to_frame[key] = cost_frame
        elif key in GROUP_PATHS:    field_to_frame[key] = paths_frame

    # Track row counter per frame
    frame_rows = {training_frame: 2, cost_frame: 2, paths_frame: 2}

    # Legacy single-frame variable for save/reset buttons (use cost_frame as primary)
    settings_frame = cost_frame

    for key, label, typ, desc in FIELDS:
        settings_frame = field_to_frame.get(key, cost_frame)
        row_idx        = frame_rows[settings_frame]
        frame_rows[settings_frame] += 1
        ttk.Label(settings_frame, text=label).grid(
            row=row_idx, column=0, sticky="w", pady=3)

        if key == "TRAIN_SIZE":
            # Percentage only — 100% = all papers (TRAIN_SIZE=0 in backend)
            raw_num = re.match(r'^\d+', current.get(key, "0"))
            n0      = int(raw_num.group()) if raw_num else 0
            pct0    = 100.0 if n0 == 0 else round(n0 / TOTAL_PAPERS * 100, 1)

            pct_var = tk.StringVar(value=str(pct0))

            cell = ttk.Frame(settings_frame)
            cell.grid(row=row_idx, column=1, sticky="w", padx=(8, 0), pady=3)
            ttk.Entry(cell, textvariable=pct_var, width=7).pack(side="left")
            ttk.Label(cell, text="%  (100% = all papers)", foreground="#888").pack(side="left")

            entries[key] = pct_var  # converted to int on save

        elif key == "COST_FRACTION":
            # Show as percentage: 0.25 ↔ 25%
            try:
                frac0 = float(current.get(key, "0.25"))
                pct0  = round(frac0 * 100, 1)
            except ValueError:
                frac0, pct0 = 0.25, 25.0

            pct_var  = tk.StringVar(value=str(pct0))
            frac_var = tk.StringVar(value=str(frac0))

            cell = ttk.Frame(settings_frame)
            cell.grid(row=row_idx, column=1, sticky="w", padx=(8, 0), pady=3)
            ttk.Entry(cell, textvariable=pct_var,  width=6).pack(side="left")
            ttk.Label(cell, text="% = ", foreground="#888").pack(side="left")
            ttk.Label(cell, textvariable=frac_var, foreground="#555", width=5).pack(side="left")

            flag = [False]

            def _sync_frac(*, pv=pct_var, fv=frac_var, f=flag):
                if f[0]: return
                f[0] = True
                try:
                    fv.set(str(round(float(pv.get()) / 100, 4)))
                except ValueError:
                    pass
                f[0] = False

            pct_var.trace_add("write", lambda *a, **kw: _sync_frac())
            entries[key] = pct_var  # save as pct, convert on write

        else:
            var = tk.StringVar(value=current.get(key, ""))
            width = 32 if typ is str else 12
            entry = ttk.Entry(settings_frame, textvariable=var, width=width)
            entry.grid(row=row_idx, column=1, sticky="w", padx=(8, 0), pady=3)
            entries[key] = var

        ttk.Label(settings_frame, text=desc, foreground="#666").grid(
            row=row_idx, column=2, sticky="w", padx=(12, 0), pady=3)

    # Buttons placed in the root, below the notebook — visible from all tabs

    def save():
        new_vals = {}
        for key, _, typ, _ in FIELDS:
            raw = entries[key].get().strip()
            if key == "TRAIN_SIZE":
                # 100% → 0 (all); otherwise convert % → paper count
                try:
                    p = float(raw)
                    raw = "0" if p >= 100 else str(int(p / 100 * TOTAL_PAPERS))
                except ValueError:
                    raw = "0"
            # COST_FRACTION stored as % in UI, convert back to decimal
            elif key == "COST_FRACTION":
                try:
                    raw = str(round(float(raw) / 100, 4))
                except ValueError:
                    pass
            try:
                if typ is int:
                    int(raw)
                elif typ is float:
                    float(raw)
            except ValueError:
                messagebox.showerror("Invalid value", f"{key}: '{raw}' is not a valid {typ.__name__}")
                return
            new_vals[key] = raw
        write_config(new_vals)
        # Refresh all entries from disk so normalized values show up immediately
        refresh_all()
        messagebox.showinfo("Saved", "config.py updated successfully.")

    # ── Live scaling: scale from TRUE MEDIANS (snapshot at startup), not from
    #    current entry values — prevents compounding/double-scaling bugs.
    SCALABLE_KEYS = [
        "SALARY_PROFESSOR", "SALARY_GRAD", "SALARY_MASTERS",
        "SALARY_OTHER_ACADEMIC", "SALARY_UNDERGRAD",
    ]
    # Snapshot the median values at startup — these are the "1.0" baseline
    salary_baseline = {}
    for k in SCALABLE_KEYS:
        if k in entries:
            try:
                salary_baseline[k] = int(entries[k].get())
            except ValueError:
                salary_baseline[k] = 0
    sync_flag = [False]

    def on_mult_change(*_):
        if sync_flag[0]:
            return
        try:
            mult = float(entries["SALARY_MULTIPLIER"].get())
        except (ValueError, KeyError):
            return
        sync_flag[0] = True
        try:
            for k, base in salary_baseline.items():
                if k in entries:
                    entries[k].set(str(int(round(base * mult))))
        finally:
            sync_flag[0] = False

    if "SALARY_MULTIPLIER" in entries:
        entries["SALARY_MULTIPLIER"].trace_add("write", on_mult_change)

    def refresh_all():
        """Re-read config and update all entry boxes with proper display formatting."""
        saved = read_config()
        for key in entries:
            raw = saved.get(key, "")
            try:
                # Apply same display conversions as initial load
                if key == "COST_FRACTION":
                    entries[key].set(str(round(float(raw) * 100, 1)))   # 0.25 → 25.0
                elif key == "TRAIN_SIZE":
                    n = int(re.match(r'^\d+', raw).group()) if re.match(r'^\d+', raw) else 0
                    entries[key].set("100.0" if n == 0 else
                                      str(round(n / TOTAL_PAPERS * 100, 1)))
                else:
                    entries[key].set(raw)
            except Exception:
                pass

    def reset():
        saved = read_config()
        for key, var in entries.items():
            entries[key].set(saved.get(key, ""))

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Button(btn_frame, text="Save to config.py", command=save).pack(side="left", padx=(0, 8))
    ttk.Button(btn_frame, text="Reset", command=reset).pack(side="left")

    # ── Tab 2: Run Scripts ─────────────────────────────────────────────────────
    run_frame = ttk.Frame(notebook, padding=10)
    notebook.add(run_frame, text="  Run Pipeline  ")

    ttk.Label(run_frame, text="Run scripts in order. Each must complete before the next.",
              foreground="#555").pack(anchor="w", pady=(0, 8))

    log = tk.Text(run_frame, height=22, width=80, state="disabled",
                  font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                  insertbackground="white", relief="flat")
    log.pack(fill="both", expand=True, pady=(0, 8))

    scroll = ttk.Scrollbar(run_frame, command=log.yview)
    log.configure(yscrollcommand=scroll.set)

    btn_row = ttk.Frame(run_frame)
    btn_row.pack(fill="x")

    _all_run_btns.clear()
    for script, label in SCRIPTS:
        btn = ttk.Button(btn_row, text=label)
        btn.configure(command=lambda s=script, b=btn, lbl=label: run_script(s, log, b, lbl))
        btn.pack(side="left", padx=(0, 6))
        _all_run_btns.append((btn, label))

    # ── Tab 3: Query ───────────────────────────────────────────────────────────
    query_frame = ttk.Frame(notebook, padding=10)
    notebook.add(query_frame, text="  Query  ")

    SEARCH_MODES = [
        "Paper Title",
        "Author Name",
        "Category  (e.g. cs.CV)",
        "Year  (e.g. 2024)",
        "Cost Bucket  (Low / Medium / High)",
        "Predict from Abstract",
    ]

    # ── Single search row ────────────────────────────────────────────────────
    top = ttk.Frame(query_frame)
    top.pack(fill="x", pady=(0, 6))

    mode_var  = tk.StringVar(value=SEARCH_MODES[0])
    query_var = tk.StringVar()

    mode_cb = ttk.Combobox(top, textvariable=mode_var, values=SEARCH_MODES,
                            state="readonly", width=28)
    mode_cb.pack(side="left", padx=(0, 6))

    CONSTRAINED = {
        "Category  (e.g. cs.CV)":          ["cs.AI", "cs.LG", "cs.CL", "cs.NE", "cs.IR", "cs.CV"],
        "Year  (e.g. 2024)":               ["2022", "2023", "2024", "2025", "2026"],
        "Cost Bucket  (Low / Medium / High)": ["Low", "Medium", "High"],
    }

    q_entry  = ttk.Entry(top, textvariable=query_var, width=48)
    q_picker = ttk.Combobox(top, textvariable=query_var, state="readonly", width=46)
    q_entry.pack(side="left", padx=(0, 6))

    result_var = tk.StringVar(value="Select a mode and enter a query, then press Search.")
    ttk.Button(top, text="Search", command=lambda: _run_query()).pack(side="left")

    # ── Multi-line input (shown only for Predict from Abstract) ──────────────
    abstract_frame = ttk.Frame(query_frame)
    PLACEHOLDER = "Paste title and abstract text here..."
    abstract_box  = tk.Text(abstract_frame, height=6, width=80, wrap="word",
                            font=("Consolas", 9), foreground="#888")
    abstract_box.insert("1.0", PLACEHOLDER)
    abstract_box.pack(fill="x")

    def _on_focus_in(e):
        if abstract_box.get("1.0","end").strip() == PLACEHOLDER:
            abstract_box.delete("1.0","end")
            abstract_box.config(foreground="#000")
    def _on_focus_out(e):
        if not abstract_box.get("1.0","end").strip():
            abstract_box.insert("1.0", PLACEHOLDER)
            abstract_box.config(foreground="#888")
    abstract_box.bind("<FocusIn>",  _on_focus_in)
    abstract_box.bind("<FocusOut>", _on_focus_out)

    def _on_mode_change(*_):
        m = mode_var.get()
        query_var.set("")
        if m.startswith("Predict"):
            q_entry.pack_forget(); q_picker.pack_forget()
            abstract_frame.pack(fill="x", pady=(0,6))
        elif m in CONSTRAINED:
            abstract_frame.pack_forget(); q_entry.pack_forget()
            q_picker["values"] = CONSTRAINED[m]
            q_picker.pack(side="left", padx=(0,6), after=mode_cb)
        else:
            abstract_frame.pack_forget(); q_picker.pack_forget()
            q_entry.pack(side="left", padx=(0,6), after=mode_cb)
    mode_cb.bind("<<ComboboxSelected>>", _on_mode_change)

    ttk.Separator(query_frame, orient="horizontal").pack(fill="x", pady=(4,6))

    result_lbl = ttk.Label(query_frame, textvariable=result_var,
                           foreground="#0055aa", wraplength=720,
                           justify="left", font=("Consolas", 9))
    result_lbl.pack(anchor="w")

    # ── Bind Enter key ────────────────────────────────────────────────────────
    q_entry.bind("<Return>",      lambda e: _run_query())
    abstract_box.bind("<Control-Return>", lambda e: _run_query())

    _df_cache = [None]   # cache df in memory after first load — searches become instant

    def _run_query():
        import importlib, importlib.util
        _spec = importlib.util.spec_from_file_location("config", CONFIG_FILE)
        _cfg  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)
        mode  = mode_var.get()
        q     = query_var.get().strip()
        base  = os.path.dirname(CONFIG_FILE)
        inp   = os.path.join(base, _cfg.INPUT_DIR.rstrip("/\\"))
        sample_path = os.path.join(inp, "sample_df_with_costs.pkl")

        # ── Predict from Abstract ─────────────────────────────────────────────
        if mode.startswith("Predict"):
            abstract = abstract_box.get("1.0","end").strip()
            if not abstract or abstract == PLACEHOLDER:
                result_var.set("Paste title and abstract text into the text box first."); return
            vocab_path   = os.path.join(inp, "vocab.csv")
            idf_path     = os.path.join(inp, "idf.npy")
            weights_path = os.path.join(inp, "lr_weights.npy")
            if not all(os.path.exists(p) for p in [vocab_path, idf_path, weights_path]):
                result_var.set("Model files not found — run scripts 01-04 first."); return
            try:
                import numpy as np, pandas as pd, re as _re
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                vocab_df = pd.read_csv(vocab_path)
                vocab    = dict(zip(vocab_df["word"], vocab_df["idx"]))
                idf      = np.load(idf_path)
                stops    = set(ENGLISH_STOP_WORDS) | {"model","paper","propose","method",
                           "approach","show","result","experiment","dataset","performance",
                           "based","learning","neural","network","train"}
                tokens = [t for t in _re.sub(r"[^a-z0-9]+"," ",abstract.lower()).split()
                          if len(t)>2 and t not in stops]
                V  = len(vocab); tf = np.zeros(V, dtype=np.float32)
                for t in tokens:
                    if t in vocab: tf[vocab[t]] += 1
                s = tf.sum();
                if s>0: tf/=s
                vec = tf*idf; n=np.linalg.norm(vec)
                if n>0: vec/=n
                X = vec.reshape(1,-1)
                _lo = int(np.load(os.path.join(inp,"low_thresh.npy"))[0])
                _hi = int(np.load(os.path.join(inp,"high_thresh.npy"))[0])
                LABELS = {0:f"Low  (<${_lo:,})",1:f"Medium  (${_lo:,}–${_hi:,})",2:f"High  (>${_hi:,})"}
                out = [f"Abstract: {abstract[:90]}{'...' if len(abstract)>90 else ''}", ""]
                nb_ll = os.path.join(inp,"nb_log_likelihood.npy")
                nb_lp = os.path.join(inp,"nb_log_prior.npy")
                if os.path.exists(nb_ll):
                    ll=np.load(nb_ll); lp=np.load(nb_lp)
                    out.append(f"  Naive Bayes          → {LABELS[int(np.argmax(X@ll.T+lp))]}")
                W = np.load(weights_path)
                out.append(f"  Logistic Regression  → {LABELS[int(np.argmax(X.astype(np.float64)@W.T))]}")
                mlp_p = os.path.join(inp,"mlp_state.pt")
                if os.path.exists(mlp_p):
                    try:
                        import torch, torch.nn as nn
                        class MLP(nn.Module):
                            def __init__(self,d):
                                super().__init__()
                                self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(0.3),
                                    nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),
                                    nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
                            def forward(self,x): return self.net(x)
                        m=MLP(V); m.load_state_dict(torch.load(mlp_p,map_location="cpu")); m.eval()
                        with torch.no_grad():
                            p=int(m(torch.tensor(X,dtype=torch.float32)).argmax(1).item())
                        out.append(f"  MLP                  → {LABELS[p]}")
                    except Exception: pass
                result_var.set("\n".join(out))
            except Exception as e:
                result_var.set(f"Prediction error: {e}")
            return

        # ── Dataset search modes ──────────────────────────────────────────────
        if not q:
            result_var.set("Enter a search term."); return
        if not os.path.exists(sample_path):
            result_var.set("Pipeline outputs not found — run scripts 01-03 first."); return
        try:
            import pandas as pd
            if _df_cache[0] is None:
                result_var.set("Loading dataset into memory...")
                root.update()
                _df_cache[0] = pd.read_pickle(sample_path)
            df = _df_cache[0]

            if mode.startswith("Paper Title"):
                mask = df["title"].str.contains(q, case=False, na=False)
                col  = "title"
            elif mode.startswith("Author"):
                mask = df["authors"].apply(lambda a: any(q.lower() in x.lower() for x in a))
                col  = "title"
            elif mode.startswith("Category"):
                mask = df["primary_category"].str.lower() == q.lower()
                col  = "title"
            elif mode.startswith("Year"):
                mask = df["year"].astype(str) == q
                col  = "title"
            elif mode.startswith("Cost Bucket"):
                mask = df["cost_bucket"].str.lower() == q.strip().capitalize().lower()
                col  = "title"
            else:
                result_var.set("Unknown mode."); return

            hits = df[mask]
            if hits.empty:
                result_var.set(f"No results for '{q}' in {mode.split('(')[0].strip()}."); return

            lines = [f"{len(hits):,} result(s) for '{q}':", ""]
            for _, r in hits.head(8).iterrows():
                lines.append(
                    f"• [{r['cost_bucket']}] ${r['total_cost']:,.0f}  "
                    f"{r['author_count']} authors  {r['primary_category']}  {r['year']}\n"
                    f"  {str(r[col])[:110]}")
            if len(hits) > 8:
                lines.append(f"\n  ...and {len(hits)-8:,} more")
            result_var.set("\n".join(lines))
        except Exception as e:
            result_var.set(f"Error: {e}")

    root.mainloop()


if __name__ == "__main__":
    build_ui()
