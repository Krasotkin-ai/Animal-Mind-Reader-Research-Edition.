"""
Convert an NWB session into a simple trial-level neural classification dataset.

The script tries to infer a categorical trial label from common trial columns.
For DANDI 000231, inspect the columns first if automatic detection fails.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .nwb_loader import load_nwb, get_unit_spike_times, get_trials

LABEL_CANDIDATES = [
    "shape", "object", "stimulus", "stimulus_name",
    "shape_name", "condition", "trial_type"
]

def choose_label_column(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            vals = df[c].dropna()
            if len(vals) and vals.nunique() >= 2:
                return c
    return None

def choose_time_columns(df):
    starts = [c for c in ["start_time", "start", "trial_start"] if c in df.columns]
    stops = [c for c in ["stop_time", "stop", "trial_stop"] if c in df.columns]
    if not starts or not stops:
        raise ValueError(
            f"Could not identify trial start/stop columns. Columns: {list(df.columns)}"
        )
    return starts[0], stops[0]

def bin_spikes(spike_times, start, stop, bin_size=0.02):
    n_bins = int(np.ceil((stop - start) / bin_size))
    X = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    edges = start + np.arange(n_bins + 1) * bin_size
    for u, st in enumerate(spike_times):
        x = st[(st >= start) & (st < stop)]
        if len(x):
            idx = np.searchsorted(edges, x, side="right") - 1
            idx = idx[(idx >= 0) & (idx < n_bins)]
            np.add.at(X[u], idx, 1)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pre", type=float, default=0.5)
    ap.add_argument("--post", type=float, default=1.5)
    ap.add_argument("--bin-size", type=float, default=0.02)
    args = ap.parse_args()

    io, nwb = load_nwb(args.nwb)
    try:
        trials = get_trials(nwb)
        label_col = choose_label_column(trials)
        if label_col is None:
            raise ValueError(
                "No automatic label column found. Run an inspection script and "
                "adapt LABEL_CANDIDATES to the exact NWB trial columns."
            )

        start_col, stop_col = choose_time_columns(trials)
        spike_times = get_unit_spike_times(nwb)

        X, y = [], []
        for _, row in trials.iterrows():
            label = row[label_col]
            if label is None or (isinstance(label, float) and np.isnan(label)):
                continue
            start = float(row[start_col]) - args.pre
            stop = float(row[stop_col]) + args.post
            if stop <= start:
                continue
            counts = bin_spikes(spike_times, start, stop, args.bin_size)
            # Convert to a time x neurons matrix.
            X.append(counts.T)
            y.append(str(label))

        if not X:
            raise RuntimeError("No usable trials were produced.")

        # Variable-length trials are padded/truncated to a common length.
        max_t = max(x.shape[0] for x in X)
        n_units = X[0].shape[1]
        X_pad = np.zeros((len(X), max_t, n_units), dtype=np.float32)
        mask = np.zeros((len(X), max_t), dtype=bool)
        for i, x in enumerate(X):
            t = min(max_t, x.shape[0])
            X_pad[i, :t] = x[:t]
            mask[i, :t] = True

        labels, y_idx = np.unique(np.asarray(y), return_inverse=True)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            X=X_pad,
            mask=mask,
            y=y_idx.astype(np.int64),
            labels=labels.astype(str),
            label_column=label_col,
            bin_size=np.array(args.bin_size),
        )

        print(f"Saved {len(y)} trials -> {out}")
        print(f"Shape: {X_pad.shape}")
        print(f"Classes: {labels.tolist()}")
        print(f"Label column: {label_col}")
    finally:
        io.close()

if __name__ == "__main__":
    main()
