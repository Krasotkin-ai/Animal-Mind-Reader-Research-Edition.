import argparse
from src.nwb_loader import load_nwb, get_trials, get_unit_spike_times

ap = argparse.ArgumentParser()
ap.add_argument("--nwb", required=True)
args = ap.parse_args()

io, nwb = load_nwb(args.nwb)
try:
    print("=== NWB ===")
    print("identifier:", nwb.identifier)
    print("session:", nwb.session_description)
    print("\n=== trials columns ===")
    trials = get_trials(nwb)
    print(list(trials.columns))
    print("\n=== sample trials ===")
    print(trials.head(10).to_string())
    print("\n=== units ===")
    units = get_unit_spike_times(nwb)
    print("number of sorted units:", len(units))
    print("\n=== candidate labels ===")
    for c in ["stimulus", "shape", "object", "condition", "trial_type", "outcome"]:
        if c in trials.columns:
            print(c, "unique:", trials[c].dropna().unique()[:20])
finally:
    io.close()
