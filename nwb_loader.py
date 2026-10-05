from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

def load_nwb(path: str):
    io = NWBHDF5IO(str(path), mode="r", load_namespaces=True)
    return io, io.read()

def find_spike_units(nwb):
    if nwb.units is None:
        raise ValueError("This NWB file has no units table.")
    return nwb.units

def get_unit_spike_times(nwb):
    units = find_spike_units(nwb)
    out = []
    for i in range(len(units.id[:])):
        st = np.asarray(units["spike_times"][i], dtype=float)
        out.append(st)
    return out

def get_trials(nwb):
    if nwb.trials is None:
        raise ValueError("This NWB file has no trials table.")
    return nwb.trials.to_dataframe()
