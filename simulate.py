"""Step 1: generate SIMULATED bioreactor data.

Nothing here is real plant data. We invent 20 batches with realistic-looking
sensor values, then inject 3 known faults so we can later check whether the
monitor finds them (we know the "right answer", so we can score it).
"""
import numpy as np
import pandas as pd

N_BATCHES = 20
HOURS = 14 * 24      # 14-day batch, one reading per hour = 336 readings
N_REFERENCE = 10     # B01-B10 = known-good historical batches used to set limits

# Fault plan: batch -> (fault type, start hour). All faults are in the "new" batches B11-B20.
FAULTS = {
    "B12": ("temp_drift", 150),   # temperature slowly creeps up (e.g. jacket temperature controller issue)
    "B15": ("ph_spike", 200),     # pH jumps for a few hours (e.g. base pump over-dosing)
    "B18": ("do_decline", 100),   # dissolved oxygen slowly falls (e.g. sparger fouling)
}
# Which sensor each fault shows up in (used only when scoring the alarms)
FAULT_SENSOR = {"temp_drift": "temp_c", "ph_spike": "ph", "do_decline": "do_pct"}
PH_SPIKE_HOURS = 6


def simulate_batch(batch_id, rng, fault="none", start=0):
    """Return one batch as a DataFrame with one row per hour."""
    hours = np.arange(HOURS)
    # Agitation ramps from 100 to 140 rpm as the cells grow and need more mixing.
    agitation = 100 + 40 * hours / HOURS + rng.normal(0, 2, HOURS)
    # DO depends on agitation: faster stirring dissolves more oxygen.
    # This link between two sensors is what PCA will learn as "normal".
    do = 40 + 0.25 * (agitation - 120) + rng.normal(0, 1.0, HOURS)
    ph = rng.normal(7.0, 0.03, HOURS)      # pH held near 7.0 by its controller
    temp = rng.normal(37.0, 0.1, HOURS)    # temperature held near 37 C

    since = np.maximum(hours - start, 0)   # hours since the fault started (0 before it)
    if fault == "temp_drift":
        temp += 0.01 * since                          # +0.24 C per day
    elif fault == "ph_spike":
        ph[start:start + PH_SPIKE_HOURS] += 0.3       # sudden jump, then back to normal
    elif fault == "do_decline":
        do -= 0.04 * since                            # ~1% DO lost per day; agitation looks normal

    return pd.DataFrame({"batch_id": batch_id, "hour": hours, "ph": ph, "temp_c": temp,
                         "do_pct": do, "agitation_rpm": agitation})


def simulate_all(seed=42):
    """Simulate every batch. Returns (batches table, sensor_readings table)."""
    rng = np.random.default_rng(seed)   # fixed seed -> identical data every run (reproducible)
    batch_rows, readings = [], []
    for i in range(1, N_BATCHES + 1):
        batch_id = f"B{i:02d}"                                  # B01, B02, ... B20
        fault, start = FAULTS.get(batch_id, ("none", None))
        readings.append(simulate_batch(batch_id, rng, fault, start or 0))

        end = None                                              # last hour the fault is present
        if fault == "ph_spike":
            end = start + PH_SPIKE_HOURS - 1
        elif fault != "none":
            end = HOURS - 1                                     # drifts last until the batch ends
        batch_rows.append({"batch_id": batch_id, "is_reference": int(i <= N_REFERENCE),
                           "injected_fault": fault, "fault_start_hour": start,
                           "fault_end_hour": end, "data_source": "SIMULATED"})
    return pd.DataFrame(batch_rows), pd.concat(readings, ignore_index=True)
