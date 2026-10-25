"""Step 6: score the alarms against the faults we injected (the ground truth)."""
from simulate import FAULT_SENSOR

METHODS = ["SPC_3sigma", "PCA_T2"]


def is_true_alarm(alert, batches):
    """True if the alarm points at a real injected fault: right batch, during the fault, right sensor.
    A PCA alarm covers all sensors at once, so any PCA alarm during the fault counts."""
    batch = batches.loc[alert["batch"]]
    if batch.injected_fault == "none":
        return False
    during_fault = batch.fault_start_hour <= alert["hour"] <= batch.fault_end_hour
    right_sensor = alert["method"] == "PCA_T2" or alert["sensor"] == FAULT_SENSOR[batch.injected_fault]
    return during_fault and right_sensor


def evaluate(alerts, batches, monitored_ids):
    """Summarise how well the alarms matched the injected faults."""
    faulty = [b for b in monitored_ids if batches.loc[b, "injected_fault"] != "none"]
    true_alerts = [a for a in alerts if is_true_alarm(a, batches)]
    caught = {a["batch"] for a in true_alerts}

    # First hour each method raised a true alarm, per faulty batch (None = never caught it)
    first_alarm = {}
    for b in faulty:
        first_alarm[b] = {}
        for m in METHODS:
            hours = [a["hour"] for a in true_alerts if a["batch"] == b and a["method"] == m]
            first_alarm[b][m] = min(hours) if hours else None

    n_true = len(true_alerts)
    return {
        "faults_injected": len(faulty),
        "faults_caught": len(caught),
        "true_alarms": n_true,
        "false_alarms": len(alerts) - n_true,
        # Precision (per alarm): of all alarms raised, what share pointed at a real fault?
        # Low precision = people waste time reviewing noise and start ignoring alarms.
        "precision": n_true / len(alerts) if alerts else 0.0,
        # Recall (per fault): of the faults injected, what share did we catch at least once?
        # Low recall = real problems slip through.
        "recall": len(caught) / len(faulty) if faulty else 0.0,
        "first_alarm_hour": first_alarm,
    }
