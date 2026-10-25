"""Run the whole pipeline with one command:  python run.py

simulate -> SQLite -> SQL queries -> SPC + PCA -> alerts JSON + chart -> validation summary
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # draw straight to a PNG file, no pop-up window
import matplotlib.pyplot as plt

from database import batch_summary, biggest_ph_jumps, build_database, load_batches, load_readings
from evaluate import evaluate
from monitor import compute_limits, fit_pca, hotelling_t2, pca_alerts, spc_alerts
from simulate import simulate_all

OUT = Path(__file__).parent / "outputs"
CHART_BATCH, CHART_SENSOR = "B18", "do_pct"   # the DO-decline batch: the most interesting chart


def run_pipeline(db_path):
    """Do all the work and return the results (no printing, no files). Tests call this too."""
    batches, readings = simulate_all()
    conn = build_database(batches, readings, db_path)

    batches = load_batches(conn)
    reference = load_readings(conn, reference=True)    # 10 known-good batches -> learn "normal"
    new = load_readings(conn, reference=False)         # 10 new batches -> monitor them

    limits = compute_limits(reference)
    model = fit_pca(reference)
    alerts = spc_alerts(new, limits) + pca_alerts(new, model)
    alerts.sort(key=lambda a: (a["batch"], a["hour"]))

    summary = evaluate(alerts, batches, sorted(new.batch_id.unique()))
    return {"conn": conn, "batches": batches, "reference": reference, "new": new,
            "limits": limits, "model": model, "alerts": alerts, "summary": summary}


def save_alerts(alerts, path):
    report = {"data_source": "SIMULATED - not real plant data",
              "intended_use": "Decision support only. A person reviews every alert.",
              "n_alerts": len(alerts),
              "alerts": alerts}
    path.write_text(json.dumps(report, indent=2))


def plot_control_chart(result, path):
    """Two stacked charts for one batch: DO vs its SPC limits (top), T² vs its limit (bottom)."""
    data = result["new"][result["new"].batch_id == CHART_BATCH]
    lim = result["limits"].loc[CHART_SENSOR]
    t2 = hotelling_t2(data, result["model"])
    threshold = result["model"]["threshold"]
    fault_start = result["batches"].loc[CHART_BATCH, "fault_start_hour"]

    blue, ink, alarm = "#2a78d6", "#52514e", "#d03b3b"
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    top.plot(data.hour, data[CHART_SENSOR], color=blue, lw=1.2)
    top.axhline(lim["mean"], color=ink, lw=0.8)
    for y in (lim["lower"], lim["upper"]):
        top.axhline(y, color=ink, lw=1, ls="--")
    top.text(1, lim["upper"], " +3σ limit", va="bottom", color=ink, fontsize=9)
    top.text(1, lim["lower"], " -3σ limit", va="bottom", color=ink, fontsize=9)
    top.set_ylabel("Dissolved oxygen (%)")
    top.set_title("SPC: DO stays inside its 3σ limits, so univariate monitoring raises no alarm", loc="left")

    over = t2 > threshold
    bottom.plot(data.hour, t2, color=blue, lw=1.2)
    bottom.scatter(data.hour[over], t2[over], color=alarm, s=12, zorder=3, label="T² alarm")
    bottom.axhline(threshold, color=ink, lw=1, ls="--")
    bottom.text(1, threshold, " T² limit", va="bottom", color=ink, fontsize=9)
    bottom.set_yscale("log")                 # T² grows a lot; log scale keeps small values visible
    bottom.set_ylabel("Hotelling's T² (log)")
    bottom.set_xlabel("Hour of batch")
    bottom.set_title("PCA: T² flags DO that is too low for the current agitation", loc="left")
    bottom.legend(loc="upper left", frameon=False)

    for ax in (top, bottom):
        ax.axvline(fault_start, color=ink, lw=0.8, ls=":")
        ax.text(fault_start, 0.92, " fault starts", transform=ax.get_xaxis_transform(), color=ink, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    fig.suptitle(f"SIMULATED DATA - batch {CHART_BATCH}, slow DO decline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    print("=" * 64 + "\n  BIOREACTOR BATCH MONITOR  -  ALL DATA IS SIMULATED\n" + "=" * 64)
    result = run_pipeline(OUT / "bioreactor.db")

    print("\nPer-batch summary (SQL: JOIN + GROUP BY)")
    print(batch_summary(result["conn"]).to_string(index=False))
    print("\nLargest hour-to-hour pH changes (SQL window function: LAG)")
    print(biggest_ph_jumps(result["conn"]).to_string(index=False))
    print("\nControl limits from the 10 known-good reference batches (mean ± 3 sigma)")
    print(result["limits"].round(3).to_string())
    print(f"Hotelling's T² limit (99.73th percentile of good data): {result['model']['threshold']:.1f}")

    save_alerts(result["alerts"], OUT / "alerts.json")
    plot_control_chart(result, OUT / f"control_chart_{CHART_BATCH}.png")

    s, batches = result["summary"], result["batches"]
    print("\nValidation summary (monitored batches B11-B20)")
    print(f"  Faults caught : {s['faults_caught']} of {s['faults_injected']}")
    print(f"  True alarms   : {s['true_alarms']}")
    print(f"  False alarms  : {s['false_alarms']}")
    print(f"  Precision     : {s['precision']:.1%}   (share of alarms that were real)")
    print(f"  Recall        : {s['recall']:.0%}    (share of faults caught)")
    print("\nFirst alarm hour per fault (None = method never caught it)")
    for b, firsts in s["first_alarm_hour"].items():
        print(f"  {b} {batches.loc[b, 'injected_fault']:<11} starts hour {int(batches.loc[b, 'fault_start_hour']):>3}"
              f"  ->  SPC: {firsts['SPC_3sigma']},  PCA T²: {firsts['PCA_T2']}")
    print(f"\nSaved {len(result['alerts'])} alerts to outputs/alerts.json "
          f"and the chart to outputs/control_chart_{CHART_BATCH}.png")


if __name__ == "__main__":
    main()
