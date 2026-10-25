"""Steps 3 + 4: univariate SPC limits and multivariate PCA (Hotelling's T²) monitoring."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SENSORS = ["ph", "temp_c", "do_pct", "agitation_rpm"]


def make_alert(row, sensor, value, limit, method):
    """One alert = one reading that crossed a limit. Plain Python types so it can go into JSON."""
    return {"batch": row.batch_id, "hour": int(row.hour), "sensor": sensor,
            "value": round(float(value), 3), "limit": round(float(limit), 3), "method": method}


# ---------------- Univariate SPC: each sensor on its own ----------------

def compute_limits(reference):
    """Control limits = mean ± 3 standard deviations, per sensor.

    These must come from KNOWN-GOOD batches only. Limits are a description of
    "normal". If a faulty batch were included, its abnormal values would shift the
    mean and inflate the standard deviation, so the limits would get wider and the
    monitor would learn to accept the very fault it is meant to catch.
    """
    mean = reference[SENSORS].mean()
    sigma = reference[SENSORS].std()
    return pd.DataFrame({"mean": mean, "lower": mean - 3 * sigma, "upper": mean + 3 * sigma})


def spc_alerts(data, limits):
    """Flag every reading that falls outside its sensor's control limits."""
    alerts = []
    for sensor in SENSORS:
        low, high = limits.loc[sensor, "lower"], limits.loc[sensor, "upper"]
        outside = data[(data[sensor] < low) | (data[sensor] > high)]
        for row in outside.itertuples():
            value = getattr(row, sensor)
            limit = high if value > high else low     # report whichever limit was crossed
            alerts.append(make_alert(row, sensor, value, limit, "SPC_3sigma"))
    return alerts


# ---------------- Multivariate: all sensors together ----------------

def fit_pca(reference):
    """Learn what "normal" looks like across all sensors at once, from good batches only.

    1. StandardScaler puts every sensor on the same scale (pH varies by ~0.03,
       agitation by ~12 rpm), so no sensor dominates just because of its units.
    2. PCA rotates the data onto uncorrelated directions (principal components).
       With only 4 sensors we keep all 4 components, so T² here equals the
       Mahalanobis distance. With hundreds of sensors you would keep only the top
       few components and add a residual statistic (SPE / Q) for the rest.
    3. The alarm threshold is the 99.73th percentile of T² on good data, i.e. the
       same expected false-alarm rate as the 3-sigma rule (0.27%).
    """
    scaler = StandardScaler().fit(reference[SENSORS])
    pca = PCA().fit(scaler.transform(reference[SENSORS]))
    model = {"scaler": scaler, "pca": pca}
    model["threshold"] = np.percentile(hotelling_t2(reference, model), 99.73)
    return model


def hotelling_t2(data, model):
    """T² = sum over components of (score² / that component's variance in good data).

    Roughly: "how far from normal is this hour, counting all sensors AND how they
    usually move together". A DO value that is fine on its own, but too low for
    the current agitation, gets a high T² even though SPC sees nothing.
    """
    scores = model["pca"].transform(model["scaler"].transform(data[SENSORS]))
    return np.sum(scores ** 2 / model["pca"].explained_variance_, axis=1)


def pca_alerts(data, model):
    """Flag every hour whose T² is above the threshold learned from good batches."""
    t2 = hotelling_t2(data, model)
    high = t2 > model["threshold"]           # True/False for each hour
    return [make_alert(row, "all_sensors", value, model["threshold"], "PCA_T2")
            for row, value in zip(data[high].itertuples(), t2[high])]
