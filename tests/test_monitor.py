"""Validation tests: does the monitor do what we claim?  Run with:  pytest -v"""
import pandas as pd
import pytest

from monitor import compute_limits
from run import run_pipeline
from simulate import HOURS, N_REFERENCE


@pytest.fixture(scope="module")
def result():
    # Run the pipeline once with an in-memory database, so the tests leave no files behind.
    # scope="module" = share this one result across all tests in the file (faster).
    return run_pipeline(":memory:")


def test_all_three_injected_faults_are_detected(result):
    s = result["summary"]
    assert s["faults_injected"] == 3
    assert s["faults_caught"] == 3
    assert s["recall"] == 1.0


def test_good_batches_raise_few_false_alarms(result):
    batches = result["batches"]
    good = [b for b in result["new"].batch_id.unique() if batches.loc[b, "injected_fault"] == "none"]
    alarms_on_good = [a for a in result["alerts"] if a["batch"] in good]
    checks = len(good) * HOURS * 5            # each hour: 4 sensor checks + 1 T² check
    assert len(alarms_on_good) / checks < 0.01   # 3-sigma alone expects about 0.27%
    assert result["summary"]["precision"] > 0.9


def test_limits_come_from_known_good_batches_only(result):
    ref, batches = result["reference"], result["batches"]
    ref_ids = sorted(ref.batch_id.unique())
    assert ref_ids == [f"B{i:02d}" for i in range(1, N_REFERENCE + 1)]
    assert (batches.loc[ref_ids, "injected_fault"] == "none").all()
    # The stored limit must equal mean + 3*sigma of the reference data, nothing else
    expected = ref.temp_c.mean() + 3 * ref.temp_c.std()
    assert result["limits"].loc["temp_c", "upper"] == pytest.approx(expected)


def test_a_faulty_batch_in_the_reference_would_widen_limits(result):
    # Shows WHY limits need good data: adding the temperature-drift batch inflates sigma.
    new = result["new"]
    polluted = pd.concat([result["reference"], new[new.batch_id == "B12"]])
    assert compute_limits(polluted).loc["temp_c", "upper"] > result["limits"].loc["temp_c", "upper"]


def test_pca_catches_the_do_decline_that_spc_misses(result):
    first = result["summary"]["first_alarm_hour"]["B18"]
    assert first["SPC_3sigma"] is None          # DO never leaves its own limits
    assert first["PCA_T2"] is not None          # but it breaks the DO-agitation relationship


def test_alerts_have_the_required_fields(result):
    for alert in result["alerts"]:
        assert set(alert) == {"batch", "hour", "sensor", "value", "limit", "method"}
