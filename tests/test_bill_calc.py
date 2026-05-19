"""Tests for bill_calc — replicates user's *_baseline_bills.py methodology.

Most tests use synthetic hourly loads + a synthetic rate scenario so they
run without parent data files. A handful of integration tests use the
real retail Excel and rate scenarios (which ARE in the repo now).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import bill_calc as bc


# ============================================================================
# Period masks: structure + total coverage
# ============================================================================

def test_pge_period_masks_cover_all_8760_hours_exactly_once():
    masks = bc.build_period_masks("pge")
    assert set(masks.keys()) == {
        "summer_peak", "summer_offpeak", "winter_peak", "winter_offpeak"}
    total = sum(m.sum() for m in masks.values())
    assert total == 8760


def test_sce_period_masks_cover_all_8760_hours():
    masks = bc.build_period_masks("sce")
    assert set(masks.keys()) == {
        "summer_peak", "summer_offpeak",
        "winter_peak", "winter_midpeak", "winter_offpeak"}
    total = sum(m.sum() for m in masks.values())
    assert total == 8760


def test_sdge_period_masks_cover_all_8760_hours():
    masks = bc.build_period_masks("sdge")
    assert set(masks.keys()) == {
        "summer_peak", "summer_midpeak", "summer_offpeak",
        "winter_peak", "winter_midpeak", "winter_offpeak"}
    total = sum(m.sum() for m in masks.values())
    assert total == 8760


def test_summer_months_match_user_convention():
    """User's *_baseline_bills.py uses months 6-10 (Jun-Oct) across all
    three utilities. Regression check that we mirror this."""
    for u in ("pge", "sce", "sdge"):
        assert bc.SUMMER_MONTHS_BY_UTILITY[u] == (6, 10)


def test_pge_no_midpeak_period():
    """PGE has 4 periods, no midpeak (matches actual E-TOU-C tariff
    structure and the user's pge_baseline_bills code)."""
    masks = bc.build_period_masks("pge")
    assert not any("midpeak" in k for k in masks.keys())


def test_sce_has_only_winter_midpeak_not_summer():
    """SCE has winter midpeak only (no summer midpeak), matching the
    rate_scenarios_sce_fresh.csv schema."""
    masks = bc.build_period_masks("sce")
    assert "winter_midpeak" in masks
    assert "summer_midpeak" not in masks


def test_sdge_has_both_midpeaks():
    masks = bc.build_period_masks("sdge")
    assert "winter_midpeak" in masks
    assert "summer_midpeak" in masks


# ============================================================================
# Period mask hour landings (sanity)
# ============================================================================

def test_5pm_in_july_is_summer_peak_all_utilities():
    """Hour 17 on July 1 (day-of-year 182) should land summer_peak for
    all utilities. Hour index = (181 * 24) + 17 = 4361."""
    h = 181 * 24 + 17
    for u in ("pge", "sce", "sdge"):
        masks = bc.build_period_masks(u)
        assert masks["summer_peak"][h], u


def test_2am_in_january_is_winter_offpeak_all_utilities():
    """January 2nd at 2am: hour index = 24 + 2 = 26."""
    h = 26
    for u in ("pge", "sce", "sdge"):
        masks = bc.build_period_masks(u)
        assert masks["winter_offpeak"][h], u


def test_10am_in_february_is_winter_midpeak_sce_sdge():
    """SCE winter midpeak 8-16; SDGE midpeak 6-16 OR 21-22.
    Feb 1 at 10am: hour index = 31*24 + 10 = 754. Should be winter_midpeak
    for SCE and SDGE, winter_offpeak for PGE (no midpeak)."""
    h = 31 * 24 + 10
    assert bc.build_period_masks("sce")["winter_midpeak"][h]
    assert bc.build_period_masks("sdge")["winter_midpeak"][h]
    assert bc.build_period_masks("pge")["winter_offpeak"][h]


# ============================================================================
# Hourly rate array
# ============================================================================

def test_hourly_rate_array_lengths_to_8760():
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
    })
    arr = bc.build_hourly_rate_array(scenario, "pge")
    assert arr.shape == (8760,)


def test_hourly_rate_array_applies_periods_correctly():
    """Single-period scenario: only that period's rate should appear."""
    scenario = pd.Series({
        "summer_peak":    0.99,
        "summer_offpeak": 0.01,
        "winter_peak":    0.01,
        "winter_offpeak": 0.01,
    })
    arr = bc.build_hourly_rate_array(scenario, "pge")
    masks = bc.build_period_masks("pge")
    assert np.all(arr[masks["summer_peak"]] == 0.99)
    assert np.all(arr[masks["summer_offpeak"]] == 0.01)


# ============================================================================
# Baseline credit
# ============================================================================

def test_baseline_credit_zero_when_rate_is_zero():
    load = np.ones(8760) * 1.0
    df = pd.DataFrame({
        "puma": ["G06000101"],
        "summer_baseline_allowance": [10.0],
        "winter_baseline_allowance": [10.0],
    })
    assert bc.compute_baseline_credit(load, "G06000101", df, 0.0) == 0.0


def test_baseline_credit_zero_for_unknown_puma():
    load = np.ones(8760) * 1.0
    df = pd.DataFrame({
        "puma": ["G06000101"],
        "summer_baseline_allowance": [10.0],
        "winter_baseline_allowance": [10.0],
    })
    assert bc.compute_baseline_credit(load, "G06999999", df, 0.1) == 0.0


def test_baseline_credit_capped_at_allowance():
    """If hourly load is very large, baseline credit is capped at
    rate * allowance * days per month, NOT rate * consumption."""
    # 5 kW continuous = 5 * 24 = 120 kWh/day -> 3600+ kWh/month
    load = np.ones(8760) * 5.0
    df = pd.DataFrame({
        "puma": ["G06000101"],
        "summer_baseline_allowance": [10.0],  # 10 kWh/day
        "winter_baseline_allowance": [10.0],
    })
    credit = bc.compute_baseline_credit(load, "G06000101", df, 0.10)
    # Expected: rate(0.10) * 365 days * 10 kWh/day = $365/yr
    assert math.isclose(credit, 365.0, abs_tol=0.01)


def test_baseline_credit_equals_rate_times_load_when_below_allowance():
    """If load is well below baseline allowance, credit = rate * total
    consumption (uncapped)."""
    # 0.1 kW continuous = 2.4 kWh/day, well under 10 kWh/day allowance
    load = np.ones(8760) * 0.1
    df = pd.DataFrame({
        "puma": ["G06000101"],
        "summer_baseline_allowance": [10.0],
        "winter_baseline_allowance": [10.0],
    })
    credit = bc.compute_baseline_credit(load, "G06000101", df, 0.10)
    # Expected: 0.10 * 8760 * 0.1 = $87.60
    assert math.isclose(credit, 87.60, abs_tol=0.01)


# ============================================================================
# Full bill computation (synthetic inputs - no parent files needed)
# ============================================================================

def _synthetic_retail_data(care_discount=0.35, baseline_credit_rate=0.0):
    """Build retail_data dict for tests without touching the real Excel."""
    return {
        "care_discount": care_discount,
        "baseline_credit_rate": baseline_credit_rate,
        "baseline_df": pd.DataFrame({
            "puma": ["G06000101"],
            "summer_baseline_allowance": [0.0],   # zero allowance -> no credit
            "winter_baseline_allowance": [0.0],
        }),
    }


def test_zero_load_zero_vol_bill_only_fixed():
    """No load -> bill is just the fixed annual charge."""
    load = np.zeros(8760)
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care":     6.0,
        "fixed_monthly_non_care": 24.0,
    })
    rd = _synthetic_retail_data(care_discount=0.35)
    # Low income: fixed = $6 * 12 = $72
    assert math.isclose(
        bc.compute_annual_bill(load, scenario, "Low", "G06000101", "pge", rd),
        72.0, abs_tol=1e-6)
    # Medium income: fixed = $24 * 12 = $288
    assert math.isclose(
        bc.compute_annual_bill(load, scenario, "Medium", "G06000101", "pge", rd),
        288.0, abs_tol=1e-6)


def test_care_volumetric_discount_applied_for_low_income():
    """A flat 1 kWh/hour load at uniform $0.10/kWh: vol_bill = $876.
    Low income gets 35% discount -> $569.40. Plus fixed."""
    load = np.ones(8760) * 1.0
    scenario = pd.Series({
        "summer_peak": 0.10, "summer_offpeak": 0.10,
        "winter_peak": 0.10, "winter_offpeak": 0.10,
        "fixed_monthly_care":     0.0,
        "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data(care_discount=0.35)
    low_bill = bc.compute_annual_bill(
        load, scenario, "Low", "G06000101", "pge", rd)
    high_bill = bc.compute_annual_bill(
        load, scenario, "High", "G06000101", "pge", rd)
    assert math.isclose(high_bill, 876.0, abs_tol=0.1)
    assert math.isclose(low_bill, 876.0 * 0.65, abs_tol=0.1)


def test_high_and_medium_both_pay_non_care_fixed():
    """High and Medium are both Non-CARE; both get fixed_monthly_non_care."""
    load = np.zeros(8760)
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care":     6.0,
        "fixed_monthly_non_care": 24.0,
    })
    rd = _synthetic_retail_data()
    med = bc.compute_annual_bill(load, scenario, "Medium", "G06000101", "pge", rd)
    high = bc.compute_annual_bill(load, scenario, "High", "G06000101", "pge", rd)
    assert med == high == 288.0


def test_full_bill_formula_matches_user_baseline_bills():
    """Reproduce the exact PGE baseline-bill formula from
    pge_baseline_bills.py:382-415 on a controlled input.

    Setup: hourly load = 1 kWh constantly; scenario peak = $1, offpeak = $0
    so vol_bill_no_credit = number_of_peak_hours.
    Peak hours per year (PGE): months 6-10, hours 16-21 = 5 months * 30.5
    days/mo on avg * 5 hours = ~762 hours (approximately).
    With allowance=0 -> no credit. With care_discount=0.35, Low:
    vol_bill = (peak_hours * $1) * 0.65 + fixed_annual.
    """
    load = np.ones(8760)
    scenario = pd.Series({
        "summer_peak": 1.0, "summer_offpeak": 0.0,
        "winter_peak": 0.0, "winter_offpeak": 0.0,
        "fixed_monthly_care":     6.0,
        "fixed_monthly_non_care": 24.0,
    })
    rd = _synthetic_retail_data(care_discount=0.35)
    masks = bc.build_period_masks("pge")
    expected_peak_hours = int(masks["summer_peak"].sum())

    bill_low = bc.compute_annual_bill(
        load, scenario, "Low", "G06000101", "pge", rd)
    expected_low = expected_peak_hours * 0.65 + 6.0 * 12
    assert math.isclose(bill_low, expected_low, abs_tol=1e-6)

    bill_high = bc.compute_annual_bill(
        load, scenario, "High", "G06000101", "pge", rd)
    expected_high = expected_peak_hours + 24.0 * 12
    assert math.isclose(bill_high, expected_high, abs_tol=1e-6)


def test_hourly_load_shape_validated():
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data()
    try:
        bc.compute_annual_bill(
            np.zeros(100), scenario, "Low", "G06000101", "pge", rd)
    except ValueError as exc:
        assert "8760" in str(exc)
        return
    raise AssertionError("expected ValueError for wrong-shape load")


# ============================================================================
# Integration with real retail Excel + rate scenarios (in repo)
# ============================================================================

def test_load_retail_data_pge_pulls_real_values():
    """E-TOU-C from real PGE Excel should give care_discount ~35% and
    baseline_credit_rate ~$0.09566."""
    rd = bc.load_retail_data("pge")
    assert math.isclose(rd["care_discount"], 0.35, abs_tol=0.001)
    assert math.isclose(rd["baseline_credit_rate"], 0.09566, abs_tol=0.0001)
    # baseline_df should have 275 PUMA rows
    assert len(rd["baseline_df"]) == 275
    assert {"puma", "summer_baseline_allowance",
            "winter_baseline_allowance"}.issubset(rd["baseline_df"].columns)


def test_load_retail_data_sce_pulls_real_values():
    rd = bc.load_retail_data("sce")
    # SCE care_discount per earlier inspection: -0.325 -> abs 0.325
    assert math.isclose(rd["care_discount"], 0.325, abs_tol=0.001)


def test_load_retail_data_sdge_pulls_real_values():
    rd = bc.load_retail_data("sdge")
    # SDGE TOU-DR care_discount per earlier inspection: -0.37 -> abs 0.37
    assert math.isclose(rd["care_discount"], 0.37, abs_tol=0.001)


def test_real_pge_scenario_bill_sane_order_of_magnitude():
    """Sanity check: F0_WF0_ROE0 with 6000 kWh/yr typical load should
    produce a plausible annual bill (a few thousand $)."""
    import pandas as pd
    rates = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data"
        / "rate_scenarios_extended_pge.csv")
    scenario = rates[rates["scenario_id"] == "F0_WF0_ROE0"].iloc[0]
    # Synthetic flat 6000 kWh/yr profile, valid PUMA (use one from
    # retail Excel)
    rd = bc.load_retail_data("pge")
    sample_puma = rd["baseline_df"]["puma"].iloc[0]
    load = np.ones(8760) * (6000 / 8760)
    bill = bc.compute_annual_bill(
        load, scenario, "Medium", sample_puma, "pge", rd)
    # Non-CARE at $0 fixed (F0) - vol only.
    # Avg rate ~$0.47/kWh * 6000 - baseline credit. Expect roughly
    # $1500-$3000 depending on PUMA allowance.
    assert 1000 < bill < 4000, f"bill ${bill:.0f} out of range"


if __name__ == "__main__":
    failures = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}  {e}")
                failures += 1
            except Exception as e:
                print(f"  ERR   {name}  {type(e).__name__}: {e}")
                failures += 1
    print(f"\n{failures} failure(s)" if failures else "\nall passed")
    sys.exit(failures)
