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
        bc.compute_annual_bill(load, scenario, "Low", "G06000101", "pge", retail_data=rd),
        72.0, abs_tol=1e-6)
    # Medium income: fixed = $24 * 12 = $288
    assert math.isclose(
        bc.compute_annual_bill(load, scenario, "Medium", "G06000101", "pge", retail_data=rd),
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
        load, scenario, "Low", "G06000101", "pge", retail_data=rd)
    high_bill = bc.compute_annual_bill(
        load, scenario, "High", "G06000101", "pge", retail_data=rd)
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
    med = bc.compute_annual_bill(load, scenario, "Medium", "G06000101", "pge", retail_data=rd)
    high = bc.compute_annual_bill(load, scenario, "High", "G06000101", "pge", retail_data=rd)
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
        load, scenario, "Low", "G06000101", "pge", retail_data=rd)
    expected_low = expected_peak_hours * 0.65 + 6.0 * 12
    assert math.isclose(bill_low, expected_low, abs_tol=1e-6)

    bill_high = bc.compute_annual_bill(
        load, scenario, "High", "G06000101", "pge", retail_data=rd)
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
            np.zeros(100), scenario, "Low", "G06000101", "pge", retail_data=rd)
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


# ============================================================================
# Hourly load loader (Stage 1)
# ============================================================================

def test_load_hourly_returns_none_when_baseline_dir_missing(tmp_path,
                                                             monkeypatch):
    """If Baseline_<U>/ folder isn't present, loader returns None
    (so callers can skip gracefully without the parent hourly data)."""
    from src import config
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": tmp_path / "nonexistent_dir"}})
    assert bc.load_hourly_baseline_load("pge", 12345) is None


def test_load_hourly_returns_none_when_no_matching_parquet(tmp_path,
                                                            monkeypatch):
    """Directory exists but no matching <bldg_id>-*.parquet -> None."""
    from src import config
    base = tmp_path / "Baseline_PGE"
    base.mkdir()
    (base / "99999-0.parquet").touch()  # an unrelated parquet
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": base}})
    assert bc.load_hourly_baseline_load("pge", 12345) is None


def test_load_hourly_aggregates_15min_to_8760(tmp_path, monkeypatch):
    """Builds a synthetic 35040-row parquet and verifies the loader
    sums every 4 rows to produce an 8760-array."""
    from src import config
    base = tmp_path / "Baseline_PGE"
    base.mkdir()
    fake = pd.DataFrame({
        bc.BASELINE_PARQUET_COL: np.full(35040, 0.25),  # 0.25 kWh / 15-min
    })
    fake.to_parquet(base / "42-0.parquet")
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": base}})
    out = bc.load_hourly_baseline_load("pge", 42)
    assert out is not None
    assert out.shape == (8760,)
    # Each hour: 4 × 0.25 = 1.0 kWh
    assert np.allclose(out, 1.0)


def test_load_hourly_returns_none_when_column_missing(tmp_path, monkeypatch):
    from src import config
    base = tmp_path / "Baseline_PGE"
    base.mkdir()
    pd.DataFrame({"some_other_col": np.zeros(100)}).to_parquet(
        base / "7-0.parquet")
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": base}})
    assert bc.load_hourly_baseline_load("pge", 7) is None


def test_sizing_optimizer_hourly_reuses_same_loader():
    """sizing_optimizer_hourly re-exports the same function (DRY)."""
    from src import sizing_optimizer_hourly as soh
    assert soh.load_hourly_load is bc.load_hourly_baseline_load


# ============================================================================
# Hourly Upgrade 11 loader + delta (Stage 2)
# ============================================================================

def _setup_paired_parquets(tmp_path, monkeypatch, bldg_id,
                            baseline_kwh_per_15min,
                            upgrade11_kwh_per_15min):
    """Write a matched pair of Baseline / Upgrade11 parquets for one
    building and monkeypatch config to point at them."""
    from src import config
    base_dir = tmp_path / "Baseline_PGE"
    upg_dir = tmp_path / "Upgrade11_PGE"
    base_dir.mkdir()
    upg_dir.mkdir()
    pd.DataFrame({
        bc.BASELINE_PARQUET_COL: np.full(35040, baseline_kwh_per_15min),
    }).to_parquet(base_dir / f"{bldg_id}-0.parquet")
    pd.DataFrame({
        bc.BASELINE_PARQUET_COL: np.full(35040, upgrade11_kwh_per_15min),
    }).to_parquet(upg_dir / f"{bldg_id}-11.parquet")
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": base_dir,
                 "upgrade11_parquets": upg_dir}})


def test_load_hourly_upgrade11_load_returns_8760(tmp_path, monkeypatch):
    _setup_paired_parquets(tmp_path, monkeypatch, 100, 0.25, 0.5)
    out = bc.load_hourly_upgrade11_load("pge", 100)
    assert out is not None
    assert out.shape == (8760,)
    assert np.allclose(out, 2.0)   # 4 × 0.5 kWh/15min


def test_load_hourly_upgrade11_load_returns_none_when_dir_missing(
        tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "upgrade11_parquets": tmp_path / "nonexistent"}})
    assert bc.load_hourly_upgrade11_load("pge", 100) is None


def test_load_hourly_upgrade11_load_returns_none_when_config_is_none(
        tmp_path, monkeypatch):
    """Older config entries set upgrade11_parquets to None directly.
    Loader handles this without crashing."""
    from src import config
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "upgrade11_parquets": None}})
    assert bc.load_hourly_upgrade11_load("pge", 100) is None


def test_upgrade11_delta_is_upgrade_minus_baseline(tmp_path, monkeypatch):
    """upgrade11 at 0.5 kWh/15min, baseline at 0.25 kWh/15min:
    delta per hour = (4 × 0.5) − (4 × 0.25) = 1.0 kWh."""
    _setup_paired_parquets(tmp_path, monkeypatch, 200, 0.25, 0.5)
    delta = bc.load_hourly_upgrade11_delta("pge", 200)
    assert delta is not None
    assert delta.shape == (8760,)
    assert np.allclose(delta, 1.0)


def test_upgrade11_delta_zero_when_equal(tmp_path, monkeypatch):
    """If upgrade11 == baseline at every quarter-hour, delta is zero."""
    _setup_paired_parquets(tmp_path, monkeypatch, 300, 0.3, 0.3)
    delta = bc.load_hourly_upgrade11_delta("pge", 300)
    assert np.allclose(delta, 0.0)


def test_upgrade11_delta_none_when_either_parquet_missing(
        tmp_path, monkeypatch):
    """If only the baseline parquet exists (no upgrade11), delta is None."""
    from src import config
    base = tmp_path / "Baseline_PGE"
    base.mkdir()
    pd.DataFrame({bc.BASELINE_PARQUET_COL: np.zeros(35040)}).to_parquet(
        base / "400-0.parquet")
    monkeypatch.setattr(
        config, "PIPELINE_OUTPUTS",
        {**config.PIPELINE_OUTPUTS,
         "pge": {**config.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": base,
                 "upgrade11_parquets": tmp_path / "no_upgrade_dir"}})
    assert bc.load_hourly_upgrade11_delta("pge", 400) is None


def test_sce_upgrade11_path_configured():
    """Stage 2 also updates config.py: SCE upgrade11_parquets is no
    longer None — it points at Upgrade11_SCE (consistent with
    PGE / SDGE)."""
    from src import config
    sce_path = config.utility_paths("sce")["upgrade11_parquets"]
    assert sce_path is not None
    assert sce_path.name == "Upgrade11_SCE"


# ============================================================================
# EV hourly profile (Stage 3)
# ============================================================================

def test_ev_hourly_load_shape_is_8760():
    out = bc.ev_hourly_load(3000.0, "smart_tou")
    assert out.shape == (8760,)


def test_ev_hourly_load_sum_equals_annual():
    """Total must equal annual_ev_kwh exactly (no leakage from tiling)."""
    for annual in (1000.0, 3636.36, 5000.0):
        out = bc.ev_hourly_load(annual, "smart_tou")
        assert math.isclose(out.sum(), annual, rel_tol=1e-9)


def test_ev_hourly_load_overnight_concentrates_in_early_hours():
    """overnight_offpeak puts 95% of charging in hours 0-6, 5% in 7-23.
    Tile by 365 and aggregate by hour-of-day: hours 0-6 should hold 95%
    of the total annual EV kWh."""
    annual = 3000.0
    out = bc.ev_hourly_load(annual, "overnight_offpeak")
    by_hod = out.reshape(365, 24).sum(axis=0)
    overnight_kwh = by_hod[0:7].sum()
    rest_kwh = by_hod[7:24].sum()
    assert math.isclose(overnight_kwh, annual * 0.95, rel_tol=1e-6)
    assert math.isclose(rest_kwh, annual * 0.05, rel_tol=1e-6)


def test_ev_hourly_load_opportunistic_is_flat():
    annual = 2400.0
    out = bc.ev_hourly_load(annual, "opportunistic")
    expected_per_hour = annual / 8760.0
    assert np.allclose(out, expected_per_hour, rtol=1e-9)


def test_ev_hourly_load_smart_tou_matches_overnight():
    """Per current proxy in vmt_sensitivity, smart_tou == overnight_offpeak.
    If you later change CHARGING_PROFILES['smart_tou'], update this test."""
    annual = 1000.0
    smart = bc.ev_hourly_load(annual, "smart_tou")
    overnight = bc.ev_hourly_load(annual, "overnight_offpeak")
    assert np.allclose(smart, overnight)


def test_ev_hourly_load_unknown_profile_raises():
    try:
        bc.ev_hourly_load(1000.0, "totally_made_up_profile")
    except ValueError as exc:
        assert "totally_made_up_profile" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown profile")


def test_ev_hourly_load_zero_annual_returns_zeros():
    """Bundle with no EV should produce zero EV-load array."""
    out = bc.ev_hourly_load(0.0, "smart_tou")
    assert out.shape == (8760,)
    assert np.all(out == 0.0)


# ============================================================================
# Signed-net-load + EEC export (Stage 5a / 5b)
# ============================================================================

def test_negative_hourly_loads_treated_as_exports():
    """A negative hour in the net-load array represents grid export.
    Without an eec_hourly array, exports get zero credit (matches the
    old positive-only behavior).
    """
    load = np.zeros(8760)
    load[12] = -3.0    # 3 kWh export at noon
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data()
    bill = bc.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge", retail_data=rd)
    # No imports at all (only an export, no EEC); bill = 0
    assert bill == 0.0


def test_export_credit_applied_with_eec_hourly():
    """Export credit = sum(grid_out * eec_hourly)."""
    load = np.zeros(8760)
    load[12] = -3.0    # 3 kWh export at noon
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data()
    eec = np.ones(8760) * 0.08   # flat 8c export rate
    bill = bc.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge",
        eec_hourly=eec, retail_data=rd)
    # Export credit = 3 * 0.08 = $0.24, no fixed, no imports
    # Bill = 0 + 0 - 0.24 = -0.24 (negative = household gets paid)
    assert math.isclose(bill, -0.24, abs_tol=1e-9)


def test_export_credit_uses_hour_specific_eec():
    """EEC varies by hour; credit is sum(grid_out[h] * eec[h])."""
    load = np.zeros(8760)
    load[100] = -1.0
    load[200] = -2.0
    eec = np.zeros(8760)
    eec[100] = 0.05
    eec[200] = 0.15
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data()
    bill = bc.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge",
        eec_hourly=eec, retail_data=rd)
    # Credit = 1*0.05 + 2*0.15 = 0.35 -> bill = -0.35
    assert math.isclose(bill, -0.35, abs_tol=1e-9)


def test_import_export_mix():
    """A mixed-sign hourly load: some imports, some exports."""
    load = np.zeros(8760)
    # Daytime export, evening import (hours pick winter so winter_peak applies)
    load[100] = -2.0       # winter offpeak export
    load[20] = 5.0         # Jan 1 at 8pm -> winter peak (16-21 window)
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data()
    eec = np.ones(8760) * 0.10
    bill = bc.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge",
        eec_hourly=eec, retail_data=rd)
    # Import: 5 kWh * $0.40 (winter peak) = $2.00
    # Export credit: 2 kWh * $0.10 = $0.20
    # Net: 2.00 - 0.20 = $1.80
    assert math.isclose(bill, 1.80, abs_tol=1e-9)


def test_eec_hourly_shape_validated():
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail_data()
    try:
        bc.compute_annual_bill(
            np.zeros(8760), scenario, "Low", "G06000101", "pge",
            eec_hourly=np.zeros(100), retail_data=rd)
    except ValueError as exc:
        assert "8760" in str(exc) and "eec" in str(exc).lower()
        return
    raise AssertionError("expected ValueError for wrong eec shape")


def test_positive_only_load_path_unchanged():
    """If hourly_net_load is all positive (no PV), result must equal
    the bill we'd get without thinking about exports. Regression on the
    pre-Stage-5 API."""
    load = np.ones(8760) * 0.5
    scenario = pd.Series({
        "summer_peak": 0.10, "summer_offpeak": 0.10,
        "winter_peak": 0.10, "winter_offpeak": 0.10,
        "fixed_monthly_care": 6.0, "fixed_monthly_non_care": 24.0,
    })
    rd = _synthetic_retail_data(care_discount=0.0)
    # 0.5 kWh/hr * 8760 = 4380 kWh; rate $0.10 -> $438 vol + $288 fixed
    bill = bc.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge", retail_data=rd)
    assert math.isclose(bill, 438.0 + 288.0, abs_tol=0.1)


def test_load_hourly_eec_returns_8760_for_all_utilities():
    """eec_hourly_2025_wide.csv must yield 8760-arrays per utility."""
    for u in ("pge", "sce", "sdge"):
        arr = bc.load_hourly_eec(u)
        assert arr.shape == (8760,), u
        assert arr.dtype == float, u


def test_load_hourly_eec_pge_avg_in_expected_range():
    """PGE NBT 2025 hourly EEC averages ~$0.10/kWh per CPUC EEC table."""
    arr = bc.load_hourly_eec("pge")
    assert 0.03 < arr.mean() < 0.20


def test_load_hourly_eec_is_cached():
    """Repeated calls return the same array object (cache hit)."""
    a = bc.load_hourly_eec("pge")
    b = bc.load_hourly_eec("pge")
    assert a is b


def test_load_hourly_eec_unknown_utility_raises():
    try:
        bc.load_hourly_eec("ladwp")
    except ValueError as exc:
        assert "ladwp" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown utility")


# ============================================================================
# Expanded-hourly-load assembly (Stage 4)
# ============================================================================

def test_assemble_baseline_only_returns_baseline_unchanged():
    baseline = np.ones(8760) * 0.5
    out = bc.assemble_bundle_hourly_load(baseline)
    assert out.shape == (8760,)
    assert np.allclose(out, 0.5)


def test_assemble_returns_a_copy_not_the_input():
    """Caller shouldn't accidentally mutate the baseline array."""
    baseline = np.ones(8760) * 1.0
    out = bc.assemble_bundle_hourly_load(baseline)
    out[0] = 999.0
    assert baseline[0] == 1.0


def test_assemble_ev_adds_to_baseline():
    baseline = np.ones(8760) * 0.5
    ev = np.ones(8760) * 0.2
    out = bc.assemble_bundle_hourly_load(baseline, ev_load=ev)
    assert np.allclose(out, 0.7)


def test_assemble_hp_delta_adds_to_baseline():
    baseline = np.ones(8760) * 0.5
    hp = np.full(8760, 0.3)
    out = bc.assemble_bundle_hourly_load(baseline, hp_delta=hp)
    assert np.allclose(out, 0.8)


def test_assemble_pv_subtracts_from_baseline():
    baseline = np.ones(8760) * 0.5
    pv = np.ones(8760) * 0.2
    out = bc.assemble_bundle_hourly_load(baseline, pv_gen=pv)
    assert np.allclose(out, 0.3)


def test_assemble_full_bundle_composition():
    """Full bundle: net = baseline + ev + hp - pv + battery"""
    baseline = np.ones(8760) * 1.0
    ev = np.ones(8760) * 0.3
    hp = np.ones(8760) * 0.2
    pv = np.ones(8760) * 0.7
    batt = np.ones(8760) * 0.1
    out = bc.assemble_bundle_hourly_load(
        baseline, ev_load=ev, hp_delta=hp, pv_gen=pv, battery_net=batt)
    # 1.0 + 0.3 + 0.2 - 0.7 + 0.1 = 0.9
    assert np.allclose(out, 0.9)


def test_assemble_can_go_negative_when_pv_exceeds_load():
    """When PV gen exceeds load at an hour, net load is negative
    (i.e. household is exporting to grid)."""
    baseline = np.zeros(8760)
    baseline[12] = 1.0
    pv = np.zeros(8760)
    pv[12] = 5.0  # solar surplus at noon
    out = bc.assemble_bundle_hourly_load(baseline, pv_gen=pv)
    assert out[12] == -4.0


def test_assemble_battery_net_positive_increases_import():
    """battery_net > 0 means battery is charging from the grid; net
    grid import goes up."""
    baseline = np.ones(8760) * 0.5
    batt = np.zeros(8760)
    batt[2] = 1.0  # battery charging at 2am
    out = bc.assemble_bundle_hourly_load(baseline, battery_net=batt)
    assert out[2] == 1.5


def test_assemble_battery_net_negative_reduces_import():
    """battery_net < 0 means battery is discharging to household;
    net grid import goes down."""
    baseline = np.ones(8760) * 1.0
    batt = np.zeros(8760)
    batt[18] = -0.8  # battery discharging at 6pm
    out = bc.assemble_bundle_hourly_load(baseline, battery_net=batt)
    assert math.isclose(out[18], 0.2, abs_tol=1e-9)


def test_assemble_rejects_wrong_shape_baseline():
    try:
        bc.assemble_bundle_hourly_load(np.zeros(100))
    except ValueError as exc:
        assert "8760" in str(exc) and "baseline" in str(exc)
        return
    raise AssertionError("expected ValueError for wrong baseline shape")


def test_assemble_rejects_wrong_shape_component():
    """If a non-None component isn't shape (8760,), raise."""
    try:
        bc.assemble_bundle_hourly_load(
            np.zeros(8760), ev_load=np.zeros(100))
    except ValueError as exc:
        assert "8760" in str(exc) and "ev_load" in str(exc)
        return
    raise AssertionError("expected ValueError for wrong ev shape")


def test_assemble_linearity():
    """net = baseline + sum(signed components) — linearity sanity."""
    baseline = np.linspace(0, 1, 8760)
    ev = np.linspace(0, 0.5, 8760)
    hp = np.linspace(0, 0.3, 8760)
    pv = np.linspace(0, 0.8, 8760)
    expected = baseline + ev + hp - pv
    out = bc.assemble_bundle_hourly_load(
        baseline, ev_load=ev, hp_delta=hp, pv_gen=pv)
    assert np.allclose(out, expected)


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
        load, scenario, "Medium", sample_puma, "pge", retail_data=rd)
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
