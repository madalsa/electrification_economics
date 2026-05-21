"""Tests for bill.py — replicates user's *_baseline_bills.py methodology."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import bill


# ---- Period masks ----

def test_pge_4_periods_cover_8760():
    masks = bill.build_period_masks("pge")
    assert set(masks) == {"summer_peak", "summer_offpeak",
                           "winter_peak", "winter_offpeak"}
    assert sum(m.sum() for m in masks.values()) == 8760


def test_sce_5_periods_cover_8760():
    masks = bill.build_period_masks("sce")
    assert "winter_midpeak" in masks
    assert sum(m.sum() for m in masks.values()) == 8760


def test_sdge_6_periods_cover_8760():
    masks = bill.build_period_masks("sdge")
    assert "summer_midpeak" in masks and "winter_midpeak" in masks
    assert sum(m.sum() for m in masks.values()) == 8760


def test_5pm_in_july_is_summer_peak_all_utilities():
    h = 181 * 24 + 17  # July 1 at 5pm
    for u in ("pge", "sce", "sdge"):
        assert bill.build_period_masks(u)["summer_peak"][h]


# ---- EV hourly load ----

def test_ev_hourly_sums_to_annual():
    out = bill.ev_hourly_load(3000.0, "smart_tou")
    assert out.shape == (8760,)
    assert math.isclose(out.sum(), 3000.0, rel_tol=1e-9)


def test_ev_overnight_profile_concentrated_in_early_hours():
    out = bill.ev_hourly_load(1000.0, "overnight_offpeak")
    by_hod = out.reshape(365, 24).sum(axis=0)
    assert math.isclose(by_hod[0:7].sum(), 950.0, rel_tol=1e-6)


# ---- Assembly ----

def test_assemble_baseline_only_returns_baseline_copy():
    baseline = np.ones(8760) * 0.5
    out = bill.assemble_bundle_hourly_load(baseline)
    assert np.allclose(out, 0.5)
    out[0] = 999
    assert baseline[0] == 0.5  # input not mutated


def test_assemble_signed_can_go_negative():
    baseline = np.zeros(8760)
    baseline[12] = 1.0
    pv = np.zeros(8760); pv[12] = 5.0
    out = bill.assemble_bundle_hourly_load(baseline, pv_gen=pv)
    assert out[12] == -4.0


# ---- Hourly loaders (None-paths only; real parquets not in sandbox) ----

def test_load_hourly_returns_none_when_baseline_dir_missing(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "PIPELINE_OUTPUTS",
        {**cfg.PIPELINE_OUTPUTS,
         "pge": {**cfg.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets": tmp_path / "no_dir"}})
    assert bill.load_hourly_baseline_load("pge", 1) is None


def test_upgrade11_delta_none_when_either_missing(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "PIPELINE_OUTPUTS",
        {**cfg.PIPELINE_OUTPUTS,
         "pge": {**cfg.PIPELINE_OUTPUTS["pge"],
                 "baseline_parquets":  tmp_path / "no1",
                 "upgrade11_parquets": tmp_path / "no2"}})
    assert bill.load_hourly_upgrade11_delta("pge", 1) is None


# ---- Retail data + EEC (integration with real files in repo) ----

def test_load_retail_data_pge_real_values():
    rd = bill.load_retail_data("pge")
    assert math.isclose(rd["care_discount"], 0.35, abs_tol=0.001)
    assert len(rd["baseline_df"]) == 275


def test_eec_hourly_pge_realistic_range():
    eec = bill.load_hourly_eec("pge")
    assert eec.shape == (8760,)
    assert 0.03 < eec.mean() < 0.20


def test_eec_multiplier_scales_array():
    base = bill.load_hourly_eec("pge", multiplier=1.0)
    scaled = bill.load_hourly_eec("pge", multiplier=1.5)
    assert np.allclose(scaled, base * 1.5)


# ---- Annual bill ----

def _synthetic_retail():
    return {
        "care_discount": 0.35,
        "baseline_credit_rate": 0.0,
        "baseline_df": pd.DataFrame({
            "puma": ["G06000101"],
            "summer_baseline_allowance": [0.0],
            "winter_baseline_allowance": [0.0]}),
    }


def test_bill_zero_load_yields_just_fixed_charge():
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 6.0, "fixed_monthly_non_care": 24.0,
    })
    rd = _synthetic_retail()
    assert math.isclose(
        bill.compute_annual_bill(np.zeros(8760), scenario, "Low",
                                  "G06000101", "pge", retail_data=rd),
        72.0)
    assert math.isclose(
        bill.compute_annual_bill(np.zeros(8760), scenario, "High",
                                  "G06000101", "pge", retail_data=rd),
        288.0)


def test_bill_care_discount_applied():
    load = np.ones(8760) * 1.0
    scenario = pd.Series({
        "summer_peak": 0.10, "summer_offpeak": 0.10,
        "winter_peak": 0.10, "winter_offpeak": 0.10,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail()
    low = bill.compute_annual_bill(load, scenario, "Low",
                                    "G06000101", "pge", retail_data=rd)
    high = bill.compute_annual_bill(load, scenario, "High",
                                     "G06000101", "pge", retail_data=rd)
    assert math.isclose(high, 876.0, abs_tol=0.1)
    assert math.isclose(low, 876.0 * 0.65, abs_tol=0.1)


def test_bill_export_credit_with_eec():
    """3 kWh export at noon, $0.08 EEC -> $0.24 credit (negative bill)."""
    load = np.zeros(8760)
    load[12] = -3.0
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    eec = np.ones(8760) * 0.08
    rd = _synthetic_retail()
    bill_amount = bill.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge",
        eec_hourly=eec, retail_data=rd)
    assert math.isclose(bill_amount, -0.24, abs_tol=1e-9)


def test_bill_no_eec_means_no_export_credit():
    """Without eec_hourly, exports are uncompensated (legacy positive-
    only behavior preserved)."""
    load = np.zeros(8760); load[12] = -3.0
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail()
    bill_amount = bill.compute_annual_bill(
        load, scenario, "Medium", "G06000101", "pge", retail_data=rd)
    assert bill_amount == 0.0


def test_bill_shape_validation():
    scenario = pd.Series({
        "summer_peak": 0.5, "summer_offpeak": 0.3,
        "winter_peak": 0.4, "winter_offpeak": 0.2,
        "fixed_monthly_care": 0.0, "fixed_monthly_non_care": 0.0,
    })
    rd = _synthetic_retail()
    try:
        bill.compute_annual_bill(np.zeros(100), scenario, "Low",
                                  "G06000101", "pge", retail_data=rd)
    except ValueError as e:
        assert "8760" in str(e)
        return
    raise AssertionError("expected ValueError")


# ---- PV profile ----

def test_solar_per_kw_sums_to_yield():
    """Synthetic fallback should sum to ~1700 kWh/kW/yr."""
    p = bill.get_solar_per_kw(9, "pge")
    assert p.shape == (8760,)
    assert math.isclose(p.sum(), bill.SYNTHETIC_PV_KWH_PER_KW_YR, rel_tol=1e-6)


if __name__ == "__main__":
    failures = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj() if name.split("_")[1] != "load" else obj(
                    Path("/tmp"), type("M", (), {"setattr": lambda *a: None})())
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL/ERR  {name}  {type(e).__name__}: {e}")
                failures += 1
    sys.exit(failures)
