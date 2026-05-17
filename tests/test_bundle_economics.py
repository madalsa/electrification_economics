"""Smoke tests for bundle_economics."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import bundle_economics as be
from src import config


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ----- bundle name parsing -----

def test_bundles_list_has_full_electrification():
    assert "pv_bat_ev_hp" in be.BUNDLES
    assert "none" in be.BUNDLES


def test_parse_bundle_full():
    assert be.parse_bundle("pv_bat_ev_hp") == (True, True, True)


def test_parse_bundle_pv_only():
    assert be.parse_bundle("pv_bat") == (True, False, False)


def test_parse_bundle_ev_only():
    assert be.parse_bundle("ev") == (False, True, False)


def test_parse_bundle_hp_only():
    assert be.parse_bundle("hp") == (False, False, True)


def test_parse_bundle_pv_bat_hp_excludes_ev():
    # Must not match "ev" substring anywhere; tokens-based parser.
    assert be.parse_bundle("pv_bat_hp") == (True, False, True)


def test_parse_bundle_none_returns_all_false():
    assert be.parse_bundle("none") == (False, False, False)


# ----- load splitters -----

def test_ev_kwh_by_tou_conserves_total():
    # 12,000 mi / 3.3 mi/kWh = 3636.4 kWh - should sum to that across periods
    out = be.ev_kwh_by_tou("pge", vmt=12000, ev_eff=3.3)
    assert math.isclose(sum(out.values()), 12000 / 3.3, rel_tol=1e-6)


def test_ev_kwh_by_tou_overnight_lands_offpeak():
    # PGE: peak is 16-21; overnight profile is 0-7 -> mostly offpeak
    out = be.ev_kwh_by_tou("pge", vmt=12000, ev_eff=3.5,
                           profile="overnight_offpeak")
    peak = out.get("summer_peak", 0) + out.get("winter_peak", 0)
    offpeak = out.get("summer_offpeak", 0) + out.get("winter_offpeak", 0)
    assert offpeak > peak * 10


def test_hp_kwh_by_tou_space_heat_winter_only():
    tou_w = {"summer_peak": 0.1, "summer_offpeak": 0.4,
             "winter_peak": 0.1, "winter_offpeak": 0.4}
    out = be.hp_kwh_by_tou(delta_hp_space=1000,
                           delta_hpwh=0, delta_induction=0,
                           tou_weights=tou_w)
    assert out["summer_peak"] == 0
    assert out["summer_offpeak"] == 0
    assert math.isclose(out["winter_peak"] + out["winter_offpeak"], 1000,
                        rel_tol=1e-6)


def test_hp_kwh_by_tou_hpwh_spreads_yearround():
    tou_w = {"summer_peak": 0.1, "summer_offpeak": 0.4,
             "winter_peak": 0.1, "winter_offpeak": 0.4}
    out = be.hp_kwh_by_tou(delta_hp_space=0,
                           delta_hpwh=1000, delta_induction=0,
                           tou_weights=tou_w)
    assert math.isclose(sum(out.values()), 1000, rel_tol=1e-6)
    # year-round implies non-zero summer
    assert out["summer_offpeak"] > 0


def test_expanded_load_sums():
    base = {"a": 100, "b": 200}
    d1 = {"a": 10, "b": 20}
    d2 = {"a": 1, "b": 2}
    out = be.expanded_load_by_tou(base, d1, d2)
    assert out["a"] == 111
    assert out["b"] == 222


def test_expanded_load_skips_none():
    base = {"a": 100}
    out = be.expanded_load_by_tou(base, None, None)
    assert out == base


# ----- bundle evaluation -----

def _make_test_rate(utility: str = "pge") -> pd.Series:
    """Reasonable PGE-shaped TOU rate row for direct unit tests."""
    return pd.Series({
        "scenario_id": "TEST_TOU",
        "rate_type":   "designed_tou",
        "summer_peak":     0.55,
        "summer_offpeak":  0.30,
        "winter_peak":     0.45,
        "winter_offpeak":  0.28,
    })


def _make_test_building(annual_kwh: float = 8000) -> pd.Series:
    """ResStock-style row with gas heating / DHW / range so HP bundle works."""
    return pd.Series({
        "bldg_id":   1,
        "utility":   "pge",
        "cec_cz":    12,
        "ami_frac":  1.5,
        "annual_kwh": annual_kwh,
        "out.electricity.total.energy_consumption.kwh": annual_kwh,
        "summer_peak_kw": 5.0,
        "out.natural_gas.heating.energy_consumption.kwh":     8000.0,
        "out.natural_gas.hot_water.energy_consumption.kwh":   3000.0,
        "out.natural_gas.range_oven.energy_consumption.kwh":  500.0,
        "delta_kwh_hp_space":   2666.7,  # 8000/3.0
        "delta_kwh_hpwh":       1000.0,  # 3000/3.0
        "delta_kwh_induction":  425.0,   # 500*0.85
        "total_therms_displaced": (8000 + 3000 + 500) / 29.3001,
        "cluster_weight":       1.0,
    })


def _common_eval_args():
    tou_w = {"summer_peak": 0.15, "summer_offpeak": 0.35,
             "winter_peak": 0.15, "winter_offpeak": 0.35}
    prices = {"summer_peak": 0.55, "summer_offpeak": 0.30,
              "winter_peak": 0.45, "winter_offpeak": 0.28}
    ev_params = {
        "vmt": 12000, "gas_price": 4.90, "ev_eff": 3.3,
        "ice_mpg": 27, "scenario": "new_new",
    }
    return tou_w, prices, ev_params


def test_none_bundle_yields_zero_npv():
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    rec = be.evaluate_bundle(
        "none", bldg, rate, "pge",
        baseline, prices, eec=0.09, fixed_monthly=15.0,
        demand_charge=0.0, avg_peak_kw=5.0,
        ev_params=ev_params, air_district="BAAQMD",
        tou_weights=tou_w)
    assert rec["npv"] == 0.0
    assert rec["capex_total"] == 0.0
    assert rec["pv_kw"] == 0.0
    assert rec["batt_kwh"] == 0.0
    # decomposition columns must all be zero too
    for col in ("npv_pv_bat", "npv_ev", "npv_hp",
                "capex_pv_bat", "capex_ev", "capex_hp",
                "gasoline_avoided", "gas_avoided_value",
                "ev_charging_cost", "hp_elec_increase",
                "bill_savings_pv_bat"):
        assert rec[col] == 0.0, col


def test_pv_bat_bundle_has_positive_capex_after_rebates():
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    rec = be.evaluate_bundle(
        "pv_bat", bldg, rate, "pge",
        baseline, prices, eec=0.09, fixed_monthly=15.0,
        demand_charge=0.0, avg_peak_kw=5.0,
        ev_params=ev_params, air_district="BAAQMD",
        tou_weights=tou_w)
    # PV is positive-cost; battery has SGIP rebate but not 100%.
    if rec["pv_kw"] > 0 or rec["batt_kwh"] > 0:
        assert rec["capex_total"] > 0
    assert "npv" in rec
    assert math.isfinite(rec["npv"])


def test_pv_bat_ev_hp_capex_exceeds_pv_bat_alone():
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    rec_pv = be.evaluate_bundle(
        "pv_bat", bldg, rate, "pge",
        baseline, prices, 0.09, 15.0, 0.0, 5.0,
        ev_params, "BAAQMD", tou_w)
    rec_full = be.evaluate_bundle(
        "pv_bat_ev_hp", bldg, rate, "pge",
        baseline, prices, 0.09, 15.0, 0.0, 5.0,
        ev_params, "BAAQMD", tou_w)
    # Adding EV + HP must add capex (HP capex post-rebate > 0; EV premium > 0).
    assert rec_full["capex_total"] > rec_pv["capex_total"]


def test_hp_only_capex_matches_payback_npv_stack():
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    rec = be.evaluate_bundle(
        "hp", bldg, rate, "pge",
        baseline, prices, 0.09, 15.0, 0.0, 5.0,
        ev_params, "BAAQMD", tou_w)
    from src import payback_npv as p
    capex = p.CapexBreakdown(
        heat_pump_space=True, heat_pump_water=True,
        induction_range=True, panel_upgrade=True)
    ctx = p.IncentiveContext(income_pct_ami=1.5)
    expected_capex, _ = p.apply_capex_stack(capex, ctx)
    assert approx(rec["capex_total"], expected_capex)


def test_npv_components_sum_to_total():
    """npv_pv_bat + npv_ev + npv_hp must equal npv (within float tol)."""
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    for bundle in be.BUNDLES:
        rec = be.evaluate_bundle(
            bundle, bldg, rate, "pge",
            baseline, prices, 0.09, 15.0, 0.0, 5.0,
            ev_params, "BAAQMD", tou_w)
        total = rec["npv_pv_bat"] + rec["npv_ev"] + rec["npv_hp"]
        assert approx(total, rec["npv"], tol=1e-6), bundle


def test_capex_components_sum_to_total():
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    for bundle in be.BUNDLES:
        rec = be.evaluate_bundle(
            bundle, bldg, rate, "pge",
            baseline, prices, 0.09, 15.0, 0.0, 5.0,
            ev_params, "BAAQMD", tou_w)
        total = rec["capex_pv_bat"] + rec["capex_ev"] + rec["capex_hp"]
        assert approx(total, rec["capex_total"], tol=1e-6), bundle


def test_annual_savings_components_sum_to_total():
    bldg = _make_test_building()
    rate = _make_test_rate()
    tou_w, prices, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    for bundle in be.BUNDLES:
        rec = be.evaluate_bundle(
            bundle, bldg, rate, "pge",
            baseline, prices, 0.09, 15.0, 0.0, 5.0,
            ev_params, "BAAQMD", tou_w)
        total = (rec["bill_savings_pv_bat"]
                 + rec["gasoline_avoided"] - rec["ev_charging_cost"]
                 + rec["gas_avoided_value"] - rec["hp_elec_increase"])
        assert approx(total, rec["annual_savings"], tol=1e-6), bundle


def test_gasoline_avoided_is_rate_independent():
    """For an EV-only bundle, gasoline_avoided should not depend on rate."""
    bldg = _make_test_building()
    tou_w, _, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    rate_hi = _make_test_rate()
    rate_lo = _make_test_rate()
    rate_lo["summer_peak"] = 0.20
    rate_lo["summer_offpeak"] = 0.10
    rate_lo["winter_peak"] = 0.18
    rate_lo["winter_offpeak"] = 0.10
    prices_hi = {"summer_peak": 0.55, "summer_offpeak": 0.30,
                 "winter_peak": 0.45, "winter_offpeak": 0.28}
    prices_lo = {"summer_peak": 0.20, "summer_offpeak": 0.10,
                 "winter_peak": 0.18, "winter_offpeak": 0.10}
    rec_hi = be.evaluate_bundle("ev", bldg, rate_hi, "pge",
                                baseline, prices_hi, 0.09, 15.0, 0.0, 5.0,
                                ev_params, "BAAQMD", tou_w)
    rec_lo = be.evaluate_bundle("ev", bldg, rate_lo, "pge",
                                baseline, prices_lo, 0.09, 15.0, 0.0, 5.0,
                                ev_params, "BAAQMD", tou_w)
    assert approx(rec_hi["gasoline_avoided"], rec_lo["gasoline_avoided"])
    # But ev_charging_cost DOES change with rate
    assert rec_hi["ev_charging_cost"] > rec_lo["ev_charging_cost"]


def test_gas_avoided_is_rate_independent():
    """For an HP-only bundle, gas_avoided_value should not depend on rate."""
    bldg = _make_test_building()
    tou_w, _, ev_params = _common_eval_args()
    baseline = {k: bldg["annual_kwh"] * (w / sum(tou_w.values()))
                for k, w in tou_w.items()}
    rate_hi = _make_test_rate()
    rate_lo = _make_test_rate()
    rate_lo["winter_peak"] = 0.18
    rate_lo["winter_offpeak"] = 0.10
    rate_lo["summer_peak"] = 0.20
    rate_lo["summer_offpeak"] = 0.10
    prices_hi = {"summer_peak": 0.55, "summer_offpeak": 0.30,
                 "winter_peak": 0.45, "winter_offpeak": 0.28}
    prices_lo = {"summer_peak": 0.20, "summer_offpeak": 0.10,
                 "winter_peak": 0.18, "winter_offpeak": 0.10}
    rec_hi = be.evaluate_bundle("hp", bldg, rate_hi, "pge",
                                baseline, prices_hi, 0.09, 15.0, 0.0, 5.0,
                                ev_params, "BAAQMD", tou_w)
    rec_lo = be.evaluate_bundle("hp", bldg, rate_lo, "pge",
                                baseline, prices_lo, 0.09, 15.0, 0.0, 5.0,
                                ev_params, "BAAQMD", tou_w)
    assert approx(rec_hi["gas_avoided_value"], rec_lo["gas_avoided_value"])
    # But hp_elec_increase DOES change with rate
    assert rec_hi["hp_elec_increase"] > rec_lo["hp_elec_increase"]


def test_fuel_price_linearity_via_decompose():
    """Decompose helper: scaling gas_price by k scales gasoline_avoided
    by k, with no other column changing."""
    from src import decompose as dc
    df = pd.DataFrame({
        "bundle": ["ev"] * 3,
        "npv":              [0.0, 0.0, 0.0],
        "gasoline_avoided": [1500.0, 1500.0, 1500.0],
        "gas_avoided_value":[0.0, 0.0, 0.0],
        "ev_charging_cost": [800.0, 800.0, 800.0],
        "hp_elec_increase": [0.0, 0.0, 0.0],
        "bill_savings_pv_bat": [0.0, 0.0, 0.0],
        "capex_total": [5000.0, 5000.0, 5000.0],
    })
    out = dc.fuel_price_elasticity(
        df, gas_prices=[4.90, 9.80], therm_prices=[2.40],
        base_gas_price=4.90, base_therm_price=2.40)
    base = out[(out["gas_price"] == 4.90)]["median_npv"].iloc[0]
    doubled = out[(out["gas_price"] == 9.80)]["median_npv"].iloc[0]
    # Doubling gas price adds (1500 * 1.0 * npv_factor) to NPV
    f = dc.npv_factor()
    assert approx(doubled - base, 1500.0 * f, tol=1e-6)


def test_artifact_runnable_when_present():
    """If a smoke run wrote outputs, NPVs should be finite."""
    p = config.DATA_DIR / "bundle_economics_sdge.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p)
    nan_share = df["npv"].isna().mean()
    assert nan_share < 0.05, f"too many NaN NPVs: {nan_share:.0%}"
    assert set(df["bundle"].unique()) >= set(be.BUNDLES)


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
