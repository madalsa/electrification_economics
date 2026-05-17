"""Bundle-level economics: combine PV+battery+EV+heat-pump per (bldg, rate).

For each representative building x rate x bundle, compute:
  - load delta induced by the bundle (EV charging kWh + heat-pump kWh)
  - optimal (pv_kw, batt_kwh) sized to the EXPANDED load (when bundle
    contains pv_bat), so the PV-EV / PV-HP synergy is captured rather
    than just additively combining standalone NPVs
  - total capex after stacked rebates
  - annual savings (PV self-cons + battery arbitrage + EV fuel saving
    + HP gas saving - HP electric cost increase)
  - 20-yr NPV (with inverter replacement at year 13 if PV present) and
    simple payback

Bundles considered (composable):
    none           - do-nothing baseline (reference)
    pv_bat         - PV + battery only
    ev             - EV only
    hp             - HP space + HPWH + induction + panel upgrade
    pv_bat_ev      - PV + battery + EV
    pv_bat_hp      - PV + battery + HP
    ev_hp          - EV + HP (no solar)
    pv_bat_ev_hp   - full residential electrification

This module reuses:
  - sizing_optimizer.evaluate_size for PV/battery dispatch heuristic
  - vmt_sensitivity.CHARGING_PROFILES + hourly_to_tou_weights for EV load
  - upgrade11_economics.project_upgrade11_annual / _avg_*_price for HP
  - payback_npv.apply_capex_stack / npv / simple_payback

Outputs:
    data/bundle_economics_<utility>.parquet  (one row per bldg x rate x bundle)
    data/bundle_summary.csv                  (median / weighted-mean by rate x bundle)

Feeds paper figures 6 (HP bundle payback by rate, with/without PV+storage)
and 7 (optimal-rate-per-customer at zero NPV).
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (
    config,
    payback_npv as p,
    sizing_optimizer as so,
    upgrade11_economics as u11,
    vmt_sensitivity as vs,
)


BUNDLES = (
    "none",
    "pv_bat",
    "ev",
    "hp",
    "pv_bat_ev",
    "pv_bat_hp",
    "ev_hp",
    "pv_bat_ev_hp",
)


def parse_bundle(bundle: str) -> tuple[bool, bool, bool]:
    """Return (has_pv_bat, has_ev, has_hp) flags for a bundle name."""
    if bundle == "none":
        return False, False, False
    tokens = bundle.split("_")
    return ("pv" in tokens), ("ev" in tokens), ("hp" in tokens)


# -----------------------------------------------------------------------------
# Load-delta builders
# -----------------------------------------------------------------------------

def ev_kwh_by_tou(utility: str, vmt: float, ev_eff: float,
                  profile: str = "smart_tou") -> dict[str, float]:
    """Annual EV charging kWh distributed across `utility`'s TOU periods.

    Uses the same hourly-to-TOU mapping as vmt_sensitivity so EV cost and
    EV-driven load expansion are consistent.
    """
    ev_kwh = vmt / ev_eff
    hourly = vs.CHARGING_PROFILES[profile]
    weights = vs.hourly_to_tou_weights(hourly, utility)
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: ev_kwh * (w / total) for k, w in weights.items()}


def hp_kwh_by_tou(delta_hp_space: float, delta_hpwh: float,
                  delta_induction: float, tou_weights: dict[str, float]
                  ) -> dict[str, float]:
    """Distribute heat-pump electric load across TOU periods.

    Space heating is allocated only to winter periods (proportional to the
    utility's baseline winter TOU weight share). HPWH + induction follow
    the year-round baseline TOU shape.
    """
    out = {k: 0.0 for k in tou_weights}
    winter_keys = [k for k in tou_weights if k.startswith("winter")]
    winter_total = sum(tou_weights[k] for k in winter_keys)
    yearround_total = sum(tou_weights.values())
    if winter_total > 0:
        for k in winter_keys:
            out[k] += delta_hp_space * (tou_weights[k] / winter_total)
    if yearround_total > 0:
        for k in tou_weights:
            out[k] += (delta_hpwh + delta_induction) * (
                tou_weights[k] / yearround_total)
    return out


def expanded_load_by_tou(baseline: dict[str, float],
                         *deltas: dict[str, float] | None,
                         ) -> dict[str, float]:
    """Add per-period kWh deltas onto a baseline TOU load."""
    out = dict(baseline)
    for d in deltas:
        if d is None:
            continue
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


# -----------------------------------------------------------------------------
# Bundle component evaluators
# -----------------------------------------------------------------------------

def grid_search_pv_bat(
    load_by_period: dict[str, float],
    prices: dict[str, float],
    eec: float,
    fixed_monthly: float,
    demand_charge: float,
    avg_peak_kw: float,
    pv_grid: list[float] | None = None,
    batt_grid: list[float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Pick (pv_kw, batt_kwh) maximizing NPV for this load + tariff combo.

    Returns (pv_kw, batt_kwh, annual_savings, net_capex, npv). The (0, 0)
    do-nothing combo is the floor; any positive-NPV size beats it.
    """
    pv_grid = pv_grid or config.PV_KW_GRID
    batt_grid = batt_grid or config.BATT_KWH_GRID
    best_pv = best_batt = 0.0
    best_save = best_capex = 0.0
    best_npv = 0.0
    for pv_kw, batt_kwh in product(pv_grid, batt_grid):
        if pv_kw == 0 and batt_kwh == 0:
            continue
        bill_chg, export_credit = so.evaluate_size(
            pv_kw, batt_kwh, load_by_period, prices, eec,
            fixed_monthly, demand_charge, avg_peak_kw)
        annual_savings = -bill_chg + export_credit
        capex = p.CapexBreakdown(pv_kw=pv_kw, battery_kwh=batt_kwh)
        ctx = p.IncentiveContext()
        net_capex, _ = p.apply_capex_stack(capex, ctx)
        cashflows = p.annual_cashflow_series(
            annual_savings,
            midlife_replacement_year=config.INVERTER_REPLACEMENT_YEAR,
            midlife_replacement_cost=(
                config.INVERTER_REPLACEMENT_COST if pv_kw > 0 else 0))
        npv = p.npv(cashflows, capex=net_capex)
        if npv > best_npv:
            best_pv, best_batt = pv_kw, batt_kwh
            best_save, best_capex, best_npv = (
                annual_savings, net_capex, npv)
    return best_pv, best_batt, best_save, best_capex, best_npv


def ev_component_savings(
    rate_row: pd.Series, utility: str,
    vmt: float, ev_eff: float, ice_mpg: float, gas_price: float,
    profile: str = "smart_tou",
) -> float:
    eff_kwh = vs.effective_kwh_price(rate_row, profile, utility)
    if np.isnan(eff_kwh):
        return 0.0
    return p.ev_annual_fuel_savings(
        vmt=vmt, gas_price=gas_price, ice_mpg=ice_mpg,
        ev_eff_mi_per_kwh=ev_eff, rate_effective_per_kwh=eff_kwh)


def hp_component_capex_savings(
    bldg: pd.Series, rate_row: pd.Series, utility: str
) -> tuple[float, float]:
    """HP-bundle net capex (after stacked rebates) + annual savings."""
    ami_frac = bldg.get("ami_frac")
    if ami_frac is None or pd.isna(ami_frac):
        ami_frac = 1.0
    capex = p.CapexBreakdown(
        heat_pump_space=True, heat_pump_water=True,
        induction_range=True, panel_upgrade=True)
    ctx = p.IncentiveContext(income_pct_ami=float(ami_frac))
    net_capex, _ = p.apply_capex_stack(capex, ctx)

    heat_price = u11._avg_winter_price(rate_row)
    yearround = u11._avg_yearround_price(rate_row)
    if np.isnan(heat_price) or np.isnan(yearround):
        return net_capex, 0.0
    elec_increase = (
        bldg["delta_kwh_hp_space"] * heat_price
        + bldg["delta_kwh_hpwh"] * yearround
        + bldg["delta_kwh_induction"] * yearround)
    gas_savings = bldg["total_therms_displaced"] * config.gas_price(utility)
    return net_capex, gas_savings - elec_increase


# -----------------------------------------------------------------------------
# Single-bundle evaluation
# -----------------------------------------------------------------------------

def evaluate_bundle(
    bundle: str,
    bldg: pd.Series,
    rate_row: pd.Series,
    utility: str,
    baseline_load_by_tou: dict[str, float],
    prices: dict[str, float],
    eec: float,
    fixed_monthly: float,
    demand_charge: float,
    avg_peak_kw: float,
    ev_params: dict,
    air_district: str,
    tou_weights: dict[str, float],
) -> dict:
    """Return the bundle's economic columns for one (bldg, rate) cell."""
    has_pv_bat, has_ev, has_hp = parse_bundle(bundle)

    if bundle == "none":
        return {
            "bundle": bundle, "pv_kw": 0.0, "batt_kwh": 0.0,
            "capex_total": 0.0, "annual_savings": 0.0,
            "npv": 0.0, "simple_payback_yrs": float("inf"),
        }

    ev_load = (ev_kwh_by_tou(utility, ev_params["vmt"], ev_params["ev_eff"])
               if has_ev else None)
    hp_load = (hp_kwh_by_tou(
                   bldg["delta_kwh_hp_space"], bldg["delta_kwh_hpwh"],
                   bldg["delta_kwh_induction"], tou_weights)
               if has_hp else None)
    load = expanded_load_by_tou(baseline_load_by_tou, ev_load, hp_load)

    pv_kw = batt_kwh = 0.0
    capex_total = 0.0
    annual_savings = 0.0

    if has_pv_bat:
        pv_kw, batt_kwh, ann_save_pv, capex_pv, _ = grid_search_pv_bat(
            load, prices, eec, fixed_monthly, demand_charge, avg_peak_kw)
        capex_total += capex_pv
        annual_savings += ann_save_pv

    if has_ev:
        ev_premium = p.ev_net_premium(
            ev_params["scenario"], air_district=air_district)
        capex_total += max(ev_premium, 0)
        annual_savings += ev_component_savings(
            rate_row, utility,
            ev_params["vmt"], ev_params["ev_eff"],
            ev_params["ice_mpg"], ev_params["gas_price"])

    if has_hp:
        hp_capex, hp_savings = hp_component_capex_savings(
            bldg, rate_row, utility)
        capex_total += hp_capex
        annual_savings += hp_savings

    midlife_yr = (config.INVERTER_REPLACEMENT_YEAR
                  if has_pv_bat and pv_kw > 0 else None)
    midlife_cost = (config.INVERTER_REPLACEMENT_COST
                    if has_pv_bat and pv_kw > 0 else 0)
    cashflows = p.annual_cashflow_series(
        annual_savings,
        midlife_replacement_year=midlife_yr,
        midlife_replacement_cost=midlife_cost)
    npv = p.npv(cashflows, capex=capex_total)
    payback = p.simple_payback(capex_total, max(annual_savings, 0))
    return {
        "bundle": bundle, "pv_kw": pv_kw, "batt_kwh": batt_kwh,
        "capex_total": capex_total, "annual_savings": annual_savings,
        "npv": npv, "simple_payback_yrs": payback,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def build_bundles_for_utility(
    utility: str,
    buildings: pd.DataFrame,
    rates: pd.DataFrame,
    ev_scenario: str = "new_new",
    vmt: float = config.VMT_DEFAULT,
    gas_price: float = config.GAS_PRICE_DEFAULT,
    vehicle_class: str = "crossover",
    bundles: tuple[str, ...] = BUNDLES,
) -> pd.DataFrame:
    bldgs = buildings[buildings["utility"].str.lower() == utility].copy()
    bldgs = u11.project_upgrade11_annual(bldgs)
    if "annual_kwh" not in bldgs.columns:
        bldgs["annual_kwh"] = bldgs[
            "out.electricity.total.energy_consumption.kwh"].astype(float)

    tou_w = so.load_tou_weights(utility)
    eec = config.EEC_ANNUAL_AVG[utility]
    rate_rows = rates[rates["rate_type"].isin(
        ("designed_tou", "demand_charge"))]

    air_district = {"pge": "BAAQMD", "sce": "SCAQMD",
                    "sdge": "SDAPCD"}[utility]
    ev_params = {
        "vmt": vmt, "gas_price": gas_price,
        "ev_eff": config.EV_EFFICIENCY[vehicle_class],
        "ice_mpg": config.ICE_MPG[vehicle_class],
        "scenario": ev_scenario,
    }

    rows = []
    for _, b in bldgs.iterrows():
        baseline_load = so.split_annual_kwh_by_tou(b["annual_kwh"], tou_w)
        avg_peak_kw = float(b.get("summer_peak_kw") or 5.0)
        for _, r in rate_rows.iterrows():
            prices = so.get_period_prices(r, list(tou_w.keys()))
            if not prices:
                continue
            fixed_monthly = float(r.get("fixed_monthly_dollars") or 0.0)
            dc = float(r.get("demand_charge_per_kw_mo") or 0.0)
            for bundle in bundles:
                rec = evaluate_bundle(
                    bundle, b, r, utility,
                    baseline_load, prices, eec, fixed_monthly, dc,
                    avg_peak_kw, ev_params, air_district, tou_w)
                rec.update({
                    "utility": utility,
                    "bldg_id": b.get("bldg_id"),
                    "cec_cz": b.get("cec_cz"),
                    "rate_id": r["scenario_id"],
                    "rate_type": r["rate_type"],
                    "cluster_weight": b.get("cluster_weight", 1.0),
                })
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, utility: str) -> pd.DataFrame:
    """Median + cluster-weighted-mean NPV per (bundle, rate)."""
    def wmean(sub: pd.DataFrame) -> float:
        w = sub["cluster_weight"].astype(float).values
        if w.sum() <= 0:
            return float("nan")
        return float(np.average(sub["npv"].values, weights=w))

    grouped = df.groupby(["bundle", "rate_id"])
    s = grouped.agg(
        median_npv=("npv", "median"),
        median_payback=("simple_payback_yrs", "median"),
        n=("npv", "size")).reset_index()
    s["weighted_npv"] = [wmean(df[(df["bundle"] == b) & (df["rate_id"] == r)])
                         for b, r in zip(s["bundle"], s["rate_id"])]
    s["utility"] = utility
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--limit-buildings", type=int, default=0)
    ap.add_argument("--ev-scenario", default="new_new",
                    choices=list(config.EV_SCENARIOS.keys()))
    ap.add_argument("--vmt", type=float, default=config.VMT_DEFAULT)
    ap.add_argument("--gas-price", type=float,
                    default=config.GAS_PRICE_DEFAULT)
    ap.add_argument("--vehicle", default="crossover",
                    choices=list(config.EV_EFFICIENCY.keys()))
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    out_dir = config.assert_safe_out_dir(args.out_dir)
    bldgs_all = pd.read_parquet(out_dir / "representative_buildings.parquet")

    needed_gas = "out.natural_gas.heating.energy_consumption.kwh"
    if needed_gas not in bldgs_all.columns:
        meta = pd.read_parquet(
            config.CR_ROOT
            / "CA_baseline_tmy_metadata_and_annual_results.parquet")
        gas_cols = [c for c in meta.columns
                    if c.startswith("out.natural_gas.")]
        meta_subset = meta[gas_cols].reset_index(drop=False).rename(
            columns={"index": "bldg_id"})
        meta_subset["bldg_id"] = meta_subset["bldg_id"].astype("int64")
        bldgs_all = bldgs_all.merge(meta_subset, on="bldg_id", how="left")

    summaries = []
    for u in args.utilities:
        rates = pd.read_csv(out_dir / f"rate_scenarios_extended_{u}.csv")
        u_b = bldgs_all[bldgs_all["utility"].str.lower() == u]
        if args.limit_buildings:
            u_b = u_b.sample(min(args.limit_buildings, len(u_b)),
                             random_state=42)
        print(f"{u}: {len(u_b)} buildings ...")
        df = build_bundles_for_utility(
            u, u_b, rates,
            ev_scenario=args.ev_scenario, vmt=args.vmt,
            gas_price=args.gas_price, vehicle_class=args.vehicle)
        path = out_dir / f"bundle_economics_{u}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {len(df):,} rows -> {path}")
        print(f"  bundles:           {df['bundle'].nunique()}")
        print(f"  median NPV by bundle (across rates):")
        med = df.groupby("bundle")["npv"].median()
        for b, v in med.items():
            print(f"    {b:<16s}  ${v:>10,.0f}")
        summaries.append(summarize(df, u))

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "bundle_summary.csv", index=False)
    print(f"\nWrote summary -> {out_dir / 'bundle_summary.csv'}")


if __name__ == "__main__":
    main()
