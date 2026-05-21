"""Orchestrator: per (medoid x rate x bundle x subsidy regime), compute NPV.

Pipeline:
  1. Load medoids from data/representative_buildings.parquet (income_category
     and is_care already stamped per medoid).
  2. For each utility, load rate scenarios + EV-TOU + retail data + EEC.
  3. For each medoid, load hourly baseline (and Upgrade11 if HP bundles).
  4. For each bundle:
       a. Assemble post-electrification hourly load (baseline + EV + HP).
       b. For PV bundles, sweep PV (1x / 1.5x / 3x annual load) x battery
          (13.5 / 27 kWh). For each (pv_kw, batt_kwh):
            - solve battery_lp_dispatch -> signed net grid load
            - compute_annual_bill(net, scenario, ...) per rate scenario
       c. For non-PV bundles, compute_annual_bill on positive-only load.
  5. Annual savings = bill_pre - bill_post + gas_savings + gasoline_savings.
  6. NPV via payback_npv.npv over 20 years with 2% real escalator.
  7. Capex via bundles.bundle_net_capex under each subsidy regime.
  8. Write one parquet row per (medoid x rate x bundle x sizing x
     subsidy_regime).

Compute estimate: ~2,500 medoids x 40 scenarios x 8 bundles x 2 subsidy
regimes = ~1.6M rows. PV bundles add x6 sizing cells with one LP solve
each = ~2.4M LP solves total. Tractable on a server overnight; smoke
mode (--limit) processes a small subset for sanity checking.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import bill, bundles, config, payback_npv as p


# Map air districts to utilities for CC4A eligibility lookup
AIR_DISTRICT_BY_UTILITY = {"pge": "BAAQMD", "sce": "SCAQMD",
                            "sdge": "SDAPCD"}


# -----------------------------------------------------------------------------
# Rate-scenario loaders
# -----------------------------------------------------------------------------

def load_rate_scenarios(utility: str) -> pd.DataFrame:
    """Read parent's 40 designed scenarios + add per-utility EV-TOU rows.

    Designed scenarios come from rate_scenarios_<u>_fresh.csv (the
    parent rate designer's output). We rename Fixed_CARE/Fixed_NonCARE
    -> fixed_monthly_care/fixed_monthly_non_care for the bill calc.
    """
    path = config.CR_ROOT / f"rate_scenarios_{utility}_fresh.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Scenario": "scenario_id",
        "Fixed_CARE": "fixed_monthly_care",
        "Fixed_NonCARE": "fixed_monthly_non_care",
    })
    df["rate_type"] = "designed_tou"
    return df


def npv_factor(years: int = config.ANALYSIS_YEARS,
               discount: float = config.DISCOUNT_RATE_REAL,
               escalator: float = config.BILL_ESCALATOR_REAL) -> float:
    """20-yr NPV of a $1/yr stream growing at `escalator`, discounted
    at `discount`. Use this to convert year-1 annual_savings into
    20-yr NPV without rebuilding the cashflow series each cell."""
    return sum(((1 + escalator) ** (t - 1)) / ((1 + discount) ** t)
               for t in range(1, years + 1))


# -----------------------------------------------------------------------------
# Per-bundle annual savings + NPV
# -----------------------------------------------------------------------------

def gas_savings_annual(
    baseline_therms: float, utility: str, is_care: bool, bundle: str,
) -> float:
    """Annual gas-bill savings ($) when bundle includes HP.

    Full Upgrade 11 = HP + HPWH + induction → all gas removed → savings
    = baseline_therms x $/therm. CARE customers get 20% off gas in real
    billing (config.GAS_CARE_DISCOUNT), so their gas savings are
    smaller in absolute terms.
    """
    _, _, has_hp = bundles.parse_bundle(bundle)
    if not has_hp or baseline_therms <= 0:
        return 0.0
    therm_price = config.gas_price(utility)
    if is_care:
        therm_price *= (1 - config.GAS_CARE_DISCOUNT)
    return baseline_therms * therm_price


def gasoline_savings_annual(
    bundle: str, vmt: float = config.VMT_DEFAULT,
    gas_price: float = config.GAS_PRICE_DEFAULT,
    ice_mpg: float = config.ICE_MPG["default"],
) -> float:
    """Annual gasoline savings ($) when bundle includes EV. The EV's
    added electric charging cost is already in the post-bill, so this
    function only counts the ICE-side avoided cost."""
    _, has_ev, _ = bundles.parse_bundle(bundle)
    if not has_ev:
        return 0.0
    return (vmt / ice_mpg) * gas_price


def evaluate_medoid_bundle(
    medoid: pd.Series, bundle: str,
    rate_scenarios: pd.DataFrame,
    retail_data: dict, eec_hourly: np.ndarray,
    utility: str,
    baseline_hourly: np.ndarray,
    hp_delta_hourly: np.ndarray | None,
    ev_hourly: np.ndarray | None,
    pv_kw: float, batt_kwh: float,
    solar_profile: np.ndarray | None,
) -> list[dict]:
    """Inner loop: one (medoid, bundle, sizing) cell across all rates.

    Returns one row per rate scenario with bill_pre, bill_post,
    annual_savings (electric + gas + gasoline), and NPV at both
    subsidy regimes.
    """
    has_pv_bat, has_ev, has_hp = bundles.parse_bundle(bundle)
    is_care = bool(medoid.get("is_care", False))
    income = str(medoid.get("income_category", "Medium"))
    puma_str = medoid.get("puma_full")

    # Build expanded positive load (baseline + EV + HP). PV+battery
    # dispatch then handles export side per rate scenario.
    expanded_positive = bill.assemble_bundle_hourly_load(
        baseline_hourly,
        ev_load=ev_hourly if has_ev else None,
        hp_delta=hp_delta_hourly if has_hp else None,
    )

    annual_therms = float(
        medoid.get("annual_therms",
                   medoid.get("out.natural_gas.total.energy_consumption.kwh", 0)
                   / 29.3001))
    gas_save = gas_savings_annual(annual_therms, utility, is_care, bundle)
    gasoline_save = gasoline_savings_annual(bundle)

    rows = []
    for _, scenario in rate_scenarios.iterrows():
        scenario_id = scenario["scenario_id"]

        # Pre-bundle bill (no electrification, no PV)
        bill_pre = bill.compute_annual_bill(
            baseline_hourly, scenario, income, puma_str, utility,
            eec_hourly=None, retail_data=retail_data)

        # Post-bundle bill — with battery LP if PV present
        if has_pv_bat and pv_kw > 0 and batt_kwh > 0:
            rate_arr = bill.build_hourly_rate_array(scenario, utility)
            solar = solar_profile * pv_kw if solar_profile is not None else (
                bill.get_solar_per_kw(int(medoid["cec_cz"]), utility) * pv_kw)
            dispatch = bill.battery_lp_dispatch(
                hourly_load=expanded_positive,
                solar_gen=solar,
                rate_array=rate_arr,
                eec_rates=eec_hourly,
                batt_kwh=batt_kwh,
                batt_pmax_kw=batt_kwh * bill.BATTERY_C_RATE)
            if dispatch is None:
                continue  # LP infeasible; skip this cell
            net_load = dispatch["grid_in"] - dispatch["grid_out"]
        else:
            net_load = expanded_positive  # no PV/battery

        bill_post = bill.compute_annual_bill(
            net_load, scenario, income, puma_str, utility,
            eec_hourly=eec_hourly if has_pv_bat else None,
            retail_data=retail_data)

        electric_savings = bill_pre - bill_post
        annual_savings = electric_savings + gas_save + gasoline_save

        # Net capex + NPV under each subsidy regime
        npv_results = {}
        for regime in bundles.SUBSIDY_REGIMES:
            net_capex, _ = bundles.bundle_net_capex(
                bundle, pv_kw, batt_kwh, income, regime,
                air_district=AIR_DISTRICT_BY_UTILITY.get(utility))
            cashflows = p.annual_cashflow_series(
                annual_savings,
                midlife_replacement_year=(
                    config.INVERTER_REPLACEMENT_YEAR if pv_kw > 0 else None),
                midlife_replacement_cost=(
                    config.INVERTER_REPLACEMENT_COST if pv_kw > 0 else 0))
            npv_value = p.npv(cashflows, capex=net_capex)
            npv_results[f"npv_{regime}"] = npv_value
            npv_results[f"net_capex_{regime}"] = net_capex

        rows.append({
            "bldg_id": medoid.get("bldg_id"),
            "utility": utility,
            "cec_cz":  medoid.get("cec_cz"),
            "income_category": income,
            "is_care": is_care,
            "ami_frac": medoid.get("ami_frac"),
            "cluster_weight": medoid.get("cluster_weight", 1.0),
            "bundle": bundle,
            "pv_kw":   pv_kw,
            "batt_kwh": batt_kwh,
            "rate_id": scenario_id,
            "Fixed_Pct_TD":    scenario.get("Fixed_Pct_TD"),
            "Remove_Wildfire": scenario.get("Remove_Wildfire"),
            "ROE_Reduction":   scenario.get("ROE_Reduction"),
            "bill_pre":        bill_pre,
            "bill_post":       bill_post,
            "electric_savings": electric_savings,
            "gas_savings":     gas_save,
            "gasoline_savings": gasoline_save,
            "annual_savings":  annual_savings,
            "annual_therms":   annual_therms,
            **npv_results,
        })
    return rows


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run(utilities: list[str], limit: int | None = None,
        eec_multiplier: float = 1.0) -> pd.DataFrame:
    """Run the full sweep. limit=None processes all medoids; otherwise
    samples `limit` per utility for smoke testing."""
    medoids_all = pd.read_parquet(config.DATA_DIR / "representative_buildings.parquet")
    if "annual_kwh" not in medoids_all.columns:
        medoids_all["annual_kwh"] = medoids_all[
            "out.electricity.total.energy_consumption.kwh"].astype(float)
    medoids_all["annual_therms"] = (
        medoids_all["out.natural_gas.total.energy_consumption.kwh"].astype(float)
        / 29.3001)

    all_rows = []
    for u in utilities:
        u_meds = medoids_all[medoids_all["utility"].str.lower() == u].copy()
        if limit:
            u_meds = u_meds.sample(min(limit, len(u_meds)), random_state=42)
        print(f"\n{u.upper()}: {len(u_meds)} medoids")

        rate_scenarios = load_rate_scenarios(u)
        retail = bill.load_retail_data(u)
        eec = bill.load_hourly_eec(u, multiplier=eec_multiplier)

        # EV hourly profile (annual_kwh from VMT/efficiency; same for all medoids)
        ev_annual_kwh = config.VMT_DEFAULT / config.EV_EFFICIENCY["default"]
        ev_hourly = bill.ev_hourly_load(ev_annual_kwh, "smart_tou")

        t0 = time.time()
        for idx, (_, m) in enumerate(u_meds.iterrows()):
            bldg_id = int(m["bldg_id"])
            baseline_hr = bill.load_hourly_baseline_load(u, bldg_id)
            if baseline_hr is None:
                continue
            hp_delta_hr = bill.load_hourly_upgrade11_delta(u, bldg_id)
            cz = int(m["cec_cz"])
            solar_per_kw = bill.get_solar_per_kw(cz, u)

            for bundle in bundles.BUNDLES:
                if bundle == "none":
                    # Bundle "none" = baseline; bill_pre == bill_post; NPV = 0
                    continue
                has_pv_bat, _, has_hp = bundles.parse_bundle(bundle)
                if has_hp and hp_delta_hr is None:
                    continue  # can't model HP without Upgrade11 parquet

                if has_pv_bat:
                    # PV sized by EXPANDED annual load (baseline + EV + HP)
                    expanded_kwh = float(m["annual_kwh"])
                    if has_hp and hp_delta_hr is not None:
                        expanded_kwh += float(hp_delta_hr.sum())
                    _, has_ev, _ = bundles.parse_bundle(bundle)
                    if has_ev:
                        expanded_kwh += ev_annual_kwh
                    pv_grid = bundles.pv_sizing_grid(expanded_kwh)
                    batt_grid = bundles.BATTERY_SIZING_KWH
                else:
                    pv_grid = [0.0]
                    batt_grid = [0.0]

                for pv_kw in pv_grid:
                    for batt_kwh in batt_grid:
                        if (pv_kw == 0) != (batt_kwh == 0):
                            continue  # PV without battery or battery without PV: skip
                        all_rows.extend(evaluate_medoid_bundle(
                            m, bundle, rate_scenarios, retail, eec, u,
                            baseline_hr, hp_delta_hr,
                            ev_hourly if "ev" in bundle.split("_") else None,
                            pv_kw, batt_kwh,
                            solar_per_kw))

            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(u_meds) - idx - 1) / rate
                print(f"  {idx + 1}/{len(u_meds)} medoids "
                      f"({rate:.1f}/s; ~{eta / 60:.1f} min remaining)")

    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--limit", type=int, default=None,
                    help="Sample this many medoids per utility (smoke).")
    ap.add_argument("--eec-multiplier", type=float, default=1.0,
                    help="Scale NBT EEC: 1.0 = current, 1.25 / 1.50 "
                         "for CPUC-softening sensitivity.")
    ap.add_argument("--out", default=str(config.DATA_DIR / "npv_results.parquet"))
    args = ap.parse_args()

    out_path = config.assert_safe_out_dir(Path(args.out).parent) / Path(args.out).name
    df = run(args.utilities, limit=args.limit,
             eec_multiplier=args.eec_multiplier)
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
