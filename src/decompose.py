"""Post-hoc decomposition of bundle_economics outputs.

Reads data/bundle_economics_<u>.parquet and reports:

  1. Per-bundle median NPV split into pv_bat, ev, hp components, and
     into rate-dependent vs rate-independent annual cashflows. Answers:
     "where does the bundle's value come from?"

  2. Rate-design sensitivity per (bldg, bundle): the spread of total NPV
     across the 6 rate scenarios, expressed both in absolute dollars and
     as a fraction of mean NPV magnitude. Answers: "how much does the
     choice of rate move NPV for this customer?"

  3. Fuel-price elasticity, computed analytically (no re-run):
       gasoline_avoided  is linear in gas_price (gasoline $/gal)
       gas_avoided_value is linear in therm_price ($/therm)
     so scaling those columns by (alt_price / base_price) gives the NPV
     under alternative fuel prices. ev_charging_cost and hp_elec_increase
     do NOT depend on fuel prices - they're rate-driven.

Output:
  data/bundle_decomposition_<utility>.csv
  data/bundle_rate_sensitivity_<utility>.csv
  data/bundle_fuel_elasticity_<utility>.csv

CLI:
    python -m electrification_economics.src.decompose \
        [--utilities pge sce sdge]
        [--gas-prices 3.50 4.90 6.50]
        [--therm-prices 1.50 2.50 3.50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


def npv_factor(years: int = config.ANALYSIS_YEARS,
               discount: float = config.DISCOUNT_RATE_REAL,
               escalator: float = config.BILL_ESCALATOR_REAL) -> float:
    """Annuity factor: 20-yr NPV of a $1/yr stream that escalates at
    `escalator` and is discounted at `discount`. Lets us re-NPV a scaled
    year-1 cashflow without re-running the cashflow generator."""
    total = 0.0
    for t in range(1, years + 1):
        total += ((1 + escalator) ** (t - 1)) / ((1 + discount) ** t)
    return total


def per_bundle_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Median decomposition of NPV per bundle (across all bldgs & rates)."""
    grp = df.groupby("bundle")
    out = grp.agg(
        median_npv=("npv", "median"),
        median_npv_pv_bat=("npv_pv_bat", "median"),
        median_npv_ev=("npv_ev", "median"),
        median_npv_hp=("npv_hp", "median"),
        median_capex_total=("capex_total", "median"),
        median_capex_pv_bat=("capex_pv_bat", "median"),
        median_capex_ev=("capex_ev", "median"),
        median_capex_hp=("capex_hp", "median"),
        median_gasoline_avoided=("gasoline_avoided", "median"),
        median_gas_avoided_value=("gas_avoided_value", "median"),
        median_ev_charging_cost=("ev_charging_cost", "median"),
        median_hp_elec_increase=("hp_elec_increase", "median"),
        median_bill_savings_pv_bat=("bill_savings_pv_bat", "median"),
        n=("npv", "size"),
    ).reset_index()

    # Convenience: NPV-equivalent of year-1 cashflow streams
    f = npv_factor()
    out["npv_gasoline_avoided_20yr"] = out["median_gasoline_avoided"] * f
    out["npv_gas_avoided_20yr"] = out["median_gas_avoided_value"] * f
    out["npv_ev_charging_cost_20yr"] = out["median_ev_charging_cost"] * f
    out["npv_hp_elec_increase_20yr"] = out["median_hp_elec_increase"] * f
    out["npv_bill_savings_pv_bat_20yr"] = (
        out["median_bill_savings_pv_bat"] * f)

    # What fraction of NPV is fuel-cost driven (rate-independent) vs
    # electric-bill driven (rate-dependent)?
    rate_indep_pv = (out["npv_gasoline_avoided_20yr"]
                     + out["npv_gas_avoided_20yr"])
    rate_dep_pv = (out["npv_bill_savings_pv_bat_20yr"]
                   - out["npv_ev_charging_cost_20yr"]
                   - out["npv_hp_elec_increase_20yr"])
    denom = rate_indep_pv.abs() + rate_dep_pv.abs() + out["median_capex_total"]
    out["rate_indep_pv_share"] = rate_indep_pv / denom.replace(0, np.nan)
    out["rate_dep_pv_share"] = rate_dep_pv / denom.replace(0, np.nan)
    out["capex_share"] = -out["median_capex_total"] / denom.replace(0, np.nan)
    return out


def rate_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Per (bldg, bundle): NPV spread across rates. Variant: also report
    spread of just the rate-dependent components, which is the actual
    rate-design effect (rate-independent terms cancel out)."""
    f = npv_factor()
    df = df.copy()
    df["rate_dep_npv"] = (
        df["bill_savings_pv_bat"]
        - df["ev_charging_cost"]
        - df["hp_elec_increase"]
    ) * f

    grp = df.groupby(["bldg_id", "bundle"])
    out = grp.agg(
        mean_npv=("npv", "mean"),
        std_npv=("npv", "std"),
        min_npv=("npv", "min"),
        max_npv=("npv", "max"),
        spread_npv=("npv", lambda s: float(s.max() - s.min())),
        spread_rate_dep_npv=("rate_dep_npv",
                             lambda s: float(s.max() - s.min())),
        n_rates=("rate_id", "nunique"),
    ).reset_index()

    out["spread_pct_of_mean"] = (
        out["spread_npv"] / out["mean_npv"].abs().replace(0, np.nan))
    return out


def rate_sensitivity_summary(rs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-bldg rate sensitivities up to bundle level."""
    return rs.groupby("bundle").agg(
        median_spread_npv=("spread_npv", "median"),
        median_spread_rate_dep_npv=("spread_rate_dep_npv", "median"),
        median_spread_pct=("spread_pct_of_mean", "median"),
        p90_spread_npv=("spread_npv", lambda s: float(s.quantile(0.90))),
        n_buildings=("bldg_id", "nunique"),
    ).reset_index()


def fuel_price_elasticity(
    df: pd.DataFrame,
    gas_prices: list[float],
    therm_prices: list[float],
    base_gas_price: float,
    base_therm_price: float,
) -> pd.DataFrame:
    """Scale gasoline_avoided / gas_avoided_value linearly and re-NPV.

    Linearity holds because both columns are products of (rate-independent)
    fuel price and (rate-independent) consumption volume. Other operating
    cashflows (electric bill change from PV/EV/HP load) are unaffected.

    Returns a long-form table with one row per (bundle, gas_price,
    therm_price) showing median total NPV under that price pair.
    """
    f = npv_factor()
    rows = []
    for gp in gas_prices:
        for tp in therm_prices:
            gas_scale = gp / base_gas_price if base_gas_price > 0 else 0
            therm_scale = tp / base_therm_price if base_therm_price > 0 else 0
            d_gasoline_y1 = df["gasoline_avoided"] * (gas_scale - 1)
            d_gas_y1 = df["gas_avoided_value"] * (therm_scale - 1)
            adjusted_npv = df["npv"] + (d_gasoline_y1 + d_gas_y1) * f
            sub = pd.DataFrame({
                "bundle": df["bundle"], "npv_adj": adjusted_npv,
            })
            agg = sub.groupby("bundle")["npv_adj"].median().reset_index()
            agg["gas_price"] = gp
            agg["therm_price"] = tp
            agg.rename(columns={"npv_adj": "median_npv"}, inplace=True)
            rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--gas-prices", nargs="+", type=float,
                    default=list(config.GAS_PRICE_RANGE) + [
                        config.GAS_PRICE_DEFAULT])
    ap.add_argument("--therm-prices", nargs="+", type=float,
                    default=[1.50, 2.50, 3.50, 4.50])
    args = ap.parse_args()

    data_dir = config.assert_safe_out_dir(args.data_dir)

    for u in args.utilities:
        path = data_dir / f"bundle_economics_{u}.parquet"
        if not path.exists():
            print(f"{u}: {path.name} not found; skipping")
            continue
        df = pd.read_parquet(path)
        print(f"\n=== {u}: {len(df):,} rows ===")

        decomp = per_bundle_decomposition(df)
        out_path = data_dir / f"bundle_decomposition_{u}.csv"
        decomp.to_csv(out_path, index=False)
        print(f"  decomposition -> {out_path}")

        rs = rate_sensitivity(df)
        rs_sum = rate_sensitivity_summary(rs)
        out_path = data_dir / f"bundle_rate_sensitivity_{u}.csv"
        rs_sum.to_csv(out_path, index=False)
        print(f"  rate sensitivity -> {out_path}")
        print(rs_sum.to_string(index=False))

        fe = fuel_price_elasticity(
            df, gas_prices=args.gas_prices, therm_prices=args.therm_prices,
            base_gas_price=config.GAS_PRICE_DEFAULT,
            base_therm_price=config.gas_price(u))
        out_path = data_dir / f"bundle_fuel_elasticity_{u}.csv"
        fe.to_csv(out_path, index=False)
        print(f"  fuel elasticity -> {out_path}")


if __name__ == "__main__":
    main()
