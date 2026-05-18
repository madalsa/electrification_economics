"""Extend the rate-design space with rates relevant to the paper.

The existing parent pipeline produces 40 designed scenarios per utility
along three policy axes (Fixed_Pct_TD, Remove_Wildfire, ROE_Reduction).
The paper's canonical-6 subset of those is the headline rate-reform set
(see CANONICAL_8 below). For the personal-economics paper we ADD:

  EV-only TOU (separate submetered tariff):
    - EV_TOU:  super-off-peak overnight / on-peak 4-9pm
    TODO: replace single proxy with per-utility rows (PGE EV2-A,
    SCE TOU-EV-9-PRIME, SDGE EV-TOU-5) once 2026 rate sheets verified.

  Export-regime overlays (modeled as overlay, not re-priced import):
    - EXPORT_NBT_HOURLY:     default for new interconnections (CPUC NBT)
    - EXPORT_NBT_SCALED_125: NBT softened by 25% (CPUC adjustment scenario)
    - EXPORT_NBT_SCALED_150: NBT softened by 50% (more aggressive softening)

The existing 40 scenarios cover the fixed-charge / wildfire / ROE space;
we re-emit them in the extended schema but unchanged.

Removed from the default flow (kept in module for future paper / option):
  - DC_5 / DC_15 demand-charge variants. Residential DC is not currently
    on the CPUC table in CA and our pipeline doesn't model behavioral
    response to the price signal, so DC scenarios would be misleading
    as a headline rate-reform comparator. The functions
    add_demand_charge_scenarios remains in the module for a future paper
    focused on residential DC + electrification compatibility (the
    Borenstein peak-demand angle).
  - NEM 2.0 retail export, flat 5c / 15c counterfactuals. NEM 2.0 is
    grandfathered out and abstract flat rates aren't policy-relevant.

OUTPUT: rate_scenarios_extended_<utility>.csv with extended schema:
    scenario_id, rate_type, fixed_monthly_dollars,
    demand_charge_per_kw_mo, peak_window,
    summer_peak, summer_midpeak, summer_offpeak,
    winter_peak, winter_midpeak, winter_offpeak,
    ev_super_offpeak, ev_on_peak,
    export_regime, source_scenario, notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


# Approximate residential customer counts (FERC Form 1, 2024).
RESIDENTIAL_CUSTOMERS = {"pge": 5_500_000, "sce": 4_590_000, "sdge": 1_450_000}

# Average residential billing-peak kW per customer (back-of-envelope from
# CA_baseline_tmy_metadata_and_annual_results - mean of summer/winter
# peak kW across IOU population). Used to approximate DC revenue.
AVG_PEAK_KW_PER_CUSTOMER = {"pge": 5.0, "sce": 5.5, "sdge": 4.8}

# Annual residential kWh per customer (Total_Revenue / Vol_Avg / N).
# Computed in main() rather than hardcoded.

CANONICAL_8 = [
    # The paper's 8 (2 actual tariff + 6 designed). Designed subset:
    "F0_WF0_ROE0",       # status quo
    "F0_WF0_ROE1.0",     # ROE-only reduction
    "F50_WF0_ROE0",      # 50% fixed, no wildfire removal
    "F50_WF1_ROE0",      # 50% fixed + wildfire socialized
    "F100_WF0_ROE0",     # full fixed
    "F100_WF1_ROE0",     # full fixed + wildfire socialized
]
# "ACTUAL_TARIFF_*" tags injected per utility from the actual TOU schedule
# (see calculate_TOU_rates*.ipynb in parent repo).


def load_base_scenarios(utility: str) -> pd.DataFrame:
    """Load the 40 fresh designed scenarios for a utility."""
    path = config.CR_ROOT / f"rate_scenarios_{utility}_fresh.csv"
    df = pd.read_csv(path)
    return df


def to_extended_schema(df: pd.DataFrame, utility: str) -> pd.DataFrame:
    """Re-emit existing scenarios in the extended schema."""
    out = pd.DataFrame()
    out["scenario_id"] = df["Scenario"]
    out["rate_type"] = "designed_tou"
    out["source_scenario"] = df["Scenario"]
    out["fixed_monthly_dollars"] = compute_fixed_monthly(df, utility)
    out["demand_charge_per_kw_mo"] = 0.0
    out["peak_window"] = ""
    for col in ("summer_peak", "summer_midpeak", "summer_offpeak",
                "winter_peak", "winter_midpeak", "winter_offpeak"):
        out[col] = df[col] if col in df.columns else np.nan
    out["ev_super_offpeak"] = np.nan
    out["ev_on_peak"] = np.nan
    out["export_regime"] = "nbt_hourly"  # default for new adopters
    out["notes"] = "Existing designed scenario, unchanged"
    return out


def compute_fixed_monthly(df: pd.DataFrame, utility: str) -> pd.Series:
    """Convert Fixed_NonCARE (revenue-share) to $/mo per customer.

    Fixed_NonCARE in the source is the share of T&D recovered as a fixed
    charge. Here we approximate $/mo = (share x T&D revenue) / 12 / N_customers.
    Without separate T&D vs generation split readily available, we use
    Fixed_NonCARE x Total_Revenue x ~0.45 (T&D fraction) / 12 / N as a
    rough conversion; refined later when we wire to the bill simulator.
    """
    n_cust = RESIDENTIAL_CUSTOMERS[utility]
    td_share = 0.45  # rough; revisit when wiring to bill simulator
    fixed_dollars = (
        df["Fixed_NonCARE"] * df["Total_Revenue"] * td_share / 12 / n_cust)
    return fixed_dollars.round(2)


def add_demand_charge_scenarios(
    base: pd.DataFrame, utility: str
) -> pd.DataFrame:
    """Build DC_5, DC_15 scenarios: $/kW-mo on monthly billing peak,
    with volumetric rates reduced to keep total revenue neutral.
    """
    ref = base[base["Scenario"] == "F0_WF0_ROE0"].iloc[0]
    avg_peak_kw = AVG_PEAK_KW_PER_CUSTOMER[utility]
    n_cust = RESIDENTIAL_CUSTOMERS[utility]
    annual_kwh_total = ref["Total_Revenue"] / ref["Vol_Avg"]

    rows = []
    for dc, label in [(5.0, "DC_5"), (15.0, "DC_15")]:
        # Annual DC revenue = customers * peak_kw * dc * 12
        dc_rev = n_cust * avg_peak_kw * dc * 12
        # Remaining revenue must come from volumetric
        vol_rev = ref["Total_Revenue"] - dc_rev
        new_vol_avg = vol_rev / annual_kwh_total
        # Scale TOU prices proportionally
        scale = new_vol_avg / ref["Vol_Avg"]
        row = {
            "scenario_id": label,
            "rate_type": "demand_charge",
            "source_scenario": "F0_WF0_ROE0",
            "fixed_monthly_dollars": 0.0,
            "demand_charge_per_kw_mo": dc,
            "peak_window": "monthly_max",
            "summer_peak":     ref.get("summer_peak", np.nan) * scale,
            "summer_midpeak":  ref.get("summer_midpeak", np.nan) * scale,
            "summer_offpeak":  ref.get("summer_offpeak", np.nan) * scale,
            "winter_peak":     ref.get("winter_peak", np.nan) * scale,
            "winter_midpeak":  ref.get("winter_midpeak", np.nan) * scale,
            "winter_offpeak":  ref.get("winter_offpeak", np.nan) * scale,
            "ev_super_offpeak": np.nan,
            "ev_on_peak":       np.nan,
            "export_regime":   "nbt_hourly",
            "notes": (f"DC = ${dc}/kW-mo; volumetric scaled "
                      f"x{scale:.3f} for revenue neutrality"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_ev_only_tou_scenario() -> pd.DataFrame:
    """EV-only TOU: applies to submetered EV load only.

    Standard CA EV-TOU example: PGE EV2-A summer rates ~$0.18 super
    off-peak (00:00-15:00), $0.55 peak (16:00-21:00), $0.36 off-peak
    elsewhere. Use as proxy across utilities; refine per utility filings.
    """
    return pd.DataFrame([{
        "scenario_id": "EV_TOU",
        "rate_type": "ev_submetered_tou",
        "source_scenario": "PGE_EV2A_proxy",
        "fixed_monthly_dollars": 0.0,
        "demand_charge_per_kw_mo": 0.0,
        "peak_window": "16:00-21:00",
        "summer_peak":     0.55,
        "summer_midpeak":  0.36,
        "summer_offpeak":  0.18,
        "winter_peak":     0.49,
        "winter_midpeak":  0.34,
        "winter_offpeak":  0.18,
        "ev_super_offpeak": 0.18,
        "ev_on_peak":       0.55,
        "export_regime":   "nbt_hourly",
        "notes": ("EV submetered tariff. Applies to EV load only; rest of "
                  "household billed on base rate."),
    }])


def add_export_regime_scenarios() -> pd.DataFrame:
    """Export regimes are overlays on top of any import tariff.

    Scaled-NBT variants represent CPUC-softening counterfactuals: the
    base hourly EEC values are scaled by the regime's multiplier when
    bundle_economics is run with --eec-multiplier (default 1.0, picks
    up nbt_hourly). Downstream code reads the multiplier directly;
    these rows document the regime names and serve as labels for
    figure-grouping. NEM 2.0 retail / flat 5c / flat 15c removed in
    May 2026: NEM 2.0 is grandfathered out (not a forward-looking
    regime), and flat rates aren't policy-relevant.
    """
    rows = []
    for regime, mult, note in [
        ("nbt_hourly",      1.00,
         "Default for new interconnections (post 4/15/2023); EEC hourly"),
        ("nbt_scaled_125",  1.25,
         "CPUC softening sensitivity: hourly EEC x 1.25"),
        ("nbt_scaled_150",  1.50,
         "CPUC softening sensitivity: hourly EEC x 1.50"),
    ]:
        rows.append({
            "scenario_id": f"EXPORT_{regime.upper()}",
            "rate_type": "export_overlay",
            "source_scenario": "any",
            "fixed_monthly_dollars": np.nan,
            "demand_charge_per_kw_mo": np.nan,
            "peak_window": "",
            "summer_peak": np.nan, "summer_midpeak": np.nan,
            "summer_offpeak": np.nan,
            "winter_peak": np.nan, "winter_midpeak": np.nan,
            "winter_offpeak": np.nan,
            "ev_super_offpeak": np.nan, "ev_on_peak": np.nan,
            "export_regime": regime,
            "eec_multiplier": mult,
            "notes": note,
        })
    return pd.DataFrame(rows)


def build_extended(utility: str, include_demand_charges: bool = False
                   ) -> pd.DataFrame:
    """Build the extended rate-scenarios table.

    include_demand_charges defaults False: DC_5 / DC_15 are parked for a
    follow-up paper focused on residential demand charges + electrification
    compatibility. Set True to include them (e.g. for ad-hoc sensitivity).
    """
    base = load_base_scenarios(utility)
    parts = [
        to_extended_schema(base, utility),
        add_ev_only_tou_scenario(),
        add_export_regime_scenarios(),
    ]
    if include_demand_charges:
        parts.insert(1, add_demand_charge_scenarios(base, utility))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    ap.add_argument("--include-demand-charges", action="store_true",
                    help="Include DC_5 / DC_15 hypothetical residential "
                         "demand-charge variants. Off by default (parked "
                         "for follow-up paper).")
    args = ap.parse_args()

    out_dir = config.assert_safe_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for u in args.utilities:
        df = build_extended(
            u, include_demand_charges=args.include_demand_charges)
        path = out_dir / f"rate_scenarios_extended_{u}.csv"
        df.to_csv(path, index=False)
        n_designed = (df["rate_type"] == "designed_tou").sum()
        n_dc = (df["rate_type"] == "demand_charge").sum()
        n_ev = (df["rate_type"] == "ev_submetered_tou").sum()
        n_exp = (df["rate_type"] == "export_overlay").sum()
        print(f"{u}: {len(df)} rows  "
              f"({n_designed} designed + {n_dc} DC + {n_ev} EV-TOU "
              f"+ {n_exp} export overlay) -> {path}")


if __name__ == "__main__":
    main()
