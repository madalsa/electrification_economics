"""Thin reader over the parent rate-designer outputs + EE-specific extras.

The parent `rate_designer.py` (in the california_rates repo) is the
canonical source for all designed rate scenarios. It runs the
benchmarked, revenue-neutral, income-graduated rate design across
40 scenarios per utility:

    F{0,25,50,75,100} x WF{0,1} x ROE{0,0.5,1.0,1.5}  = 40

Each scenario carries tier-specific fixed charges (Fixed_CARE,
Fixed_NonCARE) in $/month, plus utility-specific TOU prices.

EE consumes these outputs VERBATIM. Nothing in this module re-derives
or approximates fixed charges, T&D shares, customer counts, or
revenue calibration - all of that is the parent rate designer's job
and has been benchmarked there.

What this module DOES do:

  1. Read the parent's rate_scenarios_<u>_fresh.csv unchanged.
  2. Rename Fixed_CARE -> fixed_monthly_care and Fixed_NonCARE ->
     fixed_monthly_non_care for downstream consistency.
  3. Append EE-specific extra rows:
       - one EV-TOU row per utility (from src.ev_tou_schedules)
       - three export-regime overlay rows (nbt_hourly, nbt_scaled_125,
         nbt_scaled_150) with eec_multiplier column for runtime sensitivity
  4. Write rate_scenarios_extended_<u>.csv to data/.

Income-graduation handling:
  fixed_monthly_care and fixed_monthly_non_care travel through as
  separate columns. bundle_economics applies the household's tier
  (CARE if ami_frac <= 0.80, else Non-CARE) at evaluation time.
  Volumetric TOU prices are uniform across tiers per the parent
  rate designer's output schema.

OUTPUT schema (rate_scenarios_extended_<utility>.csv):
    scenario_id, rate_type, source_scenario,
    Fixed_Pct_TD, Remove_Wildfire, ROE_Reduction,
    fixed_monthly_care, fixed_monthly_non_care,
    summer_peak, summer_midpeak, summer_offpeak,
    winter_peak, winter_midpeak, winter_offpeak,
    ev_super_offpeak, ev_on_peak,
    export_regime, eec_multiplier, notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src import ev_tou_schedules as evtou


# -----------------------------------------------------------------------------
# Parent reader
# -----------------------------------------------------------------------------

def load_base_scenarios(utility: str) -> pd.DataFrame:
    """Load the 40 designed scenarios for a utility from the parent."""
    path = config.CR_ROOT / f"rate_scenarios_{utility}_fresh.csv"
    return pd.read_csv(path)


def to_extended_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Re-emit parent's scenarios in the extended schema.

    Fixed_CARE and Fixed_NonCARE are renamed to fixed_monthly_care and
    fixed_monthly_non_care so downstream code is tier-aware. All TOU
    columns and scenario metadata pass through unchanged.
    """
    out = df.copy()
    out = out.rename(columns={
        "Scenario": "scenario_id",
        "Fixed_CARE": "fixed_monthly_care",
        "Fixed_NonCARE": "fixed_monthly_non_care",
    })
    out["rate_type"] = "designed_tou"
    out["source_scenario"] = out["scenario_id"]
    out["ev_super_offpeak"] = np.nan
    out["ev_on_peak"] = np.nan
    out["export_regime"] = "nbt_hourly"
    out["eec_multiplier"] = 1.0
    out["notes"] = "Designed scenario from parent rate_designer; unmodified"
    return out


# -----------------------------------------------------------------------------
# EE-specific extra rows (EV-TOU + NBT overlays)
# -----------------------------------------------------------------------------

def add_ev_only_tou_row(utility: str) -> pd.DataFrame:
    """Single EV-TOU row for one utility. Returns an empty frame if
    the utility's schedule isn't populated yet.

    Rates are blended summer + winter at day-of-year shares and weekday
    + weekend at the CA workday share, so a single average per period
    appears in the rate-extended CSV. For high-fidelity per-(season,
    day) rates, bundle_economics calls ev_tou_schedules directly.
    """
    if utility not in evtou.populated_utilities():
        return pd.DataFrame()
    sched = evtou.EV_TOU_SCHEDULES[utility]
    period_rates: dict[str, float] = {}
    total_weight = 0.0
    for key, periods in sched["schedules"].items():
        season, day_type = key.split("_")
        day_weight = (evtou.WORKDAY_SHARE if day_type == "weekday"
                      else evtou.WEEKEND_HOLIDAY_SHARE)
        if sched["season_split"]:
            s_share, w_share = evtou._summer_winter_day_shares(utility)
            season_weight = s_share if season == "summer" else w_share
        else:
            season_weight = 1.0
        w = day_weight * season_weight
        total_weight += w
        for p in periods:
            period_rates[p["name"]] = (
                period_rates.get(p["name"], 0.0) + w * p["rate"])
    if total_weight > 0:
        period_rates = {k: v / total_weight for k, v in period_rates.items()}
    rates_sorted = sorted(period_rates.items(), key=lambda kv: kv[1])
    ev_super_offpeak = rates_sorted[0][1] if rates_sorted else np.nan
    ev_on_peak = rates_sorted[-1][1] if rates_sorted else np.nan
    return pd.DataFrame([{
        "scenario_id": f"EV_TOU_{utility.upper()}",
        "rate_type": "ev_submetered_tou",
        "source_scenario": f"{utility}_{sched['tariff_name']}",
        "Fixed_Pct_TD":    np.nan,
        "Remove_Wildfire": np.nan,
        "ROE_Reduction":   np.nan,
        # EV-TOU is parallel submetered; BSC enters via base rate
        "fixed_monthly_care":      0.0,
        "fixed_monthly_non_care":  0.0,
        "summer_peak":     period_rates.get("on_peak", np.nan),
        "summer_midpeak":  period_rates.get("mid_peak", np.nan),
        "summer_offpeak":  period_rates.get("off_peak", np.nan),
        "winter_peak":     period_rates.get("on_peak", np.nan),
        "winter_midpeak":  period_rates.get("mid_peak", np.nan),
        "winter_offpeak":  period_rates.get("off_peak", np.nan),
        "ev_super_offpeak": ev_super_offpeak,
        "ev_on_peak":       ev_on_peak,
        "export_regime":    "nbt_hourly",
        "eec_multiplier":   1.0,
        "notes": (f"{sched['tariff_name']} blended workday/weekend, "
                  f"summer/winter at day-of-year shares; volumetric "
                  f"only, BSC excluded. Use ev_tou_schedules."
                  f"effective_price_under_profile() for high-"
                  f"fidelity per-(season, day) lookups."),
    }])


def add_export_regime_overlays() -> pd.DataFrame:
    """Overlay rows for NBT scaling sensitivity. Carry an eec_multiplier
    that bundle_economics --eec-multiplier applies at runtime.
    """
    rows = []
    for regime, mult, note in [
        ("nbt_hourly",      1.00,
         "Status quo NBT for new interconnections (post 4/15/2023)"),
        ("nbt_scaled_125",  1.25,
         "Sensitivity: CPUC softens NBT, hourly EEC x 1.25"),
        ("nbt_scaled_150",  1.50,
         "Sensitivity: CPUC softens NBT, hourly EEC x 1.50"),
    ]:
        rows.append({
            "scenario_id": f"EXPORT_{regime.upper()}",
            "rate_type": "export_overlay",
            "source_scenario": "any",
            "Fixed_Pct_TD":    np.nan,
            "Remove_Wildfire": np.nan,
            "ROE_Reduction":   np.nan,
            "fixed_monthly_care":     np.nan,
            "fixed_monthly_non_care": np.nan,
            "summer_peak":     np.nan, "summer_midpeak":  np.nan,
            "summer_offpeak":  np.nan,
            "winter_peak":     np.nan, "winter_midpeak":  np.nan,
            "winter_offpeak":  np.nan,
            "ev_super_offpeak": np.nan, "ev_on_peak":     np.nan,
            "export_regime":   regime,
            "eec_multiplier":  mult,
            "notes": note,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

def build_extended(utility: str) -> pd.DataFrame:
    """Combine parent's designed scenarios + this utility's EV-TOU +
    export overlays."""
    base = load_base_scenarios(utility)
    parts = [
        to_extended_schema(base),
        add_ev_only_tou_row(utility),
        add_export_regime_overlays(),
    ]
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    out_dir = config.assert_safe_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for u in args.utilities:
        df = build_extended(u)
        path = out_dir / f"rate_scenarios_extended_{u}.csv"
        df.to_csv(path, index=False)
        n_designed = (df["rate_type"] == "designed_tou").sum()
        n_ev = (df["rate_type"] == "ev_submetered_tou").sum()
        n_exp = (df["rate_type"] == "export_overlay").sum()
        print(f"{u}: {len(df)} rows  "
              f"({n_designed} designed + {n_ev} EV-TOU + {n_exp} overlay) "
              f"-> {path}")


if __name__ == "__main__":
    main()
