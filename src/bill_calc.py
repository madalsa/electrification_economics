"""Annual-bill calculator (Path 1 of the architectural plan).

Replicates the methodology in the user's pge_baseline_bills.py /
sce_baseline_bills.py / sdge_baseline_bills.py line-by-line, so EE's
bundle_economics can compute pre- and post-electrification bills
consistently with the user's benchmarked baseline-bills code.

Bill formula (matches pge_baseline_bills.py lines 382-415):
  1. vol_bill = sum_hourly(load[h] * hourly_rate_array[h])
  2. baseline_credit = sum_months(
         baseline_credit_rate
         * min(monthly_kwh, daily_allowance * days_in_month))
  3. vol_bill -= baseline_credit
  4. if is_care: vol_bill *= (1 - care_discount)
  5. fixed_annual = (fixed_monthly_care if is_care
                     else fixed_monthly_non_care) * 12
  6. total_bill = vol_bill + fixed_annual

Per-utility period definitions match the user's code:
  PGE  (4 periods): summer 6-10, peak 16-21
  SDGE (6 periods): summer 6-10, peak 16-21, midpeak (6-16) | (21-22)
  SCE  (5 periods): summer 6-10, peak 16-21, winter midpeak 8-16
    [BEST GUESS: sce_config.build_sce_period_masks is not in this repo;
     verify against the parent. Inferred from rate_scenarios_sce_fresh
     having winter_midpeak only (no summer_midpeak) and from TOU-D-4-9
     actual tariff windows.]

CARE eligibility proxy: income_category == 'Low' (i.e., in.income < $50K
per the mapping in representative_buildings.INCOME_TO_CATEGORY).

The CARE volumetric discount factor and baseline credit rate are read
ONCE per utility from the corresponding actual-tariff row in the retail
rates Excel (E-TOU-C for PGE, TOU-D-4-9 for SCE, TOU-DR for SDGE), and
applied UNIFORMLY across all designed-scenario rates. This matches the
user's baseline-bills convention (Option B from the prior brainstorm:
CARE discount applies to all rate scenarios).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# Months that count as "summer" for each utility's residential tariff.
# Matches the user's baseline-bills code which uses months >= 6 & <= 10
# (Jun-Oct) consistently across utilities.
SUMMER_MONTHS_BY_UTILITY = {
    "pge": (6, 10),
    "sce": (6, 10),
    "sdge": (6, 10),
}

# Reference actual-tariff row in the retail Excel from which the
# baseline_credit_rate and care_discount are pulled (then applied
# uniformly across all designed scenarios per the user's baseline-bills
# convention).
REFERENCE_ACTUAL_TARIFF = {
    "pge":  "E-TOU-C",
    "sce":  "TOU-D-4-9",
    "sdge": "TOU-DR",
}

# Days-per-month for non-leap year, matching the user's bill code.
DAYS_PER_MONTH = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# -----------------------------------------------------------------------------
# 8760-hour time arrays (cached; identical across utilities)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def time_arrays() -> dict:
    """Return cached month / hour-of-day arrays for the 8760-hour year."""
    hours = np.arange(8760)
    hours_per_month = DAYS_PER_MONTH * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1
    hour_of_day = hours % 24
    return {
        "hours": hours,
        "months": months,
        "hour_of_day": hour_of_day,
        "month_boundaries": month_boundaries,
        "days_per_month": DAYS_PER_MONTH,
    }


# -----------------------------------------------------------------------------
# Period masks (utility-specific)
# -----------------------------------------------------------------------------

def build_period_masks(utility: str) -> dict[str, np.ndarray]:
    """Return dict[period_name, 8760-bool-mask] for `utility`.

    Periods match the user's baseline-bills code:
      PGE:  summer_peak, summer_offpeak, winter_peak, winter_offpeak (4)
      SCE:  summer_peak, summer_offpeak,
            winter_peak, winter_midpeak, winter_offpeak (5)
      SDGE: summer_peak, summer_midpeak, summer_offpeak,
            winter_peak, winter_midpeak, winter_offpeak (6)
    """
    ta = time_arrays()
    months = ta["months"]
    hod = ta["hour_of_day"]

    s_lo, s_hi = SUMMER_MONTHS_BY_UTILITY[utility]
    is_summer = (months >= s_lo) & (months <= s_hi)
    is_peak = (hod >= 16) & (hod < 21)   # 4-9pm, all utilities

    if utility == "pge":
        return {
            "summer_peak":    is_summer & is_peak,
            "summer_offpeak": is_summer & ~is_peak,
            "winter_peak":    ~is_summer & is_peak,
            "winter_offpeak": ~is_summer & ~is_peak,
        }
    if utility == "sce":
        # Best guess: SCE TOU-D-4-9 winter mid-peak is 8am-4pm; summer
        # has no mid-peak. Verify against sce_config.build_sce_period_masks
        # in the parent repo.
        is_winter_midpeak = (hod >= 8) & (hod < 16)
        return {
            "summer_peak":    is_summer & is_peak,
            "summer_offpeak": is_summer & ~is_peak,
            "winter_peak":    ~is_summer & is_peak,
            "winter_midpeak": ~is_summer & is_winter_midpeak,
            "winter_offpeak": ~is_summer & ~is_peak & ~is_winter_midpeak,
        }
    if utility == "sdge":
        # SDGE midpeak (both seasons): 6am-4pm OR 9pm-10pm.
        # Matches sdge_baseline_bills.py:218-219.
        is_midpeak = ((hod >= 6) & (hod < 16)) | ((hod >= 21) & (hod < 22))
        return {
            "summer_peak":    is_summer & is_peak,
            "summer_midpeak": is_summer & is_midpeak,
            "summer_offpeak": is_summer & ~is_peak & ~is_midpeak,
            "winter_peak":    ~is_summer & is_peak,
            "winter_midpeak": ~is_summer & is_midpeak,
            "winter_offpeak": ~is_summer & ~is_peak & ~is_midpeak,
        }
    raise ValueError(f"unknown utility {utility!r}")


def build_hourly_rate_array(rate_scenario: pd.Series, utility: str
                            ) -> np.ndarray:
    """Expand TOU period rates from one rate-scenario row into an 8760-
    hour $/kWh array."""
    masks = build_period_masks(utility)
    arr = np.zeros(8760)
    for period, mask in masks.items():
        v = rate_scenario.get(period)
        if v is None or pd.isna(v):
            continue
        arr[mask] = float(v)
    return arr


# -----------------------------------------------------------------------------
# Retail-Excel reader (cached per utility)
# -----------------------------------------------------------------------------

_RETAIL_CACHE: dict[str, dict] = {}


def load_retail_data(utility: str) -> dict:
    """Load and cache retail-rate metadata for `utility`.

    Returns dict with:
      care_discount         : float (volumetric discount factor, e.g. 0.35)
      baseline_credit_rate  : float ($/kWh credit on within-baseline kWh)
      baseline_df           : DataFrame (puma, summer/winter_baseline_allowance)
    """
    if utility in _RETAIL_CACHE:
        return _RETAIL_CACHE[utility]

    path = config.CR_ROOT / f"retail_rates_data_{utility.upper()}.xlsx"
    xl = pd.ExcelFile(path)
    rates_df = pd.read_excel(xl, sheet_name="retail_rates_oct32025")
    rates_df = rates_df[rates_df["utility"].notna()]

    # Pull baseline_credit_rate + care_discount from the reference
    # actual-tariff row (weekday). User's convention.
    ref_name = REFERENCE_ACTUAL_TARIFF[utility]
    ref_row = rates_df[
        (rates_df["rate_type"] == ref_name)
        & (rates_df["weekday"] == "weekday")
    ].iloc[0]

    def _safe(v):
        try:
            return float(v) if v is not None and not pd.isna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    out = {
        "care_discount":        abs(_safe(ref_row.get("care_discount"))),
        "baseline_credit_rate": _safe(ref_row.get("baseline_credit")),
        "baseline_df":          pd.read_excel(xl, sheet_name="baseline_puma"),
    }
    _RETAIL_CACHE[utility] = out
    return out


# -----------------------------------------------------------------------------
# Baseline credit (PUMA-specific monthly allowance cap)
# -----------------------------------------------------------------------------

def compute_baseline_credit(
    hourly_load: np.ndarray,
    puma_str: str,
    baseline_df: pd.DataFrame,
    baseline_credit_rate: float,
) -> float:
    """Sum over 12 months of (rate * min(monthly_kwh, monthly_allowance)).

    Returns 0.0 if PUMA has no entry in baseline_df. Matches the user's
    baseline-bills code line-by-line.
    """
    if baseline_credit_rate == 0:
        return 0.0
    bl_entry = baseline_df[baseline_df["puma"] == puma_str]
    if bl_entry.empty:
        return 0.0
    daily_summer = float(bl_entry["summer_baseline_allowance"].iloc[0])
    daily_winter = float(bl_entry["winter_baseline_allowance"].iloc[0])

    ta = time_arrays()
    month_boundaries = ta["month_boundaries"]
    total = 0.0
    for m in range(12):
        s, e = month_boundaries[m], month_boundaries[m + 1]
        monthly_kwh = float(hourly_load[s:e].sum())
        if 6 <= (m + 1) <= 10:
            monthly_bl = daily_summer * DAYS_PER_MONTH[m]
        else:
            monthly_bl = daily_winter * DAYS_PER_MONTH[m]
        total += baseline_credit_rate * min(monthly_kwh, monthly_bl)
    return total


# -----------------------------------------------------------------------------
# Hourly load loader (reads parent Baseline_<U>/ parquets)
# -----------------------------------------------------------------------------

BASELINE_PARQUET_COL = "out.electricity.total.energy_consumption"


def load_hourly_baseline_load(utility: str, bldg_id: int
                              ) -> np.ndarray | None:
    """Load 8,760-hr baseline load profile from Baseline_<U>/.

    Convention from parent pipeline: parquets named
    `<bldg_id>-<suffix>.parquet`, with the user's *_baseline_bills.py
    using `<bldg_id>-0.parquet`. Column
    `out.electricity.total.energy_consumption` is 15-min interval (35,040
    rows); summed every 4 rows to yield 8,760 hourly kWh values.

    Returns None if (a) the Baseline_<U>/ directory isn't present (lets
    callers skip gracefully when running EE without the ~21 GB of parent
    hourly data), (b) no parquet matches the bldg_id, or (c) the load
    column is missing from the parquet.
    """
    return _load_hourly_from_dir(
        config.utility_paths(utility)["baseline_parquets"], bldg_id)


def load_hourly_upgrade11_load(utility: str, bldg_id: int
                                ) -> np.ndarray | None:
    """Load 8,760-hr POST-upgrade load profile from Upgrade11_<U>/.

    Upgrade 11 = whole-home electrification (heat pump space heat +
    HPWH + induction range; gas heating/HW/cooking removed). Same
    parquet convention as the baseline; 15-min → hourly aggregation.

    Returns None on the same three failure modes as
    load_hourly_baseline_load.
    """
    return _load_hourly_from_dir(
        config.utility_paths(utility).get("upgrade11_parquets"), bldg_id)


def load_hourly_upgrade11_delta(utility: str, bldg_id: int
                                 ) -> np.ndarray | None:
    """Return 8,760-array of HP-induced hourly LOAD DELTA in kWh.

        delta[h] = upgrade11_load[h] - baseline_load[h]

    Positive at hours where Upgrade 11 adds net electric load (most
    winter hours, year-round HPWH operation). Negative at hours where
    Upgrade 11 reduces load (uncommon but possible e.g. when efficiency
    improvements outweigh the HP/HPWH additions). Composable: a bundle's
    expanded post-electrification load is
        baseline_load + (ev_delta if EV) + (hp_delta if HP) - pv_gen
        + battery_dispatch.

    Returns None if either the baseline OR the upgrade11 parquet for
    this (utility, bldg_id) can't be loaded.
    """
    upg = load_hourly_upgrade11_load(utility, bldg_id)
    base = load_hourly_baseline_load(utility, bldg_id)
    if upg is None or base is None:
        return None
    return upg - base


def _load_hourly_from_dir(parquet_dir, bldg_id: int) -> np.ndarray | None:
    """Internal: load `<bldg_id>-*.parquet` from `parquet_dir` and
    aggregate the 15-min total energy column to 8,760 hourly kWh."""
    if parquet_dir is None or not parquet_dir.exists():
        return None
    matches = list(parquet_dir.glob(f"{bldg_id}-*.parquet"))
    if not matches:
        return None
    df = pd.read_parquet(matches[0])
    if BASELINE_PARQUET_COL not in df.columns:
        return None
    load_15min = df[BASELINE_PARQUET_COL].values
    return load_15min.reshape(-1, 4).sum(axis=1)


# -----------------------------------------------------------------------------
# Annual bill calculator
# -----------------------------------------------------------------------------

def compute_annual_bill(
    hourly_load: np.ndarray,
    rate_scenario: pd.Series,
    income_category: str,
    puma_str: str,
    utility: str,
    retail_data: dict | None = None,
) -> float:
    """Annual bill in $ for one household under one rate scenario.

    hourly_load    : shape (8760,) kWh per hour
    rate_scenario  : row from rate_scenarios_extended_<u>.csv with period
                     prices and fixed_monthly_care / fixed_monthly_non_care
    income_category: 'Low' / 'Medium' / 'High' (CARE if 'Low')
    puma_str       : G06000xxx-style PUMA string (matches baseline_df['puma'])
    utility        : 'pge' / 'sce' / 'sdge'
    retail_data    : output of load_retail_data(utility); auto-loaded if None

    Returns total annual $ bill (volumetric + fixed, after baseline
    credit and CARE discount).
    """
    if retail_data is None:
        retail_data = load_retail_data(utility)
    if hourly_load.shape != (8760,):
        raise ValueError(
            f"hourly_load must be shape (8760,), got {hourly_load.shape}")

    is_care = (str(income_category).strip().lower() == "low")

    # 1. Volumetric energy charges
    rate_array = build_hourly_rate_array(rate_scenario, utility)
    vol_bill = float(np.dot(hourly_load, rate_array))

    # 2. Baseline credit on within-allowance kWh
    bl_credit = compute_baseline_credit(
        hourly_load, puma_str,
        retail_data["baseline_df"],
        retail_data["baseline_credit_rate"])
    vol_bill -= bl_credit

    # 3. CARE volumetric discount applied to vol_bill (after credit)
    if is_care and retail_data["care_discount"] > 0:
        vol_bill *= (1 - retail_data["care_discount"])

    # 4. Tier-specific fixed charge (already $/month from parent rate
    # designer; multiply by 12 for annual)
    if is_care:
        fixed_monthly = rate_scenario.get("fixed_monthly_care")
    else:
        fixed_monthly = rate_scenario.get("fixed_monthly_non_care")
    fixed_annual = float(fixed_monthly or 0.0) * 12

    return vol_bill + fixed_annual
