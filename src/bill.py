"""Hourly load assembly + bill calculator (replicates user's methodology).

Single source for everything that turns a (medoid, bundle, rate scenario)
combination into an annual bill in dollars. Mirrors the methodology in
the user's pge_baseline_bills.py / sce_baseline_bills.py /
sdge_baseline_bills.py — same period definitions, same baseline credit
logic, same CARE discount handling, same tier-specific fixed charges.
Adds export compensation (sum of grid exports x hourly EEC) for
bundles with PV/battery, which the user's electricity-only code
doesn't compute.

Layout:
  Section 1: time / period machinery (matches user's *_baseline_bills.py)
  Section 2: hourly load loaders (Baseline_<U>/, Upgrade11_<U>/)
  Section 3: EV hourly charging profile builder
  Section 4: signed-net-load assembly
  Section 5: battery dispatch LP + solar profile
  Section 6: retail-Excel reader (CARE discount, baseline credit, allowances)
  Section 7: hourly EEC loader (NBT export comp)
  Section 8: annual bill calculator (the wrapper)
"""

from __future__ import annotations

import calendar
import importlib
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config


# =============================================================================
# 1. Time / period machinery
# =============================================================================

# Months that count as "summer" for each utility's residential tariff.
# Source: parent <utility>_config.py build_time_arrays().
#   PGE / SCE: summer = Jun-Sep (months 6-9)   per E-TOU-C / TOU-D-4-9
#   SDGE:      summer = Jun-Oct (months 6-10)  per TOU-DR
SUMMER_MONTHS_BY_UTILITY = {"pge": (6, 9), "sce": (6, 9), "sdge": (6, 10)}

# Reference actual-tariff row in the retail Excel — used to pull
# baseline_credit_rate and care_discount that apply uniformly across
# all designed scenarios (matches user's *_baseline_bills.py convention).
REFERENCE_ACTUAL_TARIFF = {"pge": "E-TOU-C", "sce": "TOU-D-4-9",
                            "sdge": "TOU-DR"}

DAYS_PER_MONTH = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


@lru_cache(maxsize=1)
def time_arrays() -> dict:
    """Cached 8760-hour year arrays (months, hour-of-day, month boundaries)."""
    hours = np.arange(8760)
    hours_per_month = DAYS_PER_MONTH * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1
    hour_of_day = hours % 24
    return {"hours": hours, "months": months, "hour_of_day": hour_of_day,
            "month_boundaries": month_boundaries,
            "days_per_month": DAYS_PER_MONTH}


def build_period_masks(utility: str) -> dict[str, np.ndarray]:
    """Per-utility TOU period boolean masks (each 8760-array).

    Matches user's parent <utility>_config.build_*_period_masks
    line-by-line:
      PGE  (4 periods, E-TOU-C):     summer Jun-Sep / peak 16-21
      SCE  (5 periods, TOU-D-4-9):   summer Jun-Sep / peak 16-21 /
                                      winter midpeak 21-24 OR 0-8
                                      (overnight); winter offpeak is 8-16
      SDGE (6 periods, TOU-DR):      summer Jun-Oct / peak 16-21 /
                                      midpeak (6-16) OR (21-22)
    """
    ta = time_arrays()
    months, hod = ta["months"], ta["hour_of_day"]
    s_lo, s_hi = SUMMER_MONTHS_BY_UTILITY[utility]
    is_summer = (months >= s_lo) & (months <= s_hi)
    is_peak = (hod >= 16) & (hod < 21)

    if utility == "pge":
        return {"summer_peak": is_summer & is_peak,
                "summer_offpeak": is_summer & ~is_peak,
                "winter_peak": ~is_summer & is_peak,
                "winter_offpeak": ~is_summer & ~is_peak}
    if utility == "sce":
        # Winter midpeak = overnight (9pm-8am), NOT daytime.
        # Winter offpeak = 8am-4pm (the daytime window).
        wmid = (hod >= 21) | (hod < 8)
        return {"summer_peak": is_summer & is_peak,
                "summer_offpeak": is_summer & ~is_peak,
                "winter_peak": ~is_summer & is_peak,
                "winter_midpeak": ~is_summer & wmid,
                "winter_offpeak": ~is_summer & ~is_peak & ~wmid}
    if utility == "sdge":
        mid = ((hod >= 6) & (hod < 16)) | ((hod >= 21) & (hod < 22))
        return {"summer_peak": is_summer & is_peak,
                "summer_midpeak": is_summer & mid,
                "summer_offpeak": is_summer & ~is_peak & ~mid,
                "winter_peak": ~is_summer & is_peak,
                "winter_midpeak": ~is_summer & mid,
                "winter_offpeak": ~is_summer & ~is_peak & ~mid}
    raise ValueError(f"unknown utility {utility!r}")


def build_hourly_rate_array(rate_scenario: pd.Series, utility: str
                             ) -> np.ndarray:
    """Expand TOU period rates from one rate-scenario row -> 8760 $/kWh."""
    masks = build_period_masks(utility)
    arr = np.zeros(8760)
    for period, mask in masks.items():
        v = rate_scenario.get(period)
        if v is None or pd.isna(v):
            continue
        arr[mask] = float(v)
    return arr


# =============================================================================
# 2. Hourly load loaders (parent Baseline_<U>/ and Upgrade11_<U>/)
# =============================================================================

BASELINE_PARQUET_COL = "out.electricity.total.energy_consumption"


def _load_hourly_from_dir(parquet_dir, bldg_id: int) -> np.ndarray | None:
    """Load `<bldg_id>-*.parquet` from dir, sum 15-min -> 8760 hourly kWh."""
    if parquet_dir is None or not parquet_dir.exists():
        return None
    matches = list(parquet_dir.glob(f"{bldg_id}-*.parquet"))
    if not matches:
        return None
    df = pd.read_parquet(matches[0])
    if BASELINE_PARQUET_COL not in df.columns:
        return None
    return df[BASELINE_PARQUET_COL].values.reshape(-1, 4).sum(axis=1)


def load_hourly_baseline_load(utility: str, bldg_id: int
                              ) -> np.ndarray | None:
    """8760-hr baseline load from Baseline_<U>/<bldg_id>-*.parquet."""
    return _load_hourly_from_dir(
        config.utility_paths(utility)["baseline_parquets"], bldg_id)


def load_hourly_upgrade11_load(utility: str, bldg_id: int
                                ) -> np.ndarray | None:
    """8760-hr post-Upgrade-11 load from Upgrade11_<U>/<bldg_id>-*.parquet."""
    return _load_hourly_from_dir(
        config.utility_paths(utility).get("upgrade11_parquets"), bldg_id)


def load_hourly_upgrade11_delta(utility: str, bldg_id: int
                                 ) -> np.ndarray | None:
    """HP-induced load delta: upgrade11_load - baseline_load. None if
    either parquet missing."""
    upg = load_hourly_upgrade11_load(utility, bldg_id)
    base = load_hourly_baseline_load(utility, bldg_id)
    if upg is None or base is None:
        return None
    return upg - base


# =============================================================================
# 3. EV charging profile builder
# =============================================================================

# 24h weight vectors (sum to 1.0) representing daily fractional charging
# distribution. Used to spread annual EV kWh across hours of day.
CHARGING_PROFILES = {
    "overnight_offpeak": np.zeros(24),
    "opportunistic":     np.full(24, 1 / 24),
    "smart_tou":         np.zeros(24),
}
CHARGING_PROFILES["overnight_offpeak"][0:7] = 0.95 / 7
CHARGING_PROFILES["overnight_offpeak"][7:24] = 0.05 / 17
CHARGING_PROFILES["smart_tou"] = CHARGING_PROFILES["overnight_offpeak"].copy()


def ev_hourly_load(annual_ev_kwh: float,
                   profile_name: str = "smart_tou") -> np.ndarray:
    """8760-array EV charging load: tile 24h profile x 365, scale to total."""
    if profile_name not in CHARGING_PROFILES:
        raise ValueError(
            f"unknown EV charging profile: {profile_name!r}")
    profile_24h = np.asarray(CHARGING_PROFILES[profile_name], dtype=float)
    return np.tile(profile_24h, 365) * (float(annual_ev_kwh) / 365.0)


# =============================================================================
# 4. Signed-net-load assembly
# =============================================================================

def assemble_bundle_hourly_load(
    baseline_load: np.ndarray,
    ev_load: np.ndarray | None = None,
    hp_delta: np.ndarray | None = None,
    pv_gen: np.ndarray | None = None,
    battery_net: np.ndarray | None = None,
) -> np.ndarray:
    """Compose signed hourly grid load for one bundle.

        net[h] = baseline[h] + ev[h] + hp_delta[h] - pv[h] + battery_net[h]

    Sign: + = grid import, - = grid export. battery_net follows
    sizing_optimizer convention: + = battery charging from grid,
    - = battery discharging to household.
    """
    if baseline_load.shape != (8760,):
        raise ValueError(
            f"baseline shape (8760,) expected, got {baseline_load.shape}")
    net = baseline_load.astype(float, copy=True)
    for component, sign in (
        (ev_load, +1), (hp_delta, +1), (pv_gen, -1), (battery_net, +1)
    ):
        if component is None:
            continue
        if component.shape != (8760,):
            raise ValueError(f"component shape (8760,) expected")
        net = net + sign * np.asarray(component, dtype=float)
    return net


# =============================================================================
# 5. Battery dispatch LP + PV profile (extracted from sizing_optimizer_hourly)
# =============================================================================

# Battery params (match parent <utility>_battery_lp conventions).
BATTERY_ROUNDTRIP_EFF = 0.88
BATTERY_C_RATE = 0.4   # power kW = batt_kwh * C_RATE
SYNTHETIC_PV_KWH_PER_KW_YR = 1700   # CA central, NREL PVWatts


def battery_lp_dispatch(
    hourly_load: np.ndarray,
    solar_gen: np.ndarray,
    rate_array: np.ndarray,
    eec_rates: np.ndarray,
    batt_kwh: float,
    batt_pmax_kw: float,
) -> dict | None:
    """8760-hour LP for battery dispatch.

    Decision variables per hour (5T total):
      [0:T]      grid_in  (kWh imported from grid)
      [T:2T]     grid_out (kWh exported)
      [2T:3T]    batt_charge (kWh into battery)
      [3T:4T]    batt_discharge (kWh out of battery)
      [4T:5T]    SOC

    Returns dict with 'grid_in', 'grid_out', 'batt_charge',
    'batt_discharge', 'soc' arrays, OR None if LP infeasible.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix
    T = 8760
    eta = np.sqrt(BATTERY_ROUNDTRIP_EFF)
    cap = float(batt_kwh)
    pmax = float(batt_pmax_kw)
    net_load = hourly_load - solar_gen

    n = 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = rate_array
    c_obj[T:2 * T] = -eec_rates

    bounds = np.zeros((n, 2))
    bounds[0:T, 1] = np.inf
    bounds[T:2 * T, 1] = np.maximum(solar_gen, 0) + pmax
    bounds[2 * T:3 * T, 1] = pmax
    bounds[3 * T:4 * T, 1] = pmax
    bounds[4 * T:5 * T, 1] = cap

    rows, cols, vals = [], [], []
    tt = np.arange(T)
    rows.append(tt); cols.append(tt); vals.append(np.ones(T))
    rows.append(tt); cols.append(T + tt); vals.append(-np.ones(T))
    rows.append(tt); cols.append(2 * T + tt); vals.append(-np.ones(T))
    rows.append(tt); cols.append(3 * T + tt); vals.append(np.full(T, eta))
    soc_rows = T + tt
    rows.append(soc_rows); cols.append(4 * T + tt); vals.append(np.ones(T))
    rows.append(soc_rows[1:]); cols.append(4 * T + tt[:-1])
    vals.append(-np.ones(T - 1))
    rows.append(soc_rows); cols.append(2 * T + tt); vals.append(np.full(T, -eta))
    rows.append(soc_rows); cols.append(3 * T + tt); vals.append(np.ones(T))

    A_eq = csc_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * T, n))
    b_eq = np.zeros(2 * T)
    b_eq[0:T] = net_load
    b_eq[T] = cap * 0.5

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                  method="highs", options={"time_limit": 10.0, "presolve": True})
    if res.x is None:
        return None
    x = res.x
    return {"grid_in": x[0:T], "grid_out": x[T:2 * T],
            "batt_charge": x[2 * T:3 * T],
            "batt_discharge": x[3 * T:4 * T], "soc": x[4 * T:5 * T]}


_PV_CACHE: dict[tuple, np.ndarray] = {}


def get_solar_per_kw(cz: int, utility: str) -> np.ndarray:
    """8760-hour per-kW solar profile. Tries parent <utility>_solar
    module first; falls back to synthetic diurnal+seasonal shape."""
    key = (cz, utility)
    if key in _PV_CACHE:
        return _PV_CACHE[key]
    try:
        mod = importlib.import_module(f"{utility}_solar")
        if hasattr(mod, "build_per_cz_profiles"):
            profiles = mod.build_per_cz_profiles()
            if cz in profiles:
                _PV_CACHE[key] = profiles[cz]
                return profiles[cz]
    except Exception:
        pass
    # Synthetic fallback: midday-peaked diurnal x summer-peaked seasonal
    hours = pd.date_range("2025-01-01", periods=8760, freq="h")
    daylight = ((hours.hour - 12) / 6.0)
    diurnal = np.maximum(0, 1 - daylight ** 2)
    seasonal = 1 + 0.3 * np.cos((hours.dayofyear - 172) * 2 * np.pi / 365)
    raw = np.asarray(diurnal * seasonal)
    profile = raw / raw.sum() * SYNTHETIC_PV_KWH_PER_KW_YR
    _PV_CACHE[key] = profile
    return profile


# =============================================================================
# 6. Retail-Excel reader (CARE discount, baseline credit, PUMA allowances)
# =============================================================================

_RETAIL_CACHE: dict[str, dict] = {}


def load_retail_data(utility: str) -> dict:
    """Read retail Excel; cache per utility.

    Pulls care_discount + baseline_credit_rate from the reference
    actual-tariff row (E-TOU-C / TOU-D-4-9 / TOU-DR). Applied uniformly
    across all designed scenarios per the user's baseline-bills code.
    """
    if utility in _RETAIL_CACHE:
        return _RETAIL_CACHE[utility]
    path = config.CR_ROOT / f"retail_rates_data_{utility.upper()}.xlsx"
    xl = pd.ExcelFile(path)
    rates = pd.read_excel(xl, sheet_name="retail_rates_oct32025")
    rates = rates[rates["utility"].notna()]
    ref_name = REFERENCE_ACTUAL_TARIFF[utility]
    ref = rates[(rates["rate_type"] == ref_name) &
                (rates["weekday"] == "weekday")].iloc[0]

    def _safe(v):
        try:
            return float(v) if v is not None and not pd.isna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    out = {"care_discount":        abs(_safe(ref.get("care_discount"))),
           "baseline_credit_rate": _safe(ref.get("baseline_credit")),
           "baseline_df":          pd.read_excel(xl, sheet_name="baseline_puma")}
    _RETAIL_CACHE[utility] = out
    return out


def compute_baseline_credit(
    grid_in: np.ndarray, puma_str: str,
    baseline_df: pd.DataFrame, baseline_credit_rate: float,
) -> float:
    """Sum over 12 months of rate * min(monthly_kwh, monthly_allowance).

    Credit is on within-allowance IMPORTS only (PV-served kWh don't
    qualify). Matches user's baseline-bills code line-by-line.
    """
    if baseline_credit_rate == 0:
        return 0.0
    bl = baseline_df[baseline_df["puma"] == puma_str]
    if bl.empty:
        return 0.0
    daily_s = float(bl["summer_baseline_allowance"].iloc[0])
    daily_w = float(bl["winter_baseline_allowance"].iloc[0])
    ta = time_arrays()
    mb = ta["month_boundaries"]
    total = 0.0
    for m in range(12):
        s, e = mb[m], mb[m + 1]
        monthly_kwh = float(grid_in[s:e].sum())
        daily = daily_s if 6 <= (m + 1) <= 10 else daily_w
        monthly_bl = daily * DAYS_PER_MONTH[m]
        total += baseline_credit_rate * min(monthly_kwh, monthly_bl)
    return total


# =============================================================================
# 7. Hourly EEC loader (NBT export comp, with optional scaling multiplier)
# =============================================================================

EEC_TOTAL_COL = {"pge": "pge_total", "sce": "sce_total",
                  "sdge": "sdge_total"}

_EEC_CACHE: dict[str, np.ndarray] = {}


def load_hourly_eec(utility: str, multiplier: float = 1.0) -> np.ndarray:
    """8760-hr NBT export comp ($/kWh) for `utility`, scaled by `multiplier`.

    multiplier=1.0  -> current NBT (status quo)
    multiplier=1.25 -> NBT softening sensitivity (25% bump)
    multiplier=1.50 -> NBT softening sensitivity (50% bump)
    """
    if utility not in _EEC_CACHE:
        col = EEC_TOTAL_COL.get(utility)
        if col is None:
            raise ValueError(f"unknown utility {utility!r}")
        path = config.CR_ROOT / "eec_hourly_2025_wide.csv"
        df = pd.read_csv(path, usecols=[col])
        arr = df[col].values.astype(float)
        if arr.shape != (8760,):
            raise ValueError(
                f"{path.name}: {col} length {len(arr)}, expected 8760")
        _EEC_CACHE[utility] = arr
    return _EEC_CACHE[utility] * float(multiplier)


# =============================================================================
# 8. Annual bill calculator
# =============================================================================

def compute_annual_bill(
    hourly_net_load: np.ndarray,
    rate_scenario: pd.Series,
    income_category: str,
    puma_str: str,
    utility: str,
    eec_hourly: np.ndarray | None = None,
    retail_data: dict | None = None,
) -> float:
    """Annual bill ($) for one household under one rate scenario.

    Replicates user's *_baseline_bills.py methodology, extended for
    export comp:
        grid_in   = max(net_load, 0)
        grid_out  = max(-net_load, 0)
        vol_bill  = sum(grid_in * hourly_rate) - baseline_credit(grid_in)
        if is_care: vol_bill *= (1 - care_discount)
        export_credit = sum(grid_out * eec_hourly)        # if EEC given
        fixed_annual = (Fixed_CARE if is_care else Fixed_NonCARE) * 12
        total = vol_bill + fixed_annual - export_credit

    income_category: 'Low' (CARE) / 'Medium' / 'High'.
    eec_hourly: optional 8760-array for export compensation; pass None
                for non-PV bundles (no exports possible).
    """
    if retail_data is None:
        retail_data = load_retail_data(utility)
    if hourly_net_load.shape != (8760,):
        raise ValueError(
            f"net_load shape (8760,) expected, got {hourly_net_load.shape}")
    if eec_hourly is not None and eec_hourly.shape != (8760,):
        raise ValueError("eec_hourly must be (8760,)")

    is_care = (str(income_category).strip().lower() == "low")
    grid_in = np.maximum(hourly_net_load, 0.0)
    grid_out = np.maximum(-hourly_net_load, 0.0)

    rate_array = build_hourly_rate_array(rate_scenario, utility)
    vol_bill = float(np.dot(grid_in, rate_array))
    vol_bill -= compute_baseline_credit(
        grid_in, puma_str,
        retail_data["baseline_df"],
        retail_data["baseline_credit_rate"])
    if is_care and retail_data["care_discount"] > 0:
        vol_bill *= (1 - retail_data["care_discount"])

    fixed_monthly = (rate_scenario.get("fixed_monthly_care") if is_care
                     else rate_scenario.get("fixed_monthly_non_care"))
    fixed_annual = float(fixed_monthly or 0.0) * 12

    export_credit = (float(np.dot(grid_out, eec_hourly))
                     if eec_hourly is not None else 0.0)

    return vol_bill + fixed_annual - export_credit
