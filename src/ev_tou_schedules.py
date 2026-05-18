"""Per-utility EV-only submetered TOU schedules.

Opt-in tariffs that a household with a submetered EV charging circuit can
elect. Distinct from the standard residential TOU plan in two ways:
(1) different time windows (notably super off-peak windows that cover
midday solar hours), (2) different rates, often with a steep peak /
super-off-peak spread to incentivize overnight charging.

The standard `tou_weights_<u>.csv` from the parent pipeline does NOT
apply to EV-TOU because the periods don't align - using it would
misclassify EV-TOU's midday windows.

Schema per utility:
    tariff_name:        canonical name (e.g. "EV-TOU-5", "TOU-D-PRIME")
    effective_date:     ISO date of the rate sheet
    customer_class:     "non_cca_bundled" (delivery + generation from utility)
                        vs "cca_delivery_only" (CCA generation)
    rate_basis:         documents what's included; currently always
                        "volumetric_only_excludes_base_services_charge"
    season_split:       True if rates / periods differ by season
    summer_months:      list[int] of summer months; omit if season_split=False
    schedules:          dict keyed by f"{season}_{day_type}", where
                          season   in {"summer", "winter"}  if season_split else {"year"}
                          day_type in {"weekday", "weekend"}
                        Each value is a list of period dicts:
                          {"name": str, "hours": list[(start, end)], "rate": float}
    igfc_base_services_charge: structured doc on the utility's BSC under
                        AB 205 IGFC. NOT applied here - enters the bundle
                        bill via the base residential rate row.

Period naming conventions across utilities:
    super_off_peak (lowest rate, typically overnight + midday solar hours)
    off_peak       (low rate, shoulder periods)
    mid_peak       (intermediate rate, typically 4-9pm weekend or
                    weekday partial-peak windows)
    on_peak        (highest rate, typically weekday 4-9pm)

Convention: hours are half-open 24h intervals. (16, 21) = 4pm to 9pm.
Weekends + holidays share one schedule. Holidays = NERC standard 8 days
(New Year's, Presidents, Memorial, Independence, Labor, Veterans,
Thanksgiving, Christmas).

Helpers:
    validate_schedule(utility)                 -> list[str] of errors
    period_weights_for_schedule(profile, u, season, day_type) -> dict
    effective_price_under_profile(profile, u, season="annual") -> $/kWh
    season_for_month(u, m)                     -> "summer"/"winter"/"year"
    populated_utilities()                      -> list[str]
"""

from __future__ import annotations

import calendar

import numpy as np


# Workdays-per-year for blending weekday vs weekend+holiday exposure.
# 365 - 104 weekend days - 8 NERC holidays = 253 weekday-treated days.
WORKDAY_SHARE = 253.0 / 365.0
WEEKEND_HOLIDAY_SHARE = 1.0 - WORKDAY_SHARE


EV_TOU_SCHEDULES: dict[str, dict | None] = {

    # ---- SDGE EV-TOU-5 ----
    # Source: SDGE published rate plan; effective 2026-04-01.
    # Customer class: Non-CCA bundled (Electric Generation + Delivery).
    # Rate basis: volumetric only; the SDGE residential Base Services
    #   Charge (AB 205 IGFC, effective Oct 2025) is excluded. The BSC
    #   enters the household bill via the base residential rate row,
    #   NOT here, because EV-TOU is a parallel submetered tariff.
    # Year-round flat (no summer/winter differentiation as of 2026-04-01).
    # Weekday vs weekend share the same rates but have different super-off-
    # peak windows (weekend super-off-peak extends 12am-2pm; weekday is
    # 12am-6am + 10am-2pm).
    "sdge": {
        "tariff_name": "EV-TOU-5",
        "effective_date": "2026-04-01",
        "customer_class": "non_cca_bundled",
        "rate_basis": "volumetric_only_excludes_base_services_charge",
        "season_split": False,
        "schedules": {
            "year_weekday": [
                {"name": "super_off_peak",
                 "hours": [(0, 6), (10, 14)],
                 "rate": 0.121},
                {"name": "off_peak",
                 "hours": [(6, 10), (14, 16), (21, 24)],
                 "rate": 0.476},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.533},
            ],
            "year_weekend": [
                {"name": "super_off_peak",
                 "hours": [(0, 14)],
                 "rate": 0.121},
                {"name": "off_peak",
                 "hours": [(14, 16), (21, 24)],
                 "rate": 0.476},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.533},
            ],
        },
        "igfc_base_services_charge": {
            "structure": "per_month_tiered",
            "non_care_monthly_estimate": None,   # TODO: pull current BSC
            "note": ("SDGE residential Base Services Charge effective Oct "
                     "2025 under AB 205 IGFC; income-graduated tiers. "
                     "Applies to ALL SDGE residential customers regardless "
                     "of rate plan."),
        },
    },

    # ---- SCE TOU-D-PRIME ----
    # Source: SCE TOU-D-PRIME plan page (screenshot 2026 verified).
    # Customer class: Non-CCA bundled. SCE notes that CCA customers face
    #   different generation rates - out of scope.
    # Rate basis: volumetric only; SCE's $0.79/day Base Services Charge
    #   (~$24/mo non-CARE) is excluded, enters bill via base residential
    #   rate row.
    # Seasonal: Summer (Jun-Sep) vs Winter (Oct-May).
    # IMPORTANT: SCE summer has DIFFERENT RATES on weekday vs weekend.
    #   Summer weekday 4-9pm is on-peak 59c; summer weekend 4-9pm is
    #   mid-peak 40c. Winter weekday and weekend share the same schedule.
    # Winter "off-peak" 24c and "super-off-peak" 24c are the same rate
    #   but labeled differently (the labels are preserved here in case
    #   SCE diverges them in a future filing).
    "sce": {
        "tariff_name": "TOU-D-PRIME",
        "effective_date": "2026-04-01",
        "customer_class": "non_cca_bundled",
        "rate_basis": "volumetric_only_excludes_base_services_charge",
        "season_split": True,
        "summer_months": [6, 7, 8, 9],   # Jun-Sep per screenshot
        "schedules": {
            # Summer weekday: 2 periods (off-peak / on-peak)
            "summer_weekday": [
                {"name": "off_peak",
                 "hours": [(0, 16), (21, 24)],
                 "rate": 0.26},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.59},
            ],
            # Summer weekend: 2 periods (off-peak / mid-peak — note 40c,
            # not the weekday on-peak 59c)
            "summer_weekend": [
                {"name": "off_peak",
                 "hours": [(0, 16), (21, 24)],
                 "rate": 0.26},
                {"name": "mid_peak",
                 "hours": [(16, 21)],
                 "rate": 0.40},
            ],
            # Winter weekday: 3 distinct periods (off-peak / super-off /
            # mid-peak), where off-peak and super-off both 24c.
            "winter_weekday": [
                {"name": "off_peak",
                 "hours": [(0, 8), (21, 24)],
                 "rate": 0.24},
                {"name": "super_off_peak",
                 "hours": [(8, 16)],
                 "rate": 0.24},
                {"name": "mid_peak",
                 "hours": [(16, 21)],
                 "rate": 0.56},
            ],
            # Winter weekend: identical to winter weekday on SCE
            "winter_weekend": [
                {"name": "off_peak",
                 "hours": [(0, 8), (21, 24)],
                 "rate": 0.24},
                {"name": "super_off_peak",
                 "hours": [(8, 16)],
                 "rate": 0.24},
                {"name": "mid_peak",
                 "hours": [(16, 21)],
                 "rate": 0.56},
            ],
        },
        "igfc_base_services_charge": {
            "structure": "per_day_flat",
            "non_care_daily": 0.79,
            "non_care_monthly_estimate": 0.79 * 365 / 12,   # ~$24.04
            "note": ("SCE residential Base Services Charge $0.79/day "
                     "non-CARE under AB 205 IGFC; income-graduated tiers "
                     "(CARE/FERA reduction). Applies to ALL SCE residential "
                     "customers regardless of rate plan."),
        },
    },

    # ---- PGE EV2-A: pending screenshot verification ----
    "pge": None,
}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _hours_to_mask(hour_ranges: list[tuple[int, int]]) -> np.ndarray:
    """Build a 24-hour boolean mask from (start, end) half-open ranges."""
    mask = np.zeros(24, dtype=bool)
    for start, end in hour_ranges:
        mask[start:end] = True
    return mask


def _schedule_keys_for_utility(utility: str) -> list[str]:
    """Expected schedule keys for this utility based on season_split."""
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        return []
    seasons = ("summer", "winter") if s["season_split"] else ("year",)
    return [f"{season}_{day}" for season in seasons
            for day in ("weekday", "weekend")]


def _summer_winter_day_shares(utility: str) -> tuple[float, float]:
    """Return (summer_days/365, winter_days/365) for seasonal utility."""
    s = EV_TOU_SCHEDULES[utility]
    if not s["season_split"]:
        return 1.0, 0.0   # all "year"; winter share = 0
    # Use calendar to handle month-length variation; non-leap year is
    # close enough for this annual blending.
    summer_days = sum(calendar.monthrange(2025, m)[1] for m in s["summer_months"])
    return summer_days / 365.0, 1.0 - summer_days / 365.0


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_schedule(utility: str) -> list[str]:
    """Return list of validation errors for the utility's schedule.

    Empty list = schedule is OK. Catches (a) missing schedule keys,
    (b) hours that overlap between periods, (c) hours that no period
    claims, (d) season_split=True without summer_months.
    """
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        return [f"{utility}: schedule not yet populated"]
    errors: list[str] = []

    # Required top-level fields
    if s["season_split"] and "summer_months" not in s:
        errors.append(f"{utility}: season_split=True but summer_months missing")
    if s["season_split"]:
        months = s.get("summer_months", [])
        if not all(1 <= m <= 12 for m in months):
            errors.append(f"{utility}: invalid summer_months {months}")

    # Each expected schedule must exist and have hours-coverage-exactly-1
    expected_keys = _schedule_keys_for_utility(utility)
    for key in expected_keys:
        if key not in s["schedules"]:
            errors.append(f"{utility}: missing schedule['{key}']")
            continue
        coverage = np.zeros(24, dtype=int)
        for period in s["schedules"][key]:
            mask = _hours_to_mask(period["hours"])
            coverage = coverage + mask.astype(int)
            if "rate" not in period:
                errors.append(
                    f"{utility}.{key}.{period['name']}: missing 'rate'")
        if (coverage != 1).any():
            bad_hours = np.where(coverage != 1)[0].tolist()
            errors.append(
                f"{utility}.{key}: hours not claimed exactly once at "
                f"hours {bad_hours} (coverage: {coverage.tolist()})")

    return errors


# -----------------------------------------------------------------------------
# Period-weight / price helpers
# -----------------------------------------------------------------------------

def period_weights_for_schedule(
    hourly_weights: np.ndarray, utility: str,
    season: str, day_type: str,
) -> dict[str, float]:
    """Map a 24-hour weight vector to period shares for one specific
    (season, day_type) schedule.

    Returned shares sum to sum(hourly_weights). Raises KeyError if the
    utility's schedule isn't populated or the key doesn't exist (e.g.
    'summer_*' on a non-seasonal utility).
    """
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        raise KeyError(f"EV-TOU schedule for {utility} not populated yet")
    key = f"{season}_{day_type}"
    if key not in s["schedules"]:
        raise KeyError(
            f"{utility} has no schedule for {key} "
            f"(available: {sorted(s['schedules'].keys())})")
    out: dict[str, float] = {}
    for period in s["schedules"][key]:
        mask = _hours_to_mask(period["hours"])
        out[period["name"]] = float(hourly_weights[mask].sum())
    return out


def _schedule_cost(
    hourly_weights: np.ndarray, utility: str,
    season: str, day_type: str,
) -> float:
    """Sum of (period share x period rate) over one (season, day_type)."""
    weights = period_weights_for_schedule(
        hourly_weights, utility, season, day_type)
    periods = EV_TOU_SCHEDULES[utility]["schedules"][f"{season}_{day_type}"]
    return sum(weights[p["name"]] * p["rate"] for p in periods)


def effective_price_under_profile(
    hourly_weights: np.ndarray, utility: str, season: str = "annual",
) -> float:
    """Charging-profile-weighted average $/kWh under utility's EV-TOU.

    Blends weekday + weekend at the CA workday convention. For seasonal
    utilities, season="annual" blends summer and winter at their day-of-
    year shares; "summer" or "winter" returns only that season's
    weekday+weekend blend. For non-seasonal utilities, all season args
    collapse to the year-round number.
    """
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        raise KeyError(f"EV-TOU schedule for {utility} not populated yet")

    def season_cost(season_name: str) -> float:
        return (WORKDAY_SHARE * _schedule_cost(
                    hourly_weights, utility, season_name, "weekday")
                + WEEKEND_HOLIDAY_SHARE * _schedule_cost(
                    hourly_weights, utility, season_name, "weekend"))

    if s["season_split"]:
        if season == "annual":
            summer_share, winter_share = _summer_winter_day_shares(utility)
            cost = (summer_share * season_cost("summer")
                    + winter_share * season_cost("winter"))
        elif season in ("summer", "winter"):
            cost = season_cost(season)
        else:
            raise ValueError(
                f"unknown season {season!r} (expected annual/summer/winter)")
    else:
        # Non-seasonal: all season args are equivalent.
        cost = season_cost("year")

    total_weight = float(hourly_weights.sum())
    if total_weight <= 0:
        return float("nan")
    return cost / total_weight


def season_for_month(utility: str, month: int) -> str:
    """Return 'summer' / 'winter' for a seasonal utility, 'year' for a
    non-seasonal utility."""
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        raise KeyError(f"EV-TOU schedule for {utility} not populated yet")
    if not s["season_split"]:
        return "year"
    return "summer" if month in s["summer_months"] else "winter"


def populated_utilities() -> list[str]:
    """Utilities whose EV-TOU schedule is populated (non-None)."""
    return [u for u, s in EV_TOU_SCHEDULES.items() if s is not None]
