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

Period naming differs across utilities; period names are tariff-faithful
and NOT semantically normalized. Always look up dollar values via
rate_for_period() or effective_price_under_profile(), not by assuming a
name carries a fixed price implication.

Cross-utility period cheat sheet (2026 effective):

  PGE EV2-A — three periods, same windows every day of the year:
    off_peak     12am-3pm           22.56c summer & winter
    partial_peak 3pm-4pm + 9pm-12am 42.76c summer / 39.43c winter
    on_peak      4pm-9pm            53.81c summer / 41.10c winter
    (PGE's `partial_peak` is its tariff-specific shoulder; no other
     utility uses this exact label.)

  SCE TOU-D-PRIME — seasonal, with weekday/weekend RATE differentiation
  in summer (but not winter):
    Summer weekday: off_peak 26c (12am-4pm, 9pm-12am), on_peak 59c (4-9pm)
    Summer weekend: off_peak 26c (12am-4pm, 9pm-12am), mid_peak 40c (4-9pm)
    Winter (weekday + weekend identical):
        off_peak 24c (12am-8am, 9pm-12am),
        super_off_peak 24c (8am-4pm; numerically equal to off_peak but
                            labeled distinctly in the tariff),
        mid_peak 56c (4-9pm)
    (SCE's `mid_peak` appears in both summer weekend and winter, at
     different rates and different price-rank positions.)

  SDGE EV-TOU-5 — year-round flat, weekday vs weekend WINDOW
  differentiation (rates same):
    super_off_peak 12.1c  (CHEAPEST; weekday 12am-6am + 10am-2pm;
                           weekend 12am-2pm — covers midday solar)
    off_peak       47.6c  (MIDDLE rate — NOT the cheapest)
    on_peak        53.3c  (4-9pm daily)
    (SDGE's `off_peak` is the MIDDLE rate, not the cheapest. The
     cheapest is super_off_peak. Reading the name as "lowest rate"
     would be wrong.)

Convention: hours are half-open 24h intervals. (16, 21) = 4pm to 9pm.
Weekends + holidays share one schedule. Holidays = NERC standard 8 days
(New Year's, Presidents, Memorial, Independence, Labor, Veterans,
Thanksgiving, Christmas).

Helpers:
    validate_schedule(utility)                 -> list[str] of errors
    period_weights_for_schedule(profile, u, season, day_type) -> dict
    effective_price_under_profile(profile, u, season="annual") -> $/kWh
    rate_for_period(u, season, day_type, period_name)         -> $/kWh
    season_for_month(u, m)                     -> "summer"/"winter"/"year"
    period_names_for_utility(u)                -> sorted list of names
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
            "structure": "tiered_monthly",
            "care_monthly":     None,    # TODO: pull from SDGE BSC schedule
            "fera_monthly":     None,    # OUT OF SCOPE for paper; populate if needed
            "non_care_monthly": None,    # TODO; expected ~$24/mo per pattern
            "note": ("SDGE residential Base Services Charge effective Oct "
                     "2025 under AB 205 IGFC. Tier values not pulled yet; "
                     "expected pattern matches PGE/SCE (~$6 CARE, ~$24 "
                     "Non-CARE). FERA treated as Non-CARE for paper."),
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
            "structure": "tiered_monthly",
            "care_monthly":     None,    # TODO: pull from SCE BSC schedule
            "fera_monthly":     None,    # OUT OF SCOPE; populate if needed
            "non_care_monthly": 24.04,   # source: SCE plan page $0.79/day
            "note": ("SCE residential Base Services Charge under AB 205 "
                     "IGFC; ~$24/mo Non-CARE (source: $0.79/day flat from "
                     "EV-TOU plan page; normalized to monthly here). CARE "
                     "and FERA monthly values not pulled yet; expected "
                     "pattern matches PGE (~$6 CARE, ~$12 FERA). FERA "
                     "treated as Non-CARE for paper."),
        },
    },

    # ---- PGE EV2-A ----
    # Source: PGE EV2-A tariff sheet, Total Bundled Rates section.
    # Customer class: Non-CCA bundled (Total Bundled Rates per screenshot).
    # Rate basis: volumetric only; PGE's per-day Base Services Charge
    #   (AB 205 IGFC) is excluded, enters bill via base residential
    #   rate row.
    # Seasonal: Summer (Jun 1 - Sep 30) vs Winter (Oct 1 - May 31).
    # IMPORTANT: PGE EV2-A has NO weekday/weekend differentiation.
    #   Time-of-use windows apply every day including weekends and holidays.
    #   So summer_weekday == summer_weekend, and winter_weekday ==
    #   winter_weekend (just replicated).
    # Three periods: Off-Peak / Partial-Peak / Peak (tariff-faithful
    #   naming; "partial_peak" is PGE-specific - SDGE has no analog,
    #   SCE uses "mid_peak" with different windows).
    # Off-peak rate is identical summer and winter ($0.22558). Peak and
    #   partial-peak differ seasonally.
    "pge": {
        "tariff_name": "EV2-A",
        "effective_date": "2026-04-01",   # placeholder; verify against bill
        "customer_class": "non_cca_bundled",
        "rate_basis": "volumetric_only_excludes_base_services_charge",
        "season_split": True,
        "summer_months": [6, 7, 8, 9],   # Jun 1 - Sep 30 per tariff text
        "schedules": {
            "summer_weekday": [
                {"name": "off_peak",
                 "hours": [(0, 15)],
                 "rate": 0.22558},
                {"name": "partial_peak",
                 "hours": [(15, 16), (21, 24)],
                 "rate": 0.42760},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.53809},
            ],
            "summer_weekend": [
                # PGE: every day same as weekday
                {"name": "off_peak",
                 "hours": [(0, 15)],
                 "rate": 0.22558},
                {"name": "partial_peak",
                 "hours": [(15, 16), (21, 24)],
                 "rate": 0.42760},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.53809},
            ],
            "winter_weekday": [
                {"name": "off_peak",
                 "hours": [(0, 15)],
                 "rate": 0.22558},
                {"name": "partial_peak",
                 "hours": [(15, 16), (21, 24)],
                 "rate": 0.39428},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.41099},
            ],
            "winter_weekend": [
                {"name": "off_peak",
                 "hours": [(0, 15)],
                 "rate": 0.22558},
                {"name": "partial_peak",
                 "hours": [(15, 16), (21, 24)],
                 "rate": 0.39428},
                {"name": "on_peak",
                 "hours": [(16, 21)],
                 "rate": 0.41099},
            ],
        },
        "igfc_base_services_charge": {
            "structure": "tiered_monthly",
            "care_monthly":     6.00,    # CARE (<=200% FPL); source $0.19713/day
            "fera_monthly":     12.07,   # FERA (~200-250% FPL); OUT OF SCOPE; $0.39688/day
            "non_care_monthly": 24.13,   # Non-CARE (>250% FPL); source $0.79343/day
            "note": ("PGE Base Services Charge under AB 205 IGFC. Three "
                     "published tiers (CARE / FERA / Non-CARE). PGE bills "
                     "this per-day on the tariff; normalized to monthly "
                     "here (x365/12). This paper models only CARE vs "
                     "Non-CARE; FERA treated as Non-CARE."),
        },
    },
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


def period_names_for_utility(utility: str) -> list[str]:
    """Sorted unique period names across this utility's schedules.

    Useful for figure-grouping and for spot-checking that period naming
    conventions are tariff-faithful per utility (see module docstring).
    """
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        return []
    names: set[str] = set()
    for sched in s["schedules"].values():
        for period in sched:
            names.add(period["name"])
    return sorted(names)


def rate_for_period(
    utility: str, season: str, day_type: str, period_name: str,
) -> float:
    """Look up the $/kWh rate for one specific (season, day_type, period).

    Raises KeyError if the combination isn't defined. Use this rather
    than name-based heuristics — SDGE's 'off_peak' is the middle rate,
    not the cheapest; relying on the name to imply price rank is wrong.
    """
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        raise KeyError(f"EV-TOU schedule for {utility} not populated yet")
    key = f"{season}_{day_type}"
    if key not in s["schedules"]:
        raise KeyError(
            f"{utility} has no schedule for {key} "
            f"(available: {sorted(s['schedules'].keys())})")
    for period in s["schedules"][key]:
        if period["name"] == period_name:
            return float(period["rate"])
    raise KeyError(
        f"{utility}.{key} has no period named '{period_name}' "
        f"(available: {[p['name'] for p in s['schedules'][key]]})")
