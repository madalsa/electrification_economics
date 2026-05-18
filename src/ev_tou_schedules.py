"""Per-utility EV-only submetered TOU schedules.

Opt-in tariffs that a household with a submetered EV charging circuit
can elect. Distinct from the standard residential TOU plan in two
ways: (1) different time windows (especially super off-peak that
covers midday solar hours), (2) different rates, often with a very
steep peak / super-off-peak spread to incentivize overnight charging.

Each utility's schedule has its own period definitions. The standard
`tou_weights_<u>.csv` from the parent pipeline does NOT apply to
EV-TOU because the periods don't align — using it would misclassify
EV-TOU's 10am-2pm weekday super off-peak window as midpeak.

Schema per utility:
    tariff_name:    canonical name (e.g. "EV-TOU-5")
    effective_date: ISO date of the rate sheet used
    season_split:   False if year-round; True if summer/winter differ
                    (in which case each period has summer_rate and
                    winter_rate instead of a single rate_per_kwh)
    periods:        list of dicts, each with:
                      name:           "super_off_peak" / "off_peak" / "on_peak"
                      weekday_hours:  list of (start, end) half-open hour
                                      ranges; e.g. (0, 6) means 12am-6am
                      weekend_hours:  same; weekends + holidays
                      rate_per_kwh:   $/kWh (year-round)
                                       OR summer_rate / winter_rate (seasonal)

Conventions:
  - Hours use 24h half-open intervals: (16, 21) = 4pm to 9pm
  - Weekends and holidays share one schedule (PGE / SCE / SDGE convention)
  - Holidays: New Year's, Presidents Day, Memorial, Independence,
    Labor, Veterans, Thanksgiving, Christmas (NERC plus state observance)

Helper functions:
  period_weights_from_hourly(hourly_weights, utility)
      -> dict[period_name, share]
  effective_price_under_profile(hourly_weights, utility, season)
      -> $/kWh
"""

from __future__ import annotations

import numpy as np


# Workdays-per-year for blending weekday vs weekend+holiday exposure.
# CA convention (matches NERC): 8 holidays per year. Even when a holiday
# lands on a weekend it's still listed; double-counting is acceptable
# for the blended-exposure calculation. 365 - 104 weekend days - 8
# holidays = 253 weekday-treated days; treat 112 days as
# weekend-or-holiday for tariff bill purposes.
WORKDAY_SHARE = 253.0 / 365.0
WEEKEND_HOLIDAY_SHARE = 1.0 - WORKDAY_SHARE


EV_TOU_SCHEDULES: dict[str, dict | None] = {
    # ---- SDGE EV-TOU-5 ----
    # Source: SDGE published rate plan page; rates effective 2026-04-01.
    # Customer class: Non-CCA (full bundled service — Electric Generation
    #   and Delivery). CCA customers (Clean Energy Alliance, San Diego
    #   Community Power, etc.) face different generation rates; not modeled
    #   in the default scenario set.
    # Rates are VOLUMETRIC ONLY ($/kWh). They exclude the SDGE residential
    #   Base Services Charge (the AB 205 IGFC implementation, effective
    #   October 2025). The Base Services Charge enters the bundle bill
    #   calculation via the canonical-6 designed-TOU rates'
    #   `fixed_monthly_dollars` field — NOT here, because EV-TOU is a
    #   parallel submetered tariff and the household pays the BSC once via
    #   its base residential rate.
    # Year-round flat (no summer/winter differentiation as of 2026-04-01).
    # Weekday vs weekend share the same rates but have different time
    # windows (weekend super-off-peak extends 12am-2pm; weekday super-off-
    # peak is 12am-6am + 10am-2pm).
    # Holidays observed: New Year's, President's, Memorial, Independence,
    # Labor, Veterans, Thanksgiving, Christmas (8 days).
    "sdge": {
        "tariff_name": "EV-TOU-5",
        "effective_date": "2026-04-01",
        "customer_class": "non_cca_bundled",
        "rate_basis": "volumetric_only_excludes_base_services_charge",
        "season_split": False,
        "periods": [
            {
                "name": "super_off_peak",
                "weekday_hours": [(0, 6), (10, 14)],   # 12a-6a, 10a-2p
                "weekend_hours": [(0, 14)],             # 12a-2p
                "rate_per_kwh": 0.121,
            },
            {
                "name": "off_peak",
                "weekday_hours": [(6, 10), (14, 16), (21, 24)],
                "weekend_hours": [(14, 16), (21, 24)],
                "rate_per_kwh": 0.476,
            },
            {
                "name": "on_peak",
                "weekday_hours": [(16, 21)],
                "weekend_hours": [(16, 21)],
                "rate_per_kwh": 0.533,
            },
        ],
    },
    # ---- PGE EV2-A: pending screenshot verification ----
    "pge": None,
    # ---- SCE TOU-EV-9-PRIME (or successor): pending screenshot verification ----
    "sce": None,
}


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _hours_to_mask(hour_ranges: list[tuple[int, int]]) -> np.ndarray:
    """Build a 24-hour boolean mask from (start, end) half-open ranges."""
    mask = np.zeros(24, dtype=bool)
    for start, end in hour_ranges:
        mask[start:end] = True
    return mask


def validate_schedule(utility: str) -> list[str]:
    """Return list of validation errors for the utility's schedule.

    Empty list = schedule is OK. Catches (a) hours that overlap between
    periods, (b) hours that no period claims, (c) inconsistent
    rate-field naming.
    """
    s = EV_TOU_SCHEDULES.get(utility)
    if s is None:
        return [f"{utility}: schedule not yet populated"]
    errors: list[str] = []

    # Weekday + weekend coverage must each be all-24-hours-claimed-exactly-once
    for day_key in ("weekday_hours", "weekend_hours"):
        coverage = np.zeros(24, dtype=int)
        for period in s["periods"]:
            mask = _hours_to_mask(period[day_key])
            coverage = coverage + mask.astype(int)
        if (coverage != 1).any():
            bad_hours = np.where(coverage != 1)[0].tolist()
            errors.append(
                f"{utility}.{day_key}: hours not claimed exactly once: "
                f"{bad_hours} (coverage: {coverage.tolist()})")

    # Rate field naming
    if s["season_split"]:
        for period in s["periods"]:
            for field in ("summer_rate", "winter_rate"):
                if field not in period:
                    errors.append(
                        f"{utility}.{period['name']}: season_split=True "
                        f"but {field} missing")
    else:
        for period in s["periods"]:
            if "rate_per_kwh" not in period:
                errors.append(
                    f"{utility}.{period['name']}: season_split=False "
                    f"but rate_per_kwh missing")

    return errors


# -----------------------------------------------------------------------------
# Per-utility period-weight mapping (weekday + weekend blended)
# -----------------------------------------------------------------------------

def period_weights_from_hourly(
    hourly_weights: np.ndarray, utility: str
) -> dict[str, float]:
    """Map a 24-hour weight vector (e.g. an EV charging profile) into
    EV-TOU period shares for `utility`.

    Blends weekday + weekend exposure at the CA workday/weekend ratio
    (240/365 weekdays). Returned shares sum to sum(hourly_weights).

    Raises KeyError if the utility's schedule isn't populated yet.
    """
    s = EV_TOU_SCHEDULES[utility]
    if s is None:
        raise KeyError(
            f"EV-TOU schedule for {utility} not populated yet")
    out: dict[str, float] = {}
    for period in s["periods"]:
        wd_mask = _hours_to_mask(period["weekday_hours"])
        we_mask = _hours_to_mask(period["weekend_hours"])
        share = (
            WORKDAY_SHARE * hourly_weights[wd_mask].sum()
            + WEEKEND_HOLIDAY_SHARE * hourly_weights[we_mask].sum()
        )
        out[period["name"]] = float(share)
    return out


def effective_price_under_profile(
    hourly_weights: np.ndarray, utility: str, season: str = "annual"
) -> float:
    """Charging-profile-weighted average $/kWh under `utility`'s EV-TOU.

    season: "annual" (mean of summer + winter if seasonal), "summer", or
    "winter". For schedules without season_split, all options yield the
    same number.
    """
    s = EV_TOU_SCHEDULES[utility]
    if s is None:
        raise KeyError(
            f"EV-TOU schedule for {utility} not populated yet")
    weights = period_weights_from_hourly(hourly_weights, utility)

    # Lookup price per period under requested season
    prices: dict[str, float] = {}
    for period in s["periods"]:
        if s["season_split"]:
            sp = period["summer_rate"]
            wp = period["winter_rate"]
            prices[period["name"]] = {
                "annual": (sp + wp) / 2,
                "summer": sp,
                "winter": wp,
            }[season]
        else:
            prices[period["name"]] = period["rate_per_kwh"]

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return float("nan")
    return sum(weights[p] * prices[p] for p in weights) / total_weight


def populated_utilities() -> list[str]:
    """Utilities whose EV-TOU schedule is populated (non-None)."""
    return [u for u, s in EV_TOU_SCHEDULES.items() if s is not None]
