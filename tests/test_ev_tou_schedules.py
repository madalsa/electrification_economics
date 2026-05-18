"""Tests for EV-TOU schedule encoding + lookup helpers.

The schedule per utility must (a) cover 24 hours exactly once for both
weekday and weekend templates, (b) round-trip a flat hourly profile back
to total weight, (c) return the screenshot rates for the on-peak /
super-off-peak hours.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import ev_tou_schedules as ev


# ---- schedule integrity ----

def test_sdge_schedule_validates():
    errors = ev.validate_schedule("sdge")
    assert errors == [], errors


def test_pge_sce_unpopulated_flag_until_verified():
    """PGE / SCE schedules return validation-error sentinels until we
    paste in screenshots and verify."""
    for u in ("pge", "sce"):
        errors = ev.validate_schedule(u)
        assert any("not yet populated" in e for e in errors)


def test_populated_utilities_lists_only_sdge():
    assert ev.populated_utilities() == ["sdge"]


# ---- period-weight mapping ----

def test_flat_hourly_profile_sums_to_one():
    """A flat 1/24 charging profile should produce period weights
    that sum to ~1.0 (we blend weekday + weekend exposure)."""
    flat = np.full(24, 1.0 / 24)
    weights = ev.period_weights_from_hourly(flat, "sdge")
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_overnight_profile_lands_super_off_peak_sdge():
    """All charging between 12am-6am should be 100% super-off-peak for
    SDGE under both weekday and weekend templates (both have super-off-
    peak covering that window)."""
    profile = np.zeros(24)
    profile[0:6] = 1.0 / 6  # uniform overnight charging
    weights = ev.period_weights_from_hourly(profile, "sdge")
    assert math.isclose(weights["super_off_peak"], 1.0, abs_tol=1e-9)
    assert weights["on_peak"] == 0
    assert weights["off_peak"] == 0


def test_peak_5pm_lands_on_peak_sdge():
    """Charging concentrated at 5pm should land entirely on-peak for SDGE
    (4-9pm on-peak window applies to both weekday and weekend)."""
    profile = np.zeros(24)
    profile[17] = 1.0
    weights = ev.period_weights_from_hourly(profile, "sdge")
    assert math.isclose(weights["on_peak"], 1.0, abs_tol=1e-9)
    assert weights["super_off_peak"] == 0


def test_weekday_midday_lands_super_off_peak_sdge():
    """11am SDGE on a weekday is super-off-peak. On a weekend it's also
    super-off-peak (weekend window 12am-2pm includes 11am). So the blend
    is 100% super-off-peak."""
    profile = np.zeros(24)
    profile[11] = 1.0
    weights = ev.period_weights_from_hourly(profile, "sdge")
    assert math.isclose(weights["super_off_peak"], 1.0, abs_tol=1e-9)


def test_weekday_3pm_split_between_off_peak_and_super_off_peak():
    """3pm on weekday is off-peak (2-4pm window) for SDGE; 3pm on
    weekend is also off-peak (2-4pm). So 100% off-peak."""
    profile = np.zeros(24)
    profile[15] = 1.0
    weights = ev.period_weights_from_hourly(profile, "sdge")
    assert math.isclose(weights["off_peak"], 1.0, abs_tol=1e-9)


# ---- price lookups ----

def test_sdge_on_peak_rate_matches_screenshot():
    """Concentrating all charging at 5pm should yield the on-peak rate."""
    profile = np.zeros(24)
    profile[17] = 1.0
    price = ev.effective_price_under_profile(profile, "sdge")
    assert math.isclose(price, 0.533, abs_tol=1e-6)


def test_sdge_super_off_peak_rate_matches_screenshot():
    """Concentrating all charging at 2am should yield the super-off-peak
    rate."""
    profile = np.zeros(24)
    profile[2] = 1.0
    price = ev.effective_price_under_profile(profile, "sdge")
    assert math.isclose(price, 0.121, abs_tol=1e-6)


def test_sdge_off_peak_rate_matches_screenshot():
    """3pm charging lands off-peak."""
    profile = np.zeros(24)
    profile[15] = 1.0
    price = ev.effective_price_under_profile(profile, "sdge")
    assert math.isclose(price, 0.476, abs_tol=1e-6)


def test_sdge_year_round_no_season_difference():
    """SDGE EV-TOU-5 has no season_split, so summer / winter / annual
    must return the same number under any profile."""
    profile = np.full(24, 1.0 / 24)
    assert (ev.effective_price_under_profile(profile, "sdge", "annual")
            == ev.effective_price_under_profile(profile, "sdge", "summer")
            == ev.effective_price_under_profile(profile, "sdge", "winter"))


def test_sdge_documents_volumetric_only_basis():
    """SDGE rates exclude the Base Services Charge (AB 205 IGFC).
    A regression here would mean someone re-added the BSC inline,
    which would double-count it against the canonical-6 fixed-charge
    component in bundle_economics."""
    s = ev.EV_TOU_SCHEDULES["sdge"]
    assert s["rate_basis"] == "volumetric_only_excludes_base_services_charge"


def test_sdge_documents_non_cca_class():
    """The default SDGE EV-TOU-5 rates are for non-CCA (full bundled)
    customers. CCA customers face different generation rates and would
    need a separate schedule entry."""
    s = ev.EV_TOU_SCHEDULES["sdge"]
    assert s["customer_class"] == "non_cca_bundled"


def test_workday_share_uses_eight_holidays():
    """365 - 104 weekend days - 8 holidays = 253 workdays. Verifies
    the CA tariff convention is reflected in the blending weight."""
    expected = 253.0 / 365.0
    assert abs(ev.WORKDAY_SHARE - expected) < 1e-9
    assert abs(
        ev.WORKDAY_SHARE + ev.WEEKEND_HOLIDAY_SHARE - 1.0) < 1e-9


def test_pge_raises_until_populated():
    profile = np.full(24, 1.0 / 24)
    try:
        ev.effective_price_under_profile(profile, "pge")
    except KeyError as exc:
        assert "not populated" in str(exc)
        return
    raise AssertionError("Expected KeyError for unpopulated PGE schedule")


if __name__ == "__main__":
    failures = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}  {e}")
                failures += 1
            except Exception as e:
                print(f"  ERR   {name}  {type(e).__name__}: {e}")
                failures += 1
    print(f"\n{failures} failure(s)" if failures else "\nall passed")
    sys.exit(failures)
