"""Tests for EV-TOU schedule encoding + lookup helpers.

Each utility's schedule must (a) cover 24 hours exactly once for every
(season, day_type) it claims, (b) round-trip a flat hourly profile back
to total weight, (c) return the screenshot rates for known on-peak /
super-off-peak hours, and (d) document its IGFC base-services-charge
treatment.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import ev_tou_schedules as ev


# ============================================================================
# Schedule integrity
# ============================================================================

def test_sdge_schedule_validates():
    errors = ev.validate_schedule("sdge")
    assert errors == [], errors


def test_sce_schedule_validates():
    errors = ev.validate_schedule("sce")
    assert errors == [], errors


def test_pge_schedule_validates():
    errors = ev.validate_schedule("pge")
    assert errors == [], errors


def test_populated_utilities_lists_all_three():
    assert set(ev.populated_utilities()) == {"sdge", "sce", "pge"}


# ============================================================================
# Period weight mapping
# ============================================================================

def test_flat_hourly_profile_sums_to_one_sdge_weekday():
    flat = np.full(24, 1.0 / 24)
    weights = ev.period_weights_for_schedule(flat, "sdge", "year", "weekday")
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_flat_hourly_profile_sums_to_one_sce_summer_weekday():
    flat = np.full(24, 1.0 / 24)
    weights = ev.period_weights_for_schedule(
        flat, "sce", "summer", "weekday")
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_overnight_profile_lands_super_off_peak_sdge():
    """12am-6am charging on SDGE is super-off-peak under both day types."""
    profile = np.zeros(24)
    profile[0:6] = 1.0 / 6
    for day_type in ("weekday", "weekend"):
        weights = ev.period_weights_for_schedule(
            profile, "sdge", "year", day_type)
        assert math.isclose(weights["super_off_peak"], 1.0, abs_tol=1e-9)


def test_peak_5pm_lands_on_peak_sdge():
    profile = np.zeros(24)
    profile[17] = 1.0
    for day_type in ("weekday", "weekend"):
        weights = ev.period_weights_for_schedule(
            profile, "sdge", "year", day_type)
        assert math.isclose(weights["on_peak"], 1.0, abs_tol=1e-9)


def test_weekday_3pm_lands_off_peak_sdge():
    """3pm SDGE is off-peak (2-4pm window) on weekday and weekend."""
    profile = np.zeros(24)
    profile[15] = 1.0
    for day_type in ("weekday", "weekend"):
        weights = ev.period_weights_for_schedule(
            profile, "sdge", "year", day_type)
        assert math.isclose(weights["off_peak"], 1.0, abs_tol=1e-9)


# ============================================================================
# SCE: weekday vs weekend rate differentiation in summer
# ============================================================================

def test_sce_summer_weekday_5pm_lands_on_peak():
    profile = np.zeros(24)
    profile[17] = 1.0
    weights = ev.period_weights_for_schedule(
        profile, "sce", "summer", "weekday")
    assert weights["on_peak"] == 1.0


def test_sce_summer_weekend_5pm_lands_mid_peak():
    """The same 5pm hour on summer weekend is mid-peak, not on-peak."""
    profile = np.zeros(24)
    profile[17] = 1.0
    weights = ev.period_weights_for_schedule(
        profile, "sce", "summer", "weekend")
    assert weights["mid_peak"] == 1.0
    assert "on_peak" not in weights


def test_sce_winter_3pm_lands_super_off_peak():
    """3pm in SCE winter is super-off-peak (8am-4pm window)."""
    profile = np.zeros(24)
    profile[15] = 1.0
    for day_type in ("weekday", "weekend"):
        weights = ev.period_weights_for_schedule(
            profile, "sce", "winter", day_type)
        assert weights["super_off_peak"] == 1.0


def test_sce_winter_5pm_lands_mid_peak():
    """SCE winter 4-9pm is mid-peak (56c) for both weekday and weekend."""
    profile = np.zeros(24)
    profile[17] = 1.0
    for day_type in ("weekday", "weekend"):
        weights = ev.period_weights_for_schedule(
            profile, "sce", "winter", day_type)
        assert weights["mid_peak"] == 1.0


# ============================================================================
# Effective price lookups (round-trip to screenshot rates)
# ============================================================================

def test_sdge_on_peak_rate_matches_screenshot():
    profile = np.zeros(24); profile[17] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "sdge"), 0.533, abs_tol=1e-6)


def test_sdge_super_off_peak_rate_matches_screenshot():
    profile = np.zeros(24); profile[2] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "sdge"), 0.121, abs_tol=1e-6)


def test_sdge_off_peak_rate_matches_screenshot():
    profile = np.zeros(24); profile[15] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "sdge"), 0.476, abs_tol=1e-6)


def test_sce_summer_on_peak_blended_uses_workday_share():
    """5pm summer charging: weekday 59c, weekend 40c. Blended at
    WORKDAY_SHARE = 253/365 weekday share."""
    profile = np.zeros(24); profile[17] = 1.0
    expected = ev.WORKDAY_SHARE * 0.59 + ev.WEEKEND_HOLIDAY_SHARE * 0.40
    actual = ev.effective_price_under_profile(profile, "sce", "summer")
    assert math.isclose(actual, expected, abs_tol=1e-6)


def test_sce_winter_mid_peak_56c_matches_screenshot():
    profile = np.zeros(24); profile[17] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "sce", "winter"),
        0.56, abs_tol=1e-6)


def test_sce_winter_super_off_peak_24c_matches_screenshot():
    profile = np.zeros(24); profile[10] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "sce", "winter"),
        0.24, abs_tol=1e-6)


def test_sce_annual_blends_summer_and_winter_at_day_shares():
    """5pm charging: annual price should fall between pure summer and
    pure winter prices, biased toward winter (8 months vs 4)."""
    profile = np.zeros(24); profile[17] = 1.0
    summer = ev.effective_price_under_profile(profile, "sce", "summer")
    winter = ev.effective_price_under_profile(profile, "sce", "winter")
    annual = ev.effective_price_under_profile(profile, "sce", "annual")
    assert min(summer, winter) <= annual <= max(summer, winter)
    # Jun-Sep = 122 days, rest = 243 days. Annual leans toward winter.
    expected = (122 / 365) * summer + (243 / 365) * winter
    assert math.isclose(annual, expected, abs_tol=1e-6)


def test_sdge_year_round_no_season_difference():
    """Non-seasonal: summer / winter / annual must collapse to same number."""
    profile = np.full(24, 1.0 / 24)
    annual = ev.effective_price_under_profile(profile, "sdge", "annual")
    summer = ev.effective_price_under_profile(profile, "sdge", "summer")
    winter = ev.effective_price_under_profile(profile, "sdge", "winter")
    assert annual == summer == winter


# ============================================================================
# PGE EV2-A: three periods, no weekday/weekend differentiation
# ============================================================================

def test_pge_5pm_on_peak_summer_rate():
    profile = np.zeros(24); profile[17] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "pge", "summer"),
        0.53809, abs_tol=1e-6)


def test_pge_5pm_on_peak_winter_rate():
    profile = np.zeros(24); profile[17] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "pge", "winter"),
        0.41099, abs_tol=1e-6)


def test_pge_3pm_partial_peak_summer_rate():
    """3pm is the lone-hour partial-peak window pre-on-peak."""
    profile = np.zeros(24); profile[15] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "pge", "summer"),
        0.42760, abs_tol=1e-6)


def test_pge_3pm_partial_peak_winter_rate():
    profile = np.zeros(24); profile[15] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "pge", "winter"),
        0.39428, abs_tol=1e-6)


def test_pge_10pm_partial_peak_post_onpeak():
    """9pm-midnight is the second partial-peak window."""
    profile = np.zeros(24); profile[22] = 1.0
    assert math.isclose(
        ev.effective_price_under_profile(profile, "pge", "summer"),
        0.42760, abs_tol=1e-6)


def test_pge_2pm_off_peak_year_round_same_rate():
    """Off-peak rate is identical summer and winter for PGE."""
    profile = np.zeros(24); profile[14] = 1.0
    summer = ev.effective_price_under_profile(profile, "pge", "summer")
    winter = ev.effective_price_under_profile(profile, "pge", "winter")
    assert math.isclose(summer, 0.22558, abs_tol=1e-6)
    assert summer == winter


def test_pge_overnight_off_peak():
    """Midnight-3pm is off-peak; concentrating charging there yields
    off-peak rate."""
    profile = np.zeros(24); profile[0:15] = 1.0 / 15
    assert math.isclose(
        ev.effective_price_under_profile(profile, "pge", "summer"),
        0.22558, abs_tol=1e-6)


def test_pge_no_weekday_weekend_difference():
    """PGE EV2-A: 'every day including weekends and holidays'. The
    weekday and weekend schedules must be identical."""
    for season in ("summer", "winter"):
        wd = ev.EV_TOU_SCHEDULES["pge"]["schedules"][f"{season}_weekday"]
        we = ev.EV_TOU_SCHEDULES["pge"]["schedules"][f"{season}_weekend"]
        assert wd == we, season


def test_pge_three_tier_igfc_documented():
    """PGE publishes three tiers; this paper models only CARE vs
    Non-CARE but FERA is preserved for documentation."""
    bsc = ev.EV_TOU_SCHEDULES["pge"]["igfc_base_services_charge"]
    # Sanity: CARE < FERA < Non-CARE
    assert bsc["care_monthly"] < bsc["fera_monthly"] < bsc["non_care_monthly"]
    # Roughly $6 / $12 / $24 monthly
    assert math.isclose(bsc["care_monthly"],     6.00, abs_tol=0.5)
    assert math.isclose(bsc["fera_monthly"],    12.07, abs_tol=0.5)
    assert math.isclose(bsc["non_care_monthly"], 24.13, abs_tol=0.5)


def test_pge_season_for_month():
    for m in (6, 7, 8, 9):
        assert ev.season_for_month("pge", m) == "summer"
    for m in (1, 2, 3, 4, 5, 10, 11, 12):
        assert ev.season_for_month("pge", m) == "winter"


def test_pge_annual_blends_summer_winter_at_day_shares():
    """5pm charging: annual = (122/365)*summer_onpeak + (243/365)*winter_onpeak."""
    profile = np.zeros(24); profile[17] = 1.0
    annual = ev.effective_price_under_profile(profile, "pge", "annual")
    expected = (122 / 365) * 0.53809 + (243 / 365) * 0.41099
    assert math.isclose(annual, expected, abs_tol=1e-6)


# ============================================================================
# Season-for-month
# ============================================================================

def test_sce_season_for_month():
    for m in (6, 7, 8, 9):
        assert ev.season_for_month("sce", m) == "summer"
    for m in (1, 2, 3, 4, 5, 10, 11, 12):
        assert ev.season_for_month("sce", m) == "winter"


def test_sdge_season_for_month_always_year():
    for m in range(1, 13):
        assert ev.season_for_month("sdge", m) == "year"


# ============================================================================
# Documentation contracts (BSC, CCA, rate basis)
# ============================================================================

def test_both_utilities_document_volumetric_only_basis():
    for u in ("sdge", "sce"):
        s = ev.EV_TOU_SCHEDULES[u]
        assert s["rate_basis"] == "volumetric_only_excludes_base_services_charge"


def test_both_utilities_document_non_cca_class():
    for u in ("sdge", "sce"):
        assert ev.EV_TOU_SCHEDULES[u]["customer_class"] == "non_cca_bundled"


def test_bsc_schema_is_uniformly_tiered_monthly():
    """All three utilities now store BSC in $/month at the same tier
    granularity (CARE / FERA / Non-CARE), regardless of how the source
    tariff publishes the value. This is the unified IGFC modeling API."""
    for u in ev.populated_utilities():
        bsc = ev.EV_TOU_SCHEDULES[u]["igfc_base_services_charge"]
        assert bsc["structure"] == "tiered_monthly", u
        for field in ("care_monthly", "fera_monthly", "non_care_monthly"):
            assert field in bsc, f"{u}: missing {field}"


def test_sce_non_care_bsc_monthly_value():
    """SCE non-CARE BSC ~$24/mo (sourced from $0.79/day, normalized)."""
    bsc = ev.EV_TOU_SCHEDULES["sce"]["igfc_base_services_charge"]
    assert 23.5 < bsc["non_care_monthly"] < 24.5


def test_pge_three_tier_bsc_monthly_values():
    """PGE has all three tier values populated in monthly."""
    bsc = ev.EV_TOU_SCHEDULES["pge"]["igfc_base_services_charge"]
    assert 5 < bsc["care_monthly"] < 7
    assert 11 < bsc["fera_monthly"] < 13
    assert 23 < bsc["non_care_monthly"] < 26


def test_sdge_bsc_tier_values_not_yet_populated():
    """SDGE schema fields are in place but values pending data pull.
    This test will need updating once the SDGE BSC schedule is added."""
    bsc = ev.EV_TOU_SCHEDULES["sdge"]["igfc_base_services_charge"]
    assert bsc["care_monthly"] is None
    assert bsc["non_care_monthly"] is None


def test_workday_share_uses_eight_holidays():
    expected = 253.0 / 365.0
    assert abs(ev.WORKDAY_SHARE - expected) < 1e-9
    assert abs(ev.WORKDAY_SHARE + ev.WEEKEND_HOLIDAY_SHARE - 1.0) < 1e-9


# ============================================================================
# Cross-utility period-naming documentation contract
# ============================================================================

def test_pge_uses_partial_peak_label():
    """PGE has its tariff-specific 'partial_peak' label; no other
    utility uses this label."""
    pge_periods = set(ev.period_names_for_utility("pge"))
    assert "partial_peak" in pge_periods
    for u in ("sce", "sdge"):
        assert "partial_peak" not in set(ev.period_names_for_utility(u)), u


def test_sce_uses_mid_peak_label_pge_sdge_do_not():
    """SCE's mid_peak appears in summer weekend and winter. PGE and
    SDGE don't use this label."""
    assert "mid_peak" in set(ev.period_names_for_utility("sce"))
    assert "mid_peak" not in set(ev.period_names_for_utility("pge"))
    assert "mid_peak" not in set(ev.period_names_for_utility("sdge"))


def test_sdge_off_peak_is_middle_rate_not_cheapest():
    """SDGE's `off_peak` is the MIDDLE rate (47.6c). The cheapest is
    super_off_peak (12.1c). Reading SDGE's 'off_peak' as 'cheapest'
    would be wrong; this test prevents accidental rename / collapse."""
    off_peak = ev.rate_for_period("sdge", "year", "weekday", "off_peak")
    super_off = ev.rate_for_period("sdge", "year", "weekday",
                                    "super_off_peak")
    on_peak = ev.rate_for_period("sdge", "year", "weekday", "on_peak")
    assert super_off < off_peak < on_peak
    assert math.isclose(super_off, 0.121, abs_tol=1e-6)
    assert math.isclose(off_peak,  0.476, abs_tol=1e-6)
    assert math.isclose(on_peak,   0.533, abs_tol=1e-6)


def test_pge_period_names_are_three():
    assert set(ev.period_names_for_utility("pge")) == {
        "off_peak", "partial_peak", "on_peak"}


def test_sce_period_names_are_four_across_seasons():
    """SCE union: off_peak, on_peak (summer wd only), mid_peak (summer
    we + winter), super_off_peak (winter only)."""
    assert set(ev.period_names_for_utility("sce")) == {
        "off_peak", "on_peak", "mid_peak", "super_off_peak"}


def test_sdge_period_names_are_three():
    assert set(ev.period_names_for_utility("sdge")) == {
        "super_off_peak", "off_peak", "on_peak"}


def test_rate_for_period_lookups_match_screenshots():
    """Spot checks: pull individual rates by period name."""
    assert math.isclose(
        ev.rate_for_period("pge", "summer", "weekday", "on_peak"),
        0.53809, abs_tol=1e-6)
    assert math.isclose(
        ev.rate_for_period("sce", "summer", "weekend", "mid_peak"),
        0.40, abs_tol=1e-6)
    assert math.isclose(
        ev.rate_for_period("sce", "summer", "weekday", "on_peak"),
        0.59, abs_tol=1e-6)
    assert math.isclose(
        ev.rate_for_period("sdge", "year", "weekend", "super_off_peak"),
        0.121, abs_tol=1e-6)


def test_rate_for_period_unknown_raises():
    """Asking for a nonexistent period raises rather than silently
    returning a default."""
    try:
        ev.rate_for_period("pge", "summer", "weekday", "mid_peak")
    except KeyError:
        return
    raise AssertionError(
        "expected KeyError for mid_peak on PGE (no such period)")


def test_sdge_seasonal_key_raises():
    """Asking for summer/winter on a non-seasonal utility's specific
    schedule key should raise (the convenience effective_price collapses,
    but the lower-level period_weights_for_schedule wants an exact key)."""
    profile = np.full(24, 1.0 / 24)
    try:
        ev.period_weights_for_schedule(profile, "sdge", "summer", "weekday")
    except KeyError:
        return
    raise AssertionError(
        "expected KeyError for summer_weekday on non-seasonal SDGE")


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
