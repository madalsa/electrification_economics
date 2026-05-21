"""Tests for bundle composition + per-bundle capex."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bundles, config, payback_npv as p


def test_eight_bundles_defined():
    assert set(bundles.BUNDLES) == {
        "none", "pv_bat", "ev", "hp",
        "pv_bat_ev", "pv_bat_hp", "ev_hp", "pv_bat_ev_hp"}


def test_parse_bundle_full_electrification():
    assert bundles.parse_bundle("pv_bat_ev_hp") == (True, True, True)


def test_parse_bundle_none_returns_false_for_all():
    assert bundles.parse_bundle("none") == (False, False, False)


def test_parse_bundle_pv_bat_hp_excludes_ev():
    assert bundles.parse_bundle("pv_bat_hp") == (True, False, True)


def test_capex_breakdown_includes_panel_and_charger():
    """HP bundle gets panel upgrade; EV bundle gets L2 charger."""
    cb = bundles.bundle_capex_breakdown("pv_bat_ev_hp", pv_kw=5, batt_kwh=13.5)
    assert cb.pv_kw == 5 and cb.battery_kwh == 13.5
    assert cb.ev and cb.ev_charger
    assert cb.heat_pump_space and cb.heat_pump_water
    assert cb.induction_range and cb.panel_upgrade


def test_capex_zero_for_none_bundle():
    cb = bundles.bundle_capex_breakdown("none", pv_kw=5, batt_kwh=10)
    assert cb.gross_capex() == 0


def test_subsidy_regime_2024_restores_federal_credits():
    """Under 2024_counterfactual, PV gets 30% ITC; under 2026_base, 0."""
    _, items_2026 = bundles.bundle_net_capex(
        "pv_bat", pv_kw=6, batt_kwh=13.5, income_category="Medium",
        subsidy_regime="2026_base")
    _, items_2024 = bundles.bundle_net_capex(
        "pv_bat", pv_kw=6, batt_kwh=13.5, income_category="Medium",
        subsidy_regime="2024_counterfactual")
    assert "fed_25d_pv" not in items_2026
    assert "fed_25d_pv" in items_2024
    pv_gross = 6 * config.CAPEX["pv_per_kw"]
    assert math.isclose(items_2024["fed_25d_pv"], 0.30 * pv_gross)


def test_2024_capex_lower_than_2026():
    """OBBB took away federal subsidies, so 2026 net capex is HIGHER
    than 2024-counterfactual. This is the paper's whole question."""
    net_2026, _ = bundles.bundle_net_capex(
        "pv_bat_ev_hp", pv_kw=6, batt_kwh=13.5, income_category="Medium",
        subsidy_regime="2026_base")
    net_2024, _ = bundles.bundle_net_capex(
        "pv_bat_ev_hp", pv_kw=6, batt_kwh=13.5, income_category="Medium",
        subsidy_regime="2024_counterfactual")
    assert net_2024 < net_2026


def test_pv_sizing_grid_three_sizes_within_nbt_cap():
    """PV sizes: 1.00× / 1.15× / 1.25× of expanded annual load.
    All three are within the residential NBT interconnection eligibility
    (≤125% of historical load); the previous 3× tier is dropped because
    it's not interconnection-eligible for residential under NBT."""
    sizes = bundles.pv_sizing_grid(annual_load_kwh=10000)
    assert len(sizes) == 3
    assert math.isclose(sizes[0], 10000 / 1700, rel_tol=1e-6)
    assert math.isclose(sizes[1], 1.15 * sizes[0])
    assert math.isclose(sizes[2], 1.25 * sizes[0])
    # Largest tier still within NBT cap
    assert sizes[2] <= 1.25 * sizes[0]


def test_battery_sizing_grid_two_powerwalls():
    assert bundles.BATTERY_SIZING_KWH == [13.5, 27.0]


def test_subsidy_regimes_are_exactly_two():
    assert set(bundles.SUBSIDY_REGIMES) == {"2026_base", "2024_counterfactual"}


def test_invalid_subsidy_regime_raises():
    try:
        bundles.bundle_net_capex("pv_bat", 5, 13.5, "Medium",
                                  subsidy_regime="totally_made_up")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


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
    sys.exit(failures)
