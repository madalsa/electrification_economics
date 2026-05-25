"""Tests for bundle composition + per-bundle capex (thin wrapper)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bundles, config, subsidies


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


def test_subsidy_regimes_imported_from_subsidies():
    """bundles.SUBSIDY_REGIMES is a re-export of subsidies.REGIMES."""
    assert bundles.SUBSIDY_REGIMES == subsidies.REGIMES


def test_bundle_net_capex_zero_for_none():
    net, items = bundles.bundle_net_capex(
        "none", pv_kw=5, batt_kwh=13.5, is_care=False)
    assert net == 0
    assert items == {}


def test_bundle_net_capex_ignores_pv_sizing_for_non_pv_bundle():
    """If bundle has no PV (e.g., 'ev'), PV/battery sizing args are
    ignored — only EV capex applies."""
    net, items = bundles.bundle_net_capex(
        "ev", pv_kw=10, batt_kwh=27, is_care=False)
    assert "pv_itc" not in items
    assert "sgip_battery" not in items
    # Just EV premium + L2 charger as gross
    assert net == config.CAPEX["ev_premium"] + config.CAPEX["ev_charger"]


def test_bundle_net_capex_2024_vs_2026_regime():
    """OBBB hike: 2024 < 2026 in net capex for any tech-included bundle."""
    net_2024, _ = bundles.bundle_net_capex(
        "pv_bat_ev_hp", pv_kw=6, batt_kwh=13.5, is_care=False,
        regime="2024_federal")
    net_2026, _ = bundles.bundle_net_capex(
        "pv_bat_ev_hp", pv_kw=6, batt_kwh=13.5, is_care=False,
        regime="2026_federal")
    assert net_2024 < net_2026


def test_bundle_net_capex_2026_ca_added_helps():
    """Adding the CA state stack to 2026 federal reduces net capex."""
    net_fed, _ = bundles.bundle_net_capex(
        "pv_bat_ev_hp", pv_kw=6, batt_kwh=13.5, is_care=True,
        regime="2026_federal")
    net_ca, _ = bundles.bundle_net_capex(
        "pv_bat_ev_hp", pv_kw=6, batt_kwh=13.5, is_care=True,
        regime="2026_ca_added")
    assert net_ca < net_fed


def test_pv_sizing_grid_three_sizes_within_nbt_cap():
    sizes = bundles.pv_sizing_grid(annual_load_kwh=10000)
    assert len(sizes) == 3
    assert math.isclose(sizes[0], 10000 / 1700, rel_tol=1e-6)
    assert math.isclose(sizes[1], 1.15 * sizes[0])
    assert math.isclose(sizes[2], 1.25 * sizes[0])


def test_battery_sizing_grid_two_powerwalls():
    assert bundles.BATTERY_SIZING_KWH == [13.5, 27.0]


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
