"""Tests for the simplified subsidy stack (src/subsidies.py)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, subsidies


# ============================================================================
# Schedule structure
# ============================================================================

def test_three_regimes_defined():
    assert set(subsidies.REGIMES) == {
        "2024_federal", "2026_federal", "2026_ca_added"}


def test_2026_federal_is_all_zero():
    """OBBB zeroed every federal credit; the pure-federal-2026 regime
    must therefore have ZERO of everything (no state programs here)."""
    s = subsidies.SCHEDULES["2026_federal"]
    assert s.pv_itc_pct == 0
    assert s.battery_itc_pct == 0
    assert s.fed_25c_hp_max == 0
    assert s.fed_25c_hpwh_max == 0
    assert s.fed_25c_panel_max == 0
    assert s.fed_30d_ev == 0
    assert s.sgip_battery_general_per_kwh == 0
    assert s.sgip_battery_equity_per_kwh == 0
    assert s.state_hp_space_care == 0
    assert s.state_hp_space_non_care == 0
    assert s.state_ev_care == 0


def test_2024_federal_restores_pre_obbb_stack():
    s = subsidies.SCHEDULES["2024_federal"]
    assert s.pv_itc_pct == 0.30
    assert s.battery_itc_pct == 0.30
    assert s.fed_25c_hp_max == 2000
    assert s.fed_25c_hpwh_max == 2000
    assert s.fed_25c_panel_max == 600
    assert s.fed_30d_ev == 7500


def test_2026_ca_added_state_programs():
    s = subsidies.SCHEDULES["2026_ca_added"]
    # Federal still zero (OBBB)
    assert s.pv_itc_pct == 0
    assert s.fed_30d_ev == 0
    # State programs present
    assert s.sgip_battery_general_per_kwh == 200
    assert s.sgip_battery_equity_per_kwh == 850
    assert s.state_hp_space_care > s.state_hp_space_non_care
    assert s.state_hpwh_care > s.state_hpwh_non_care
    assert s.state_ev_care == 7500
    assert s.state_ev_non_care == 0


def test_care_gets_higher_state_subsidies_than_non_care():
    """In the CA-added regime, CARE tier is always >= Non-CARE tier."""
    s = subsidies.SCHEDULES["2026_ca_added"]
    assert s.state_hp_space_care >= s.state_hp_space_non_care
    assert s.state_hpwh_care >= s.state_hpwh_non_care
    assert s.state_induction_care >= s.state_induction_non_care
    assert s.state_ev_care >= s.state_ev_non_care
    assert s.sgip_battery_equity_per_kwh >= s.sgip_battery_general_per_kwh


# ============================================================================
# compute_net_capex behavior
# ============================================================================

def _gross_pv_bat(pv_kw, batt_kwh):
    return (pv_kw * config.CAPEX["pv_per_kw"]
            + batt_kwh * config.CAPEX["battery_per_kwh"])


def test_pv_battery_2024_gets_30pct_itc():
    pv_kw, batt_kwh = 6, 13.5
    net, items = subsidies.compute_net_capex(
        pv_kw=pv_kw, battery_kwh=batt_kwh, has_ev=False, has_hp=False,
        regime="2024_federal", is_care=False, capex_table=config.CAPEX)
    pv_gross = pv_kw * config.CAPEX["pv_per_kw"]
    batt_gross = batt_kwh * config.CAPEX["battery_per_kwh"]
    assert math.isclose(items["pv_itc"], 0.30 * pv_gross)
    assert math.isclose(items["battery_itc"], 0.30 * batt_gross)
    assert net < pv_gross + batt_gross


def test_pv_battery_2026_no_federal_no_state():
    """Pure 2026 federal regime has nothing for PV/battery."""
    net, items = subsidies.compute_net_capex(
        pv_kw=6, battery_kwh=13.5, has_ev=False, has_hp=False,
        regime="2026_federal", is_care=False,
        capex_table=config.CAPEX)
    assert items == {}
    assert math.isclose(net, _gross_pv_bat(6, 13.5))


def test_battery_sgip_care_vs_non_care():
    """CARE gets equity SGIP ($850/kWh, uncapped); Non-CARE gets general
    ($200/kWh, capped at 30 kWh)."""
    for kwh in (13.5, 27.0):
        _, items_care = subsidies.compute_net_capex(
            pv_kw=0, battery_kwh=kwh, has_ev=False, has_hp=False,
            regime="2026_ca_added", is_care=True,
            capex_table=config.CAPEX)
        _, items_nc = subsidies.compute_net_capex(
            pv_kw=0, battery_kwh=kwh, has_ev=False, has_hp=False,
            regime="2026_ca_added", is_care=False,
            capex_table=config.CAPEX)
        # Equity > General
        assert items_care["sgip_battery"] > items_nc["sgip_battery"]
        # Equity = 850 * kwh
        assert math.isclose(items_care["sgip_battery"], 850 * kwh)


def test_battery_sgip_general_capped_at_30_kwh():
    """27 kWh battery (2 Powerwalls) hits the 30 kWh general cap."""
    _, items = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=27.0, has_ev=False, has_hp=False,
        regime="2026_ca_added", is_care=False, capex_table=config.CAPEX)
    # 27 kWh is under the 30 kWh cap, so no clamp here
    assert math.isclose(items["sgip_battery"], 200 * 27.0)
    # Now hypothetically a 50 kWh battery would be capped at 30 kWh
    # (we don't have that in our grid but verify the logic)
    _, items_big = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=50.0, has_ev=False, has_hp=False,
        regime="2026_ca_added", is_care=False, capex_table=config.CAPEX)
    assert math.isclose(items_big["sgip_battery"], 200 * 30.0)


def test_hp_2024_gets_federal_25c_no_state():
    _, items = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=False, has_hp=True,
        regime="2024_federal", is_care=False, capex_table=config.CAPEX)
    assert items["fed_25c_hp"] == 2000
    assert items["fed_25c_hpwh"] == 2000
    assert items["fed_25c_panel"] == 600
    assert "state_hp_space" not in items


def test_hp_2026_ca_added_care_vs_non_care():
    _, items_care = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=False, has_hp=True,
        regime="2026_ca_added", is_care=True, capex_table=config.CAPEX)
    _, items_nc = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=False, has_hp=True,
        regime="2026_ca_added", is_care=False, capex_table=config.CAPEX)
    assert items_care["state_hp_space"] == 11000
    assert items_nc["state_hp_space"] == 5000
    assert items_care["state_hpwh"] == 10000
    assert items_nc["state_hpwh"] == 7000
    assert items_care["state_induction"] == 840
    assert "state_induction" not in items_nc
    # No federal under 2026
    assert "fed_25c_hp" not in items_care


def test_ev_state_only_for_care_in_ca_added():
    _, items_care = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=True, has_hp=False,
        regime="2026_ca_added", is_care=True, capex_table=config.CAPEX)
    _, items_nc = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=True, has_hp=False,
        regime="2026_ca_added", is_care=False, capex_table=config.CAPEX)
    assert items_care["state_ev"] == 7500
    assert "state_ev" not in items_nc


def test_ev_federal_2024_gets_7500_30d():
    _, items = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=True, has_hp=False,
        regime="2024_federal", is_care=False, capex_table=config.CAPEX)
    assert items["fed_30d_ev"] == 7500
    assert "state_ev" not in items


def test_net_capex_clamped_at_zero():
    """Rebates exceed gross capex (e.g., HPWH for CARE in CA-added):
    net should be 0, never negative."""
    # HPWH alone gross is $5,500; CARE state stack adds $10K for HPWH
    # which exceeds gross when bundled with other rebates.
    net, items = subsidies.compute_net_capex(
        pv_kw=0, battery_kwh=0, has_ev=False, has_hp=True,
        regime="2026_ca_added", is_care=True, capex_table=config.CAPEX)
    # Gross HP stack = HP + HPWH + induction + panel
    gross = (config.CAPEX["heat_pump_space"]
             + config.CAPEX["heat_pump_water"]
             + config.CAPEX["induction_range"]
             + config.CAPEX["panel_upgrade_200a"])
    # If items >= gross, net is 0; otherwise net = gross - items
    assert net >= 0
    if sum(items.values()) >= gross:
        assert net == 0


def test_2024_capex_lower_than_2026_for_pv_bundle():
    """The whole paper's point: OBBB raised net capex by removing the
    federal stack. Verify the regime ordering is right."""
    net_2024, _ = subsidies.compute_net_capex(
        pv_kw=6, battery_kwh=13.5, has_ev=True, has_hp=True,
        regime="2024_federal", is_care=False, capex_table=config.CAPEX)
    net_2026, _ = subsidies.compute_net_capex(
        pv_kw=6, battery_kwh=13.5, has_ev=True, has_hp=True,
        regime="2026_federal", is_care=False, capex_table=config.CAPEX)
    assert net_2026 > net_2024   # OBBB removed subsidies -> higher net


def test_ca_added_lower_than_pure_federal_2026():
    """Adding the CA stack on top of 2026 should reduce net capex."""
    net_fed_only, _ = subsidies.compute_net_capex(
        pv_kw=6, battery_kwh=13.5, has_ev=True, has_hp=True,
        regime="2026_federal", is_care=True, capex_table=config.CAPEX)
    net_with_ca, _ = subsidies.compute_net_capex(
        pv_kw=6, battery_kwh=13.5, has_ev=True, has_hp=True,
        regime="2026_ca_added", is_care=True, capex_table=config.CAPEX)
    assert net_with_ca < net_fed_only


def test_unknown_regime_raises():
    try:
        subsidies.compute_net_capex(
            pv_kw=6, battery_kwh=13.5, has_ev=False, has_hp=False,
            regime="totally_made_up", is_care=False,
            capex_table=config.CAPEX)
    except ValueError as e:
        assert "totally_made_up" in str(e)
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
    print(f"\n{failures} failure(s)" if failures else "\nall passed")
    sys.exit(failures)
