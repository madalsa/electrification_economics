"""Electrification bundle definitions + per-bundle capex (thin wrapper).

8 bundles modeled. HP = whole-home Upgrade 11 (HP space + HPWH +
induction + 200A panel). EV = single passenger vehicle (default VMT).

Capex subsidy logic lives in src/subsidies.py — a single table covering
the three regimes (2024_federal / 2026_federal / 2026_ca_added), tier
granularity CARE vs Non-CARE only. This file is just a router from
bundle composition to that subsidy table.

PV sizing grid: 1× / 1.15× / 1.25× expanded annual load (within CA NBT
residential interconnection cap of 125% historical load).
Battery sizing grid: 13.5 / 27 kWh (Tesla Powerwall 3 × {1, 2}).
"""

from __future__ import annotations

from src import config, subsidies


BUNDLES = (
    "none",
    "pv_bat",
    "ev",
    "hp",
    "pv_bat_ev",
    "pv_bat_hp",
    "ev_hp",
    "pv_bat_ev_hp",
)


def parse_bundle(bundle: str) -> tuple[bool, bool, bool]:
    """(has_pv_bat, has_ev, has_hp). 'none' yields all False."""
    if bundle == "none":
        return False, False, False
    tokens = bundle.split("_")
    return ("pv" in tokens), ("ev" in tokens), ("hp" in tokens)


def bundle_net_capex(
    bundle: str,
    pv_kw: float, batt_kwh: float,
    is_care: bool,
    regime: str = "2026_federal",
) -> tuple[float, dict]:
    """Net capex after subsidies for one (bundle, sizing, tier, regime).

    Thin wrapper over subsidies.compute_net_capex. PV/battery sizes are
    zeroed for bundles that don't include PV.
    """
    has_pv_bat, has_ev, has_hp = parse_bundle(bundle)
    return subsidies.compute_net_capex(
        pv_kw=pv_kw if has_pv_bat else 0.0,
        battery_kwh=batt_kwh if has_pv_bat else 0.0,
        has_ev=has_ev,
        has_hp=has_hp,
        regime=regime,
        is_care=is_care,
        capex_table=config.CAPEX,
    )


def pv_sizing_grid(annual_load_kwh: float) -> list[float]:
    """PV sizes in kW for a household with given annual load.

    Three sizes within NBT residential interconnection eligibility
    (≤125% of historical load):
      1.00× — sized to expanded annual load
      1.15× — modestly over-sized
      1.25× — at the NBT interconnection cap
    CA solar yield ~1700 kWh/kW/yr.
    """
    base_pv_kw = annual_load_kwh / 1700.0
    return [base_pv_kw, 1.15 * base_pv_kw, 1.25 * base_pv_kw]


BATTERY_SIZING_KWH = [13.5, 27.0]


# Re-export for convenience (used by run_npv to iterate)
SUBSIDY_REGIMES = subsidies.REGIMES
