"""Electrification bundle definitions + per-bundle capex by subsidy regime.

A bundle is a combination of (PV+battery, EV, HP) that a household can
adopt. We model 8 combinations including the do-nothing baseline.

Capex per bundle is computed via payback_npv.apply_capex_stack against
either the 2026-base incentive stack (post-OBBB; federal 25C/25D/30D=0)
or the 2024-counterfactual stack (pre-OBBB; restored federal credits).
The 2024 vs 2026 contrast is the paper's central question — how much
NPV gap did OBBB open, and how much can rate design close.
"""

from __future__ import annotations

from src import config, payback_npv as p


BUNDLES = (
    "none",            # baseline (no electrification)
    "pv_bat",          # PV + battery only
    "ev",              # EV only
    "hp",              # whole-home HP (HP space + HPWH + induction + panel)
    "pv_bat_ev",
    "pv_bat_hp",
    "ev_hp",
    "pv_bat_ev_hp",    # full electrification
)


def parse_bundle(bundle: str) -> tuple[bool, bool, bool]:
    """(has_pv_bat, has_ev, has_hp). 'none' yields all False."""
    if bundle == "none":
        return False, False, False
    tokens = bundle.split("_")
    return ("pv" in tokens), ("ev" in tokens), ("hp" in tokens)


def bundle_capex_breakdown(
    bundle: str, pv_kw: float = 0.0, batt_kwh: float = 0.0,
) -> p.CapexBreakdown:
    """Construct a CapexBreakdown for the bundle.

    HP bundles assume full Upgrade-11 substitution: HP space + HPWH +
    induction range + 200A panel upgrade. EV bundles add an L2 charger.
    """
    has_pv_bat, has_ev, has_hp = parse_bundle(bundle)
    return p.CapexBreakdown(
        pv_kw=pv_kw if has_pv_bat else 0.0,
        battery_kwh=batt_kwh if has_pv_bat else 0.0,
        ev=has_ev,
        ev_charger=has_ev,
        heat_pump_space=has_hp,
        heat_pump_water=has_hp,
        induction_range=has_hp,
        panel_upgrade=has_hp,
    )


def bundle_net_capex(
    bundle: str,
    pv_kw: float, batt_kwh: float,
    income_category: str,
    subsidy_regime: str = "2026_base",
    air_district: str | None = None,
) -> tuple[float, dict]:
    """Net capex after stacked subsidies for one bundle / household.

    subsidy_regime: "2026_base" (post-OBBB; default) or
                    "2024_counterfactual" (pre-OBBB federal stack restored).
    income_category: 'Low' (CARE/equity tier) / 'Medium' / 'High'.

    Returns (net_capex_$, itemized_rebates_dict).
    """
    if subsidy_regime not in ("2026_base", "2024_counterfactual"):
        raise ValueError(
            f"unknown subsidy_regime {subsidy_regime!r}; expected "
            f"'2026_base' or '2024_counterfactual'")
    capex = bundle_capex_breakdown(bundle, pv_kw, batt_kwh)
    ami_proxy = {"Low": 0.5, "Medium": 1.2, "High": 2.0}.get(
        str(income_category).strip(), 1.2)
    ctx = p.IncentiveContext(
        income_pct_ami=ami_proxy,
        air_district=air_district,
        use_2024_counterfactual=(subsidy_regime == "2024_counterfactual"),
    )
    return p.apply_capex_stack(capex, ctx)


def pv_sizing_grid(annual_load_kwh: float) -> list[float]:
    """PV sizes in kW for a household with given annual load.

    User specified: 1x, 1.5x, 3x annual load. Sized so PV annual yield
    (CA ~1700 kWh/kW/yr) equals the multiplier x annual load.
    """
    base_pv_kw = annual_load_kwh / 1700.0
    return [base_pv_kw, 1.5 * base_pv_kw, 3.0 * base_pv_kw]


# Battery sizing grid: 1 or 2 Tesla Powerwall 3 (13.5 kWh each).
BATTERY_SIZING_KWH = [13.5, 27.0]


SUBSIDY_REGIMES = ("2026_base", "2024_counterfactual")
