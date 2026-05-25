"""Single source of truth for capex subsidies.

Three regimes (each independently computed; output carries one NPV
column per regime):

  '2024_federal'  — pre-OBBB federal stack only, no CA state programs
  '2026_federal'  — post-OBBB federal (federal = $0), no CA state
  '2026_ca_added' — 2026 federal + CA state programs (optional overlay)

Tier granularity: CARE vs Non-CARE only. CARE = household with
income_category == 'Low' (in.income < $50K per
representative_buildings.INCOME_TO_CATEGORY). No AMI sub-tiers, no
DAC, no air-district splits.

Two clean analytical questions:
  Headline OBBB gap     = npv_2024_federal − npv_2026_federal
  CA recovery (state)   = npv_2026_ca_added − npv_2026_federal

Rate-design recovery is orthogonal: it's the NPV spread across the 40
rate scenarios at a given regime, particularly within 2026_federal
where the federal-removal damage is largest.

All numbers in 2026 USD. Sources documented in paper/methods.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubsidySchedule:
    """All subsidies for one regime, in flat dollars (or % for ITC).

    Federal credits apply equally to all tiers (when they exist at all).
    CA state programs are tier-aware (CARE vs Non-CARE separately).
    """
    name: str

    # ---- Federal stack ----
    pv_itc_pct: float                  # % of PV gross capex
    battery_itc_pct: float             # % of battery gross capex
    fed_25c_hp_max: float              # flat $ for HP space
    fed_25c_hpwh_max: float            # flat $ for HPWH
    fed_25c_panel_max: float           # flat $ for panel upgrade
    fed_30d_ev: float                  # flat $ for EV purchase

    # ---- CA state stack ----
    # SGIP battery: per kWh with capacity cap on general tier only
    sgip_battery_general_per_kwh: float
    sgip_battery_general_cap_kwh: float
    sgip_battery_equity_per_kwh: float   # no cap on equity tier

    # State HP / HPWH / induction / EV: flat $ per (tech, tier)
    # (aggregates TECH, SGIP-HPWH, Golden State, HOMES, CC4A/DCAP)
    state_hp_space_care: float
    state_hp_space_non_care: float
    state_hpwh_care: float
    state_hpwh_non_care: float
    state_induction_care: float
    state_induction_non_care: float
    state_ev_care: float
    state_ev_non_care: float


SCHEDULES: dict[str, SubsidySchedule] = {
    "2024_federal": SubsidySchedule(
        name="2024_federal",
        # Federal stack restored to pre-OBBB (Sections 25C, 25D, 30D, 30C)
        pv_itc_pct=0.30,
        battery_itc_pct=0.30,
        fed_25c_hp_max=2000,
        fed_25c_hpwh_max=2000,
        fed_25c_panel_max=600,
        fed_30d_ev=7500,
        # No state programs in pure federal regime
        sgip_battery_general_per_kwh=0,
        sgip_battery_general_cap_kwh=0,
        sgip_battery_equity_per_kwh=0,
        state_hp_space_care=0,
        state_hp_space_non_care=0,
        state_hpwh_care=0,
        state_hpwh_non_care=0,
        state_induction_care=0,
        state_induction_non_care=0,
        state_ev_care=0,
        state_ev_non_care=0,
    ),
    "2026_federal": SubsidySchedule(
        name="2026_federal",
        # OBBB (P.L. 119-21, July 2025) zeroed Sections 25C / 25D / 30D
        # for installations after 12/31/2025 (30D after 9/30/2025).
        pv_itc_pct=0.0,
        battery_itc_pct=0.0,
        fed_25c_hp_max=0,
        fed_25c_hpwh_max=0,
        fed_25c_panel_max=0,
        fed_30d_ev=0,
        # No state programs in pure federal-only regime
        sgip_battery_general_per_kwh=0,
        sgip_battery_general_cap_kwh=0,
        sgip_battery_equity_per_kwh=0,
        state_hp_space_care=0,
        state_hp_space_non_care=0,
        state_hpwh_care=0,
        state_hpwh_non_care=0,
        state_induction_care=0,
        state_induction_non_care=0,
        state_ev_care=0,
        state_ev_non_care=0,
    ),
    "2026_ca_added": SubsidySchedule(
        name="2026_ca_added",
        # Federal zeroed (OBBB)
        pv_itc_pct=0.0,
        battery_itc_pct=0.0,
        fed_25c_hp_max=0,
        fed_25c_hpwh_max=0,
        fed_25c_panel_max=0,
        fed_30d_ev=0,
        # CA state programs (aggregated dollar values; see methods.md)
        sgip_battery_general_per_kwh=200,   # General Market mid-step
        sgip_battery_general_cap_kwh=30,    # 30 kWh GM cap
        sgip_battery_equity_per_kwh=850,    # Equity tier; no cap
        # HP space: TECH equity ($3.5K) + HOMES low-inc ($8K) ≈ $11K CARE
        #           TECH market ($1K)  + HOMES market ($4K) = $5K Non-CARE
        state_hp_space_care=11000,
        state_hp_space_non_care=5000,
        # HPWH: TECH equity + SGIP-HPWH LI + Golden State ≈ $10K CARE
        #       TECH market + SGIP-HPWH std + Golden State ≈ $7K Non-CARE
        state_hpwh_care=10000,
        state_hpwh_non_care=7000,
        # Induction: HEAR cap when funded; not available Non-CARE
        state_induction_care=840,
        state_induction_non_care=0,
        # EV: DCAP statewide for income-eligible (≤300% FPL ≈ CARE proxy)
        # — $7,500 base. CC4A would add $2-4.5K in some districts but
        # not SDGE; we use DCAP-equivalent statewide to keep the model
        # tier-only, not district-specific.
        state_ev_care=7500,
        state_ev_non_care=0,
    ),
}


REGIMES: tuple[str, ...] = tuple(SCHEDULES.keys())


# -----------------------------------------------------------------------------
# Single entrypoint
# -----------------------------------------------------------------------------

def compute_net_capex(
    pv_kw: float, battery_kwh: float,
    has_ev: bool, has_hp: bool,
    regime: str, is_care: bool,
    capex_table: dict,
) -> tuple[float, dict[str, float]]:
    """Compute net capex after subsidies for one (bundle, regime, tier).

    Args:
        pv_kw         : PV system size in kW DC; 0 if not in bundle
        battery_kwh   : battery capacity in kWh; 0 if not in bundle
        has_ev        : True if bundle includes EV (premium + L2 charger)
        has_hp        : True if bundle includes whole-home HP
                        (= HP space + HPWH + induction + panel; matches Upgrade 11)
        regime        : one of REGIMES
        is_care       : True if household qualifies for CARE pricing
                        (income_category == 'Low' proxy)
        capex_table   : config.CAPEX (gross capex values per tech)

    Returns:
        (net_capex_after_subsidies, itemized_rebates_dict)
        Net capex is clamped at 0 (utility doesn't pay you to install).
    """
    if regime not in SCHEDULES:
        raise ValueError(
            f"unknown regime {regime!r}; choose from {REGIMES}")
    sched = SCHEDULES[regime]
    rebates: dict[str, float] = {}

    pv_gross = pv_kw * capex_table["pv_per_kw"]
    batt_gross = battery_kwh * capex_table["battery_per_kwh"]

    # ---- PV ITC ----
    if pv_kw > 0 and sched.pv_itc_pct > 0:
        rebates["pv_itc"] = sched.pv_itc_pct * pv_gross

    # ---- Battery ITC + SGIP (tier-aware) ----
    if battery_kwh > 0:
        if sched.battery_itc_pct > 0:
            rebates["battery_itc"] = sched.battery_itc_pct * batt_gross
        if is_care and sched.sgip_battery_equity_per_kwh > 0:
            rebates["sgip_battery"] = (
                battery_kwh * sched.sgip_battery_equity_per_kwh)
        elif not is_care and sched.sgip_battery_general_per_kwh > 0:
            kwh_eligible = min(
                battery_kwh, sched.sgip_battery_general_cap_kwh)
            rebates["sgip_battery"] = (
                kwh_eligible * sched.sgip_battery_general_per_kwh)

    # ---- HP (whole-home Upgrade 11: HP space + HPWH + induction + panel) ----
    if has_hp:
        if sched.fed_25c_hp_max > 0:
            rebates["fed_25c_hp"] = sched.fed_25c_hp_max
        if sched.fed_25c_hpwh_max > 0:
            rebates["fed_25c_hpwh"] = sched.fed_25c_hpwh_max
        if sched.fed_25c_panel_max > 0:
            rebates["fed_25c_panel"] = sched.fed_25c_panel_max
        state_hp = (sched.state_hp_space_care if is_care
                    else sched.state_hp_space_non_care)
        if state_hp > 0:
            rebates["state_hp_space"] = state_hp
        state_hpwh = (sched.state_hpwh_care if is_care
                      else sched.state_hpwh_non_care)
        if state_hpwh > 0:
            rebates["state_hpwh"] = state_hpwh
        state_ind = (sched.state_induction_care if is_care
                     else sched.state_induction_non_care)
        if state_ind > 0:
            rebates["state_induction"] = state_ind

    # ---- EV ----
    if has_ev:
        if sched.fed_30d_ev > 0:
            rebates["fed_30d_ev"] = sched.fed_30d_ev
        state_ev = (sched.state_ev_care if is_care
                    else sched.state_ev_non_care)
        if state_ev > 0:
            rebates["state_ev"] = state_ev

    # ---- Gross capex ----
    gross = pv_gross + batt_gross
    if has_ev:
        gross += capex_table["ev_premium"] + capex_table["ev_charger"]
    if has_hp:
        gross += (capex_table["heat_pump_space"]
                  + capex_table["heat_pump_water"]
                  + capex_table["induction_range"]
                  + capex_table["panel_upgrade_200a"])

    total_rebates = sum(rebates.values())
    net_capex = max(0.0, gross - total_rebates)
    return net_capex, rebates
