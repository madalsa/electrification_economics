"""NPV / cashflow / payback math. Subsidies live in src/subsidies.py.

Pure financial helpers — no capex stack, no incentive context, no
EV-specific premium logic. Bills come from src/bill.py; subsidies
from src/subsidies.py; this module just does the discounted-cashflow
arithmetic that ties annual savings + net capex into an NPV.

Conventions:
    - Year 0 = upfront capex (negative cashflow).
    - Years 1..N = annual savings (positive cashflow).
    - All flows in real (inflation-adjusted) dollars by default.
    - Discount rate: real if flows are real (config.DISCOUNT_RATE_REAL).
"""

from __future__ import annotations

from math import inf
from typing import Optional, Sequence

from . import config


def annual_cashflow_series(
    annual_savings_year1: float,
    years: int = config.ANALYSIS_YEARS,
    escalator_real: float = config.BILL_ESCALATOR_REAL,
    midlife_replacement_year: Optional[int] = None,
    midlife_replacement_cost: float = 0.0,
) -> list[float]:
    """Real-dollar annual cashflows. Bill savings rise at escalator_real.

    Optional midlife replacement (e.g. PV inverter at year 13) is
    subtracted in that one year.
    """
    flows = []
    for t in range(1, years + 1):
        cf = annual_savings_year1 * ((1 + escalator_real) ** (t - 1))
        if midlife_replacement_year and t == midlife_replacement_year:
            cf -= midlife_replacement_cost
        flows.append(cf)
    return flows


def npv(
    cashflows: Sequence[float],
    discount_rate: float = config.DISCOUNT_RATE_REAL,
    capex: float = 0.0,
) -> float:
    """Net present value. capex is paid at year 0 (subtracted)."""
    pv = -capex
    for t, cf in enumerate(cashflows, start=1):
        pv += cf / ((1 + discount_rate) ** t)
    return pv


def simple_payback(capex: float, annual_savings: float) -> float:
    """Years to recover capex assuming flat year-1 savings. inf if never."""
    if annual_savings <= 0 or capex <= 0:
        return inf
    return capex / annual_savings


def discounted_payback(
    capex: float,
    cashflows: Sequence[float],
    discount_rate: float = config.DISCOUNT_RATE_REAL,
) -> float:
    """Years to recover capex on a discounted basis. inf if never."""
    if capex <= 0:
        return 0.0
    cumulative = 0.0
    for t, cf in enumerate(cashflows, start=1):
        cumulative += cf / ((1 + discount_rate) ** t)
        if cumulative >= capex:
            prev = cumulative - cf / ((1 + discount_rate) ** t)
            frac = (capex - prev) / (cumulative - prev)
            return (t - 1) + frac
    return inf


def levelized_annual_cost(
    capex: float,
    cashflows: Sequence[float],
    discount_rate: float = config.DISCOUNT_RATE_REAL,
) -> float:
    """Equivalent annual cost (annuity equivalent of NPV)."""
    n = len(cashflows)
    pv_savings = sum(cf / ((1 + discount_rate) ** t)
                     for t, cf in enumerate(cashflows, start=1))
    net_pv = pv_savings - capex
    if discount_rate == 0:
        return net_pv / n
    crf = (discount_rate * (1 + discount_rate) ** n) / (
        (1 + discount_rate) ** n - 1)
    return net_pv * crf
