"""Unit tests for the (now-pure) NPV math in payback_npv.py.

Subsidy/incentive logic moved to src/subsidies.py — tested separately
in tests/test_subsidies.py. This file covers only the discounted-
cashflow arithmetic.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, payback_npv as p


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_npv_zero_rate():
    flows = [100.0, 100.0, 100.0]
    assert approx(p.npv(flows, discount_rate=0.0, capex=0.0), 300.0)


def test_npv_known_value():
    expected = 100/1.05 + 100/1.05**2 + 100/1.05**3
    assert approx(p.npv([100, 100, 100], 0.05, 0.0), expected)


def test_npv_subtracts_capex():
    capex = 1000
    flows = [200] * 6
    npv_no_cap = p.npv(flows, 0.05, 0)
    npv_with_cap = p.npv(flows, 0.05, capex)
    assert approx(npv_with_cap, npv_no_cap - capex)


def test_simple_payback_basic():
    assert approx(p.simple_payback(1000, 250), 4.0)


def test_simple_payback_zero_savings_returns_inf():
    assert p.simple_payback(1000, 0) == math.inf
    assert p.simple_payback(1000, -50) == math.inf


def test_discounted_payback_longer_than_simple():
    capex = 1000
    flows = [250] * 10
    sp = p.simple_payback(capex, 250)
    dp = p.discounted_payback(capex, flows, 0.05)
    assert dp > sp


def test_discounted_payback_never_returns_inf():
    flows = [10] * 5
    assert p.discounted_payback(1000, flows, 0.05) == math.inf


def test_annual_cashflow_escalates():
    flows = p.annual_cashflow_series(100, years=3, escalator_real=0.02)
    assert approx(flows[0], 100)
    assert approx(flows[1], 100 * 1.02)
    assert approx(flows[2], 100 * 1.02 ** 2)


def test_annual_cashflow_replacement_at_year_13():
    flows = p.annual_cashflow_series(
        100, years=15, escalator_real=0.0,
        midlife_replacement_year=13,
        midlife_replacement_cost=2500)
    assert approx(flows[12], 100 - 2500)   # year 13 (index 12)
    assert approx(flows[11], 100)


def test_levelized_annual_cost_matches_npv_at_zero_rate():
    flows = [100] * 10
    lac = p.levelized_annual_cost(0, flows, discount_rate=0.0)
    # With zero discount and zero capex, LAC = mean cashflow
    assert approx(lac, 100.0)


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
