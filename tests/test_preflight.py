"""Tests for preflight check primitives.

The actual file-presence checks are run against the live environment in
the smoke test at the bottom; the unit tests here exercise the pure
logic (Check formatting, run-plan projection).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, preflight as pf


def test_check_constants_known():
    assert pf.PASS == "PASS"
    assert pf.WARN == "WARN"
    assert pf.FAIL == "FAIL"


def test_check_dataclass_line_format():
    c = pf.Check("foo", pf.PASS)
    assert "PASS" in c.line() and "foo" in c.line()
    c2 = pf.Check("bar", pf.FAIL, "missing file")
    line = c2.line()
    assert "FAIL" in line and "bar" in line and "missing file" in line


def test_metadata_required_columns_non_empty():
    """build_features will crash without these columns; the contract list
    must not be silently emptied."""
    assert len(pf.METADATA_REQUIRED) >= 10
    # specific high-risk ones we must always check for
    for col in ("out.electricity.total.energy_consumption.kwh",
                "in.cec_climate_zone", "in.county_and_puma"):
        assert col in pf.METADATA_REQUIRED


def test_metadata_optional_includes_new_shape_features():
    """When we added the shape features, preflight must know to warn
    about their absence (else silent zero-shares)."""
    for col in ("out.electricity.hot_water.energy_consumption.kwh",
                "out.electricity.plug_loads.energy_consumption.kwh"):
        assert col in pf.METADATA_OPTIONAL


def test_metadata_optional_includes_hp_gas_endusers():
    """upgrade11_economics needs gas end-uses; preflight must warn if
    they're missing rather than fail mid-stage-4."""
    for col in ("out.natural_gas.heating.energy_consumption.kwh",
                "out.natural_gas.hot_water.energy_consumption.kwh",
                "out.natural_gas.range_oven.energy_consumption.kwh"):
        assert col in pf.METADATA_OPTIONAL


def test_tou_periods_required_minimum():
    """Without summer/winter peak + offpeak no TOU rate can be evaluated."""
    assert {"summer_peak", "summer_offpeak",
            "winter_peak", "winter_offpeak"}.issubset(
        pf.TOU_PERIODS_REQUIRED)


def test_project_run_plan_returns_lines():
    lines = pf.project_run_plan(["pge"])
    assert any("representative_buildings" in line for line in lines)
    assert any("run_npv" in line for line in lines)
    bundle_line = next(line for line in lines if "run_npv" in line)
    # The run_npv plan line lists row/cell counts
    assert any(w in bundle_line.lower() for w in ("row", "lp", "rows"))


def test_run_all_checks_returns_list():
    """Doesn't matter that parent files may be missing in this env;
    the function should return a list of Check objects either way."""
    out = pf.run_all_checks(["pge"])
    assert isinstance(out, list)
    assert all(isinstance(c, pf.Check) for c in out)
    # output_dir check is always producible (we control DATA_DIR)
    assert any("output dir" in c.name for c in out)


def test_check_output_dir_passes():
    """The EE data dir is always writable in tests."""
    out = pf.check_output_dir()
    assert len(out) == 1
    assert out[0].status in (pf.PASS, pf.WARN)


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
