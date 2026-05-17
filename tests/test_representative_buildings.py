"""Smoke tests for representative_buildings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pandas as pd

from src import config, representative_buildings as rb


def test_ami_bin_mapping():
    assert rb.AMI_BIN_TO_FRAC["0-30%"] == 0.15
    assert rb.AMI_BIN_TO_FRAC["60-80%"] == 0.70
    assert rb.AMI_BIN_TO_FRAC["150%+"] == 2.00
    assert math.isnan(rb.AMI_BIN_TO_FRAC["Not Available"])


def test_ebd_threshold_consistent():
    """Bins <=80% AMI midpoints should all flag <=EBD threshold."""
    for bin_name in ("0-30%", "30-60%", "60-80%"):
        assert rb.AMI_BIN_TO_FRAC[bin_name] <= config.EBD_AMI_THRESHOLD
    for bin_name in ("80-100%", "100-120%", "120-150%", "150%+"):
        assert rb.AMI_BIN_TO_FRAC[bin_name] > config.EBD_AMI_THRESHOLD


def test_vintage_decade_covers_all_known():
    expected = {"<1940", "1940s", "1950s", "1960s", "1970s",
                "1980s", "1990s", "2000s", "2010s"}
    assert expected.issubset(set(rb.VINTAGE_DECADE.keys()))


def test_feature_cols_include_shape_features():
    """Load-shape features must be in FEATURE_COLS so rate-design-sensitive
    archetypes get differentiated within stratum."""
    for col in ("hot_water_share", "plug_loads_share",
                "peakiness_summer", "peakiness_winter"):
        assert col in rb.FEATURE_COLS, col


def test_build_features_adds_shape_columns():
    """build_features must compute the new shape columns even when the
    underlying end-use columns aren't present in the metadata (defaults to 0).
    """
    import numpy as np
    df = pd.DataFrame({
        "out.electricity.total.energy_consumption.kwh": [10000.0, 5000.0],
        "out.natural_gas.total.energy_consumption.kwh": [0, 20000],
        "out.electricity.cooling.energy_consumption.kwh": [2000, 0],
        "out.electricity.heating.energy_consumption.kwh": [500, 1000],
        "out.electricity.hot_water.energy_consumption.kwh": [800, 400],
        "out.electricity.plug_loads.energy_consumption.kwh": [3000, 1500],
        "out.electricity.summer.peak.kw": [6.0, 3.5],
        "out.electricity.winter.peak.kw": [4.0, 4.5],
        "in.sqft": [1800, 1200],
        "in.vintage": ["1970s", "2000s"],
        "in.area_median_income": ["100-120%", "80-100%"],
        "in.geometry_building_type_recs": ["Single-Family Detached",
                                            "Single-Family Detached"],
        "in.heating_fuel": ["Natural Gas", "Natural Gas"],
    })
    out = rb.build_features(df)
    # Hot water + plug loads shares
    assert math.isclose(out["hot_water_share"].iloc[0], 0.08, abs_tol=1e-3)
    assert math.isclose(out["plug_loads_share"].iloc[0], 0.30, abs_tol=1e-3)
    # Peakiness = peak_kw / (annual_kwh/8760)
    mean_kw_0 = 10000 / 8760
    assert math.isclose(out["peakiness_summer"].iloc[0],
                        6.0 / mean_kw_0, rel_tol=1e-3)
    assert math.isclose(out["peakiness_winter"].iloc[0],
                        4.0 / mean_kw_0, rel_tol=1e-3)


def test_build_features_robust_to_missing_optional_columns():
    """If hot_water or plug_loads end-use columns aren't in metadata,
    build_features should default them to 0 rather than crash."""
    df = pd.DataFrame({
        "out.electricity.total.energy_consumption.kwh": [10000.0],
        "out.natural_gas.total.energy_consumption.kwh": [0],
        "out.electricity.cooling.energy_consumption.kwh": [2000],
        "out.electricity.heating.energy_consumption.kwh": [500],
        "out.electricity.summer.peak.kw": [6.0],
        "out.electricity.winter.peak.kw": [4.0],
        "in.sqft": [1800],
        "in.vintage": ["2000s"],
        "in.area_median_income": ["100-120%"],
        "in.geometry_building_type_recs": ["Single-Family Detached"],
        "in.heating_fuel": ["Natural Gas"],
    })
    out = rb.build_features(df)
    assert out["hot_water_share"].iloc[0] == 0
    assert out["plug_loads_share"].iloc[0] == 0


def test_build_features_zero_kwh_safe():
    """Buildings with zero annual_kwh must not raise; all shares -> 0."""
    df = pd.DataFrame({
        "out.electricity.total.energy_consumption.kwh": [0.0],
        "out.natural_gas.total.energy_consumption.kwh": [0],
        "out.electricity.cooling.energy_consumption.kwh": [0],
        "out.electricity.heating.energy_consumption.kwh": [0],
        "out.electricity.summer.peak.kw": [0.0],
        "out.electricity.winter.peak.kw": [0.0],
        "in.sqft": [1200],
        "in.vintage": ["2000s"],
        "in.area_median_income": ["100-120%"],
        "in.geometry_building_type_recs": ["Single-Family Detached"],
        "in.heating_fuel": ["None"],
    })
    out = rb.build_features(df)
    for col in ("cooling_share", "hvac_share", "hot_water_share",
                "plug_loads_share", "peakiness_summer", "peakiness_winter"):
        assert out[col].iloc[0] == 0, col


def test_output_artifact_present_after_run():
    """If user has run the script, artifact should exist; else skip."""
    p = config.DATA_DIR / "representative_buildings.parquet"
    if not p.exists():
        return  # not run yet; this is a smoke test, not a precondition
    df = pd.read_parquet(p)
    assert len(df) > 0
    # All medoids must be in scope utilities
    assert set(df["utility"].str.lower()).issubset(
        set(config.INCLUDED_UTILITIES))
    # No EBD-eligible should remain
    df["cec_cz_int"] = df["cec_cz"].astype(int)
    df["ami_frac"] = df["ami_bin"].map(rb.AMI_BIN_TO_FRAC)
    is_ebd = (df["ami_frac"] <= config.EBD_AMI_THRESHOLD) & (
        df["cec_cz_int"].isin(config.EBD_PRIORITY_CEC_CZS))
    assert not is_ebd.any(), "EBD-eligible rows leaked into representatives"


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
