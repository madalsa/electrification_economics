"""Tests for the thin rate-extended reader.

The parent rate designer is the source of truth; this module just
re-emits its scenarios under the EE-extended schema and tacks on
EV-TOU + NBT-overlay rows. So most tests check schema preservation
and the EE-specific additions, NOT the rate values themselves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, rate_designer_extended as rde


# ============================================================================
# EV-TOU rows (don't require parent files)
# ============================================================================

def test_ev_tou_row_per_utility():
    for u in ("pge", "sce", "sdge"):
        df = rde.add_ev_only_tou_row(u)
        assert len(df) == 1, u
        assert df.iloc[0]["scenario_id"] == f"EV_TOU_{u.upper()}"
        assert df.iloc[0]["rate_type"] == "ev_submetered_tou"


def test_ev_tou_row_carries_zero_fixed_charge():
    """EV-TOU is a parallel submetered tariff; BSC enters via base rate.
    The EV-TOU row itself should not double-count fixed charges."""
    df = rde.add_ev_only_tou_row("sdge")
    assert df.iloc[0]["fixed_monthly_care"] == 0.0
    assert df.iloc[0]["fixed_monthly_non_care"] == 0.0


def test_ev_tou_super_offpeak_below_on_peak():
    for u in ("pge", "sce", "sdge"):
        df = rde.add_ev_only_tou_row(u)
        assert df.iloc[0]["ev_super_offpeak"] < df.iloc[0]["ev_on_peak"], u


# ============================================================================
# Export overlays (don't require parent files)
# ============================================================================

def test_export_regimes_are_nbt_family():
    """Post-May-2026: drop NEM2 and flat counterfactuals; only NBT
    scaling sensitivities."""
    df = rde.add_export_regime_overlays()
    assert set(df["export_regime"]) == {
        "nbt_hourly", "nbt_scaled_125", "nbt_scaled_150"}


def test_export_overlay_multipliers_match_names():
    df = rde.add_export_regime_overlays()
    by_regime = df.set_index("export_regime")["eec_multiplier"]
    assert by_regime["nbt_hourly"] == 1.0
    assert by_regime["nbt_scaled_125"] == 1.25
    assert by_regime["nbt_scaled_150"] == 1.50


# ============================================================================
# Full build_extended (requires parent rate_scenarios_<u>_fresh.csv)
# ============================================================================

def test_build_extended_has_three_rate_categories():
    df = rde.build_extended("pge")
    types = set(df["rate_type"].unique())
    assert types == {"designed_tou", "ev_submetered_tou", "export_overlay"}


def test_build_extended_preserves_all_40_designed_scenarios():
    """No filtering / no canonical-6 narrowing - all 40 from the parent
    pass through verbatim."""
    src = pd.read_csv(config.CR_ROOT / "rate_scenarios_pge_fresh.csv")
    df = rde.build_extended("pge")
    designed = df[df["rate_type"] == "designed_tou"]
    assert len(designed) == len(src) == 40


def test_build_extended_preserves_tier_fixed_charges():
    """Fixed_CARE / Fixed_NonCARE round-trip into fixed_monthly_care /
    fixed_monthly_non_care; values match the parent verbatim."""
    src = pd.read_csv(config.CR_ROOT / "rate_scenarios_pge_fresh.csv")
    df = rde.build_extended("pge")
    for scenario in ("F0_WF0_ROE0", "F25_WF0_ROE0", "F50_WF0_ROE0",
                     "F100_WF0_ROE0", "F100_WF1_ROE1.5"):
        src_row = src[src["Scenario"] == scenario].iloc[0]
        ee_row = df[df["scenario_id"] == scenario].iloc[0]
        assert ee_row["fixed_monthly_care"] == src_row["Fixed_CARE"], scenario
        assert (ee_row["fixed_monthly_non_care"]
                == src_row["Fixed_NonCARE"]), scenario


def test_care_fixed_below_non_care_at_higher_F():
    """In any non-F0 scenario the CARE fixed charge is income-graduated
    BELOW the Non-CARE charge. (F0 has both at 0; equality is allowed.)
    """
    df = rde.build_extended("sce")
    designed = df[df["rate_type"] == "designed_tou"].copy()
    for _, row in designed.iterrows():
        if row["Fixed_Pct_TD"] == 0:
            assert row["fixed_monthly_care"] == row["fixed_monthly_non_care"]
        else:
            assert (row["fixed_monthly_care"]
                    < row["fixed_monthly_non_care"]), row["scenario_id"]


def test_higher_F_means_higher_fixed_charge():
    """Within a (WF, ROE) family, F0 < F25 < F50 < F75 < F100 in fixed
    charges. Sanity check that the parent's rate designer produced a
    monotonic progression."""
    df = rde.build_extended("pge")
    designed = df[df["rate_type"] == "designed_tou"]
    family = designed[
        (designed["Remove_Wildfire"] == False) &
        (designed["ROE_Reduction"] == 0.0)
    ].sort_values("Fixed_Pct_TD")
    fixed_nc = family["fixed_monthly_non_care"].tolist()
    assert fixed_nc == sorted(fixed_nc)
    fixed_c = family["fixed_monthly_care"].tolist()
    assert fixed_c == sorted(fixed_c)


def test_designed_rows_carry_parent_metadata():
    """Fixed_Pct_TD, Remove_Wildfire, ROE_Reduction must pass through
    so downstream code can group / filter by structural axes."""
    df = rde.build_extended("sdge")
    designed = df[df["rate_type"] == "designed_tou"]
    for col in ("Fixed_Pct_TD", "Remove_Wildfire", "ROE_Reduction",
                "Scaling", "Vol_Avg", "Total_Revenue"):
        assert col in designed.columns


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
