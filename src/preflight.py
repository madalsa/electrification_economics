"""Preflight: validate prerequisites before running run_npv.

Catches the failure modes that otherwise show up mid-run:
  - Parent california_rates files missing / renamed
  - ResStock metadata missing columns build_features expects (esp. the
    optional shape-feature end-uses like plug_loads / hot_water)
  - TOU weights CSV missing a utility or missing required periods
  - EEC hourly file missing a utility or row count != 8760
  - PUMA-utility mapping incomplete
  - Output directory not writable, or insufficient disk space
  - Stage outputs partially present (sizing without rates, etc.) so we
    can tell the user exactly which stages need to be (re-)run

Outputs a structured report: PASS / WARN / FAIL per check, plus a
"run plan" showing expected row counts and rough runtime per stage.
Exits with non-zero status if any FAIL.

Run BEFORE `python -m src.run_npv`.

CLI:
    python -m electrification_economics.src.preflight \
        [--utilities pge sce sdge] [--strict]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str            # PASS / WARN / FAIL
    detail: str = ""

    def line(self) -> str:
        tag = {PASS: "  PASS", WARN: "  WARN", FAIL: "  FAIL"}[self.status]
        return f"{tag}  {self.name}" + (
            f"\n        {self.detail}" if self.detail else "")


# -----------------------------------------------------------------------------
# Required ResStock metadata columns
# -----------------------------------------------------------------------------

# Hard requirements: build_features.py crashes without these.
METADATA_REQUIRED = [
    "out.electricity.total.energy_consumption.kwh",
    "out.natural_gas.total.energy_consumption.kwh",
    "out.electricity.cooling.energy_consumption.kwh",
    "out.electricity.heating.energy_consumption.kwh",
    "out.electricity.summer.peak.kw",
    "out.electricity.winter.peak.kw",
    "in.sqft", "in.vintage", "in.area_median_income",
    "in.geometry_building_type_recs", "in.heating_fuel",
    "in.cec_climate_zone", "in.tenure", "in.county_and_puma",
]

# Optional: build_features defaults to 0 if missing, but shape-feature
# clustering loses signal. Warn rather than fail.
METADATA_OPTIONAL = [
    "out.electricity.hot_water.energy_consumption.kwh",
    "out.electricity.plug_loads.energy_consumption.kwh",
    # upgrade11_economics needs these end-uses for the HP delta calc
    "out.natural_gas.heating.energy_consumption.kwh",
    "out.natural_gas.hot_water.energy_consumption.kwh",
    "out.natural_gas.range_oven.energy_consumption.kwh",
]

# TOU period columns the rate-design + sizing modules expect.
TOU_PERIODS_REQUIRED = {"summer_peak", "summer_offpeak",
                         "winter_peak", "winter_offpeak"}


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_parent_metadata() -> list[Check]:
    path = config.METADATA_PARQUET
    if not path.exists():
        return [Check("ResStock metadata parquet", FAIL,
                      f"missing: {path}\n        EE stage 0 cannot run.")]
    try:
        cols = pd.read_parquet(path, columns=None).columns
    except Exception as exc:
        return [Check("ResStock metadata parquet", FAIL,
                      f"unreadable ({type(exc).__name__}): {exc}")]
    checks = [Check("ResStock metadata parquet", PASS, str(path))]
    missing_req = [c for c in METADATA_REQUIRED if c not in cols]
    if missing_req:
        checks.append(Check(
            "metadata required columns", FAIL,
            f"missing {len(missing_req)} column(s): "
            + ", ".join(missing_req[:5])
            + ("..." if len(missing_req) > 5 else "")))
    else:
        checks.append(Check("metadata required columns", PASS,
                            f"all {len(METADATA_REQUIRED)} present"))
    missing_opt = [c for c in METADATA_OPTIONAL if c not in cols]
    if missing_opt:
        checks.append(Check(
            "metadata optional columns", WARN,
            f"{len(missing_opt)} optional column(s) absent — shape features "
            f"and/or HP module will default-to-zero:\n        "
            + "\n        ".join(missing_opt)))
    else:
        checks.append(Check("metadata optional columns", PASS,
                            f"all {len(METADATA_OPTIONAL)} present"))
    return checks


def check_puma_utility() -> list[Check]:
    path = config.PUMA_UTILITY
    if not path.exists():
        return [Check("PUMA-utility mapping", FAIL, f"missing: {path}")]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [Check("PUMA-utility mapping", FAIL,
                      f"unreadable: {exc}")]
    required = {"PUMA", "utility_acronym", "utility_type", "climate_zone"}
    missing = required - set(df.columns)
    if missing:
        return [Check("PUMA-utility mapping", FAIL,
                      f"missing columns: {missing}")]
    utes = set(df["utility_acronym"].str.lower().dropna().unique())
    not_covered = set(config.INCLUDED_UTILITIES) - utes
    if not_covered:
        return [Check("PUMA-utility mapping", FAIL,
                      f"utility coverage gap: {not_covered}")]
    return [Check("PUMA-utility mapping", PASS,
                  f"{len(df):,} rows; utilities: "
                  + ", ".join(sorted(utes & set(config.INCLUDED_UTILITIES))))]


def check_tou_weights(utilities: list[str]) -> list[Check]:
    checks = []
    for u in utilities:
        path = config.CR_ROOT / f"tou_weights_{u}.csv"
        if not path.exists():
            checks.append(Check(f"tou_weights_{u}", FAIL,
                                f"missing: {path}"))
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            checks.append(Check(f"tou_weights_{u}", FAIL,
                                f"unreadable: {exc}"))
            continue
        if not {"period", "weight"}.issubset(df.columns):
            checks.append(Check(f"tou_weights_{u}", FAIL,
                                f"missing period/weight columns"))
            continue
        periods = set(df["period"])
        missing_periods = TOU_PERIODS_REQUIRED - periods
        if missing_periods:
            checks.append(Check(
                f"tou_weights_{u}", FAIL,
                f"missing required periods: {missing_periods}"))
        else:
            checks.append(Check(
                f"tou_weights_{u}", PASS,
                f"{len(df)} period rows; sum(weight)={df['weight'].sum():.3f}"))
    return checks


def check_rate_scenarios(utilities: list[str]) -> list[Check]:
    checks = []
    for u in utilities:
        path = config.CR_ROOT / f"rate_scenarios_{u}_fresh.csv"
        if not path.exists():
            checks.append(Check(f"rate_scenarios_{u}_fresh", FAIL,
                                f"missing: {path}"))
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            checks.append(Check(f"rate_scenarios_{u}_fresh", FAIL,
                                f"unreadable: {exc}"))
            continue
        required_cols = {"Scenario", "Fixed_CARE", "Fixed_NonCARE",
                          "Fixed_Pct_TD", "Remove_Wildfire",
                          "ROE_Reduction"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            checks.append(Check(
                f"rate_scenarios_{u}_fresh", FAIL,
                f"missing required columns: {missing_cols}"))
            continue
        # Sanity: parent rate designer emits 40 scenarios per utility
        # (F{0,25,50,75,100} x WF{0,1} x ROE{0,0.5,1.0,1.5}).
        if len(df) != 40:
            checks.append(Check(
                f"rate_scenarios_{u}_fresh", WARN,
                f"{len(df)} scenarios (expected 40)"))
        else:
            checks.append(Check(
                f"rate_scenarios_{u}_fresh", PASS,
                f"40 scenarios; tier-graduated fixed-charge columns "
                f"present"))
    return checks


def check_eec_hourly(utilities: list[str]) -> list[Check]:
    path = config.EEC_HOURLY_CSV
    if not path.exists():
        return [Check("EEC hourly", FAIL, f"missing: {path}")]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [Check("EEC hourly", FAIL, f"unreadable: {exc}")]
    # Wide-format schema: datetime + one *_total column per utility
    # (pge_total / sce_total / sdge_total). bill.load_hourly_eec
    # reads these directly.
    required_cols = {"datetime"} | {f"{u}_total" for u in utilities}
    missing = required_cols - set(df.columns)
    if missing:
        return [Check("EEC hourly", FAIL,
                      f"missing columns: {missing}")]
    if len(df) != 8760:
        return [Check("EEC hourly", WARN,
                      f"{len(df)} rows (expected 8760)")]
    checks = [Check("EEC hourly", PASS,
                    f"{len(df):,} rows; per-utility _total columns present")]
    for u in utilities:
        col = f"{u}_total"
        avg = df[col].mean()
        if not (0.0 < avg < 1.0):
            checks.append(Check(f"  {col}", WARN,
                                f"avg=${avg:.4f}/kWh outside expected range"))
        else:
            checks.append(Check(f"  {col}", PASS,
                                f"avg=${avg:.4f}/kWh"))
    return checks


def check_output_dir() -> list[Check]:
    out = config.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        return [Check("output dir", FAIL, f"cannot create {out}")]
    # writability
    test = out / ".preflight_write_test"
    try:
        test.write_text("ok")
        test.unlink()
    except OSError as exc:
        return [Check("output dir", FAIL,
                      f"{out} not writable: {exc}")]
    try:
        usage = shutil.disk_usage(out)
        free_gb = usage.free / 1e9
    except OSError:
        free_gb = float("nan")
    status = WARN if free_gb < 2 else PASS
    return [Check("output dir", status,
                  f"{out} writable, {free_gb:.1f} GB free")]


def check_existing_outputs(utilities: list[str]) -> list[Check]:
    """Report which EE outputs already exist (re-running will overwrite)."""
    out = config.DATA_DIR
    expected = [
        ("medoids", ["representative_buildings.parquet",
                     "population_excluded_summary.csv"]),
        ("npv_results", ["npv_results.parquet"]),
    ]
    checks = []
    for stage, files in expected:
        present = [f for f in files if (out / f).exists()]
        if not present:
            continue
        if len(present) < len(files):
            checks.append(Check(
                f"{stage} outputs partial", WARN,
                f"{len(present)}/{len(files)} present: "
                + ", ".join(present[:3])))
        else:
            checks.append(Check(
                f"{stage} outputs present", PASS,
                f"all {len(files)} files exist (re-run will overwrite)"))
    return checks


# -----------------------------------------------------------------------------
# Run plan
# -----------------------------------------------------------------------------

def project_run_plan(utilities: list[str]) -> list[str]:
    """Estimate row counts + rough runtime for the run_npv pipeline."""
    lines = []
    n_med = None
    rep = config.DATA_DIR / "representative_buildings.parquet"
    if rep.exists():
        try:
            n_med = len(pd.read_parquet(rep, columns=["bldg_id"]))
        except Exception:
            n_med = None
    n_med_est = n_med or 2541
    n_scenarios = 40
    # 4 non-PV bundles (none, ev, hp, ev_hp) x 1 cell each = 4 cells/medoid
    # 4 PV bundles x 3 PV x 2 batt = 24 cells/medoid (each needs an LP solve)
    # Total: 28 cells/medoid x 40 scenarios x 2 subsidy regimes
    n_rows = n_med_est * 28 * n_scenarios * 2
    n_lp = n_med_est * 24 * n_scenarios   # only PV-bundle cells need LP

    lines.append(f"Plan (utilities = {' '.join(utilities)}):")
    lines.append(f"  representative_buildings: {n_med_est:,} medoid rows")
    lines.append(f"  run_npv:                  ~{n_rows:,} output rows")
    lines.append(f"                            ~{n_lp:,} LP solves (~0.5s each)")
    lines.append(f"                            estimated ~{n_lp * 0.5 / 3600:.1f} hr serial")
    return lines


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_all_checks(utilities: list[str]) -> list[Check]:
    checks = []
    checks += check_output_dir()
    checks += check_parent_metadata()
    checks += check_puma_utility()
    checks += check_tou_weights(utilities)
    checks += check_rate_scenarios(utilities)
    checks += check_eec_hourly(utilities)
    checks += check_existing_outputs(utilities)
    return checks


def main():
    ap = argparse.ArgumentParser(
        description="Validate prerequisites + show run plan for the "
                    "EE pipeline. Run BEFORE run_npv.py.")
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on WARN as well as FAIL.")
    args = ap.parse_args()

    print("=" * 70)
    print("Electrification Economics — preflight")
    print("=" * 70)
    checks = run_all_checks(args.utilities)
    for c in checks:
        print(c.line())

    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_warn = sum(1 for c in checks if c.status == WARN)
    n_pass = sum(1 for c in checks if c.status == PASS)
    print("-" * 70)
    print(f"  {n_pass} pass, {n_warn} warn, {n_fail} fail")
    print("-" * 70)

    if n_fail == 0:
        print()
        for line in project_run_plan(args.utilities):
            print(line)

    if n_fail > 0 or (args.strict and n_warn > 0):
        print()
        print("Preflight failed. Resolve above before running.")
        sys.exit(1)
    print()
    print("Preflight OK. Safe to run:")
    print("  python -m src.run_npv --limit 20      # smoke test")
    print("  python -m src.run_npv                  # full run")


if __name__ == "__main__":
    main()
