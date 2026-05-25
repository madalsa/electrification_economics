"""Shared paths, constants, and economic assumptions.

All numeric assumptions are sourced; see comments. Last verified: 2026-05-06.
Major 2026 policy changes (vs prior years) are flagged "POLICY 2026:".

Reads pipeline outputs from the parent `california_rates` repo so we don't
duplicate data. When extracted to a standalone repo, point CR_ROOT at the
local clone.
"""

from pathlib import Path
import os

EE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EE_ROOT / "data"

# Parent pipeline inputs (rate sheets, TOU weights, EEC hourly, metadata
# parquets, PUMA mapping). Default: EE_ROOT itself (standalone layout
# where these CSVs/Excels sit alongside the EE folder). Override via
# env var EE_PARENT_DIR when EE is embedded inside the california_rates
# repo (historical layout, where parent files live one directory up).
CR_ROOT = Path(os.environ.get("EE_PARENT_DIR", str(EE_ROOT)))

# Hourly parquet directories (Baseline_<U>/ and Upgrade11_<U>/) may live
# on slow / remote storage (e.g., a Google Drive mount via rclone) while
# the in-repo CSVs stay local. Override EE_PARQUET_DIR independently of
# EE_PARENT_DIR to point at the mounted location.
PARQUET_ROOT = Path(os.environ.get("EE_PARQUET_DIR", str(CR_ROOT)))

PIPELINE_OUTPUTS = {
    "pge": {
        "baseline_bills": CR_ROOT / "baseline_bills_pge_fresh.csv",
        "post_adoption":  CR_ROOT / "post_adoption_bills_pge.csv",
        "summary":        CR_ROOT / "pipeline_summary_pge.csv",
        "rate_scenarios": CR_ROOT / "rate_scenarios_pge.csv",
        "tou_weights":    CR_ROOT / "tou_weights_pge.csv",
        "baseline_parquets": PARQUET_ROOT / "Baseline_PGE",
        "upgrade11_parquets": PARQUET_ROOT / "Upgrade11_PGE",
    },
    "sce": {
        "baseline_bills": CR_ROOT / "baseline_bills_sce_fresh.csv",
        "post_adoption":  CR_ROOT / "post_adoption_bills_sce.csv",
        "summary":        CR_ROOT / "pipeline_summary_sce.csv",
        "rate_scenarios": CR_ROOT / "rate_scenarios_sce.csv",
        "tou_weights":    CR_ROOT / "tou_weights_sce.csv",
        "baseline_parquets": PARQUET_ROOT / "Baseline_SCE",
        "upgrade11_parquets": PARQUET_ROOT / "Upgrade11_SCE",
    },
    "sdge": {
        "baseline_bills": CR_ROOT / "baseline_bills_sdge_fresh.csv",
        "post_adoption":  CR_ROOT / "post_adoption_bills_sdge.csv",
        "summary":        CR_ROOT / "pipeline_summary_sdge.csv",
        "rate_scenarios": CR_ROOT / "rate_scenarios_sdge.csv",
        "tou_weights":    CR_ROOT / "tou_weights_sdge.csv",
        "baseline_parquets": PARQUET_ROOT / "Baseline_SDGE",
        "upgrade11_parquets": PARQUET_ROOT / "Upgrade11_SDGE",
    },
}

# ResStock CA building metadata + annual electricity / NG / peak kW results.
# The `_tmy_` variant bundles the annual-results columns we need
# (out.electricity.*, out.natural_gas.*, peak kW); the plain
# `metadata_rescaled` parquet has metadata only.
METADATA_PARQUET = CR_ROOT / "CA_baseline_tmy_metadata_and_annual_results.parquet"
RASS_SURVEY = CR_ROOT / "Final19_SW_CleanedSurvey.csv"
PUMA_UTILITY = CR_ROOT / "puma_utility_data.csv"

# -----------------------------------------------------------------------------
# Population scoping for the paper
# -----------------------------------------------------------------------------
# Include only IOU customers where NPV is a meaningful question.
# Exclude:
#   - POU territories (LADWP, SMUD, IID, etc.) - handled upstream by
#     PUMA-utility mapping; only PGE / SCE / SDGE carried through pipeline.
#   - EBD-eligible households (<=80% AMI in CEC priority climate zones).
#     They receive turnkey free retrofit; payback / NPV not applicable.
#     They are reported as a population share in summary stats but
#     excluded from the optimization population.
#
# CEC EBD priority climate zones - TODO: verify against current CEC list.
# Initial list based on DAC overlap and high-heat / poor-AQ CZs.
EBD_PRIORITY_CEC_CZS = {8, 9, 10, 13, 14, 15}

# Income-tier threshold: <=80% AMI defined by HUD-published county income
# limits; in ResStock metadata typically encoded as `income_tier_pct_ami`
# or two-income variant.
EBD_AMI_THRESHOLD = 0.80

INCLUDED_UTILITIES = ("pge", "sce", "sdge")


def is_ebd_eligible(cec_cz: int, ami_pct: float) -> bool:
    """True if household is excluded from the analysis as EBD-eligible."""
    return (cec_cz in EBD_PRIORITY_CEC_CZS) and (ami_pct <= EBD_AMI_THRESHOLD)

# -----------------------------------------------------------------------------
# Financial framework
# -----------------------------------------------------------------------------
# CPUC E3 ACC and DER cost-effectiveness work uses 3-5% real for customer
# perspective. Use 5% real as base; sweep 3-8% in sensitivity.
# Source: CPUC 2024 ACC Documentation v1b.
DISCOUNT_RATE_REAL = 0.05
INFLATION = 0.025
DISCOUNT_RATE_NOMINAL = (1 + DISCOUNT_RATE_REAL) * (1 + INFLATION) - 1

# 20-yr horizon matches NEM 2.0 grandfathering and PV manufacturer warranty;
# sensitivity 15 / 25 yr.
ANALYSIS_YEARS = 20

# Real bill escalator: CPUC GRC filings typically project 2-2.5%/yr real;
# 2020-2024 actuals exceeded inflation. Base 2% real; sweep 0-4%.
# Source: Cal Advocates Q1 2025 Rates Report; SCE 2025 GRC.
BILL_ESCALATOR_REAL = 0.02

# -----------------------------------------------------------------------------
# CAPEX (CA installed, 2026 averages, pre-incentive)
# -----------------------------------------------------------------------------
# Sources: EnergySage CA 2026 marketplace; SolarReviews 2026; Reliable HVAC
# LA/Ventura 2026; Today's Homeowner 2026; Custom Home Bay Area 2026;
# Expert Electric Group CA 2026; Cox Auto / KBB Mar 2026 ATP.
CAPEX = {
    "pv_per_kw":         2500,   # $/kW DC turnkey; range $2,400-$2,600 (CA)
    "battery_per_kwh":   1050,   # installed; Powerwall 3 ~$1,000-$1,100/kWh
    "ev_premium":        5800,   # KBB Mar 2026 ATP: EV $54.5K vs $49.3K avg
    "ev_charger":        1500,   # L2 EVSE installed; range $800-$2,700
    "heat_pump_space":  15000,   # 3-ton ducted CA install; range $12-18K
    "heat_pump_water":   5500,   # 50-80 gal HPWH installed; range $4-8K
    "induction_range":   3500,   # range + 240V install; SF Bay $2.5-6K
    "panel_upgrade_200a": 3500,  # 200A service upgrade; range $2-4.5K
}

# Inverter replacement cost at year ~13 for string inverters; Enphase
# microinverters typically last full PV life (set to 0 if microinverter).
INVERTER_REPLACEMENT_COST = 2500
INVERTER_REPLACEMENT_YEAR = 13

# -----------------------------------------------------------------------------
# Subsidy stack — see src/subsidies.py
# -----------------------------------------------------------------------------
# All capex-subsidy assumptions (federal ITC, state programs, tier rules)
# live in src/subsidies.py as a single SubsidySchedule table. Three
# regimes: 2024_federal, 2026_federal, 2026_ca_added. No district detail,
# no DAC, no AMI sub-tiers — CARE vs Non-CARE only.

# -----------------------------------------------------------------------------
# FUEL PRICES (CA, 2026)
# -----------------------------------------------------------------------------
# Source: CEC gasoline breakdown, EIA SCA weekly retail; PG&E and SoCalGas
# Jan 2026 rate advisories.
GAS_PRICE_DEFAULT = 4.90        # $/gal CA regular YTD 2026 baseline
GAS_PRICE_RANGE = (3.50, 6.50)  # sweep range for sensitivity

# Residential natural gas $/therm bundled non-CARE, Jan 2026.
# Use utility-specific values when available.
NG_THERM_PRICE = {
    "pge":       2.92,
    "socalgas":  2.08,   # used for SCE territory residential gas
    "sdge":      2.10,   # SDGE gas via SoCalGas-tariff structure
    "default":   2.40,
}

# CARE customers receive a ~20% discount on residential natural gas
# (PG&E gas CARE program, SoCalGas CARE, SDG&E gas CARE all roughly
# 20% off the non-CARE bundled rate as of 2026). Applied to the
# utility's $/therm above when household is_care=True. Source: PGE
# CARE program guide, SoCalGas CARE schedules.
GAS_CARE_DISCOUNT = 0.20

# -----------------------------------------------------------------------------
# EV / VEHICLE
# -----------------------------------------------------------------------------
# Source: Recurrent 2026 efficiency rankings; EPA 2024 Auto Trends Report;
# Caltrans CA VMT (2023 latest).
EV_EFFICIENCY = {
    "sedan":     4.0,    # mi/kWh real-world; range 4.0-4.3 (Model 3, Lucid)
    "crossover": 3.3,    # range 3.0-3.6 (Model Y, Leaf 2026)
    "default":   3.5,
}
ICE_MPG = {
    "sedan":     32,
    "crossover": 27,
    "default":   28,     # combined real-world; CAFE values run higher
}
VMT_DEFAULT = 12000              # mi/yr per household; CA per-vehicle ~10.2K
VMT_GRID = [5000, 8000, 12000, 15000, 20000, 25000]

# -----------------------------------------------------------------------------
# PV / Battery lifetime
# -----------------------------------------------------------------------------
# Sources: NREL ATB 2024b; Tesla Powerwall warranty; Enphase IQ warranty.
PV_LIFE_YEARS = 30
BATTERY_LIFE_YEARS = 15        # NREL ATB 2024b residential battery
BATTERY_CYCLE_WARRANTY = 3000   # Powerwall 3 / 70% retention at 10 yr

# -----------------------------------------------------------------------------
# Export rates / EEC (Energy Export Compensation)
# -----------------------------------------------------------------------------
# POLICY: All new interconnections after Apr 15, 2023 on Net Billing
# Tariff (NBT, "NEM 3.0"). Compensation = utility-published hourly EEC
# values from MIDAS, NOT a flat rate. We use the actual hourly file
# already in the parent repo:
#   ../eec_hourly_2025.csv          (datetime, utility, eec_total $/kWh)
#   ../LY2025 NBT Pricing Upload MIDAS.csv  (CPUC source data)
# Source: CPUC NEM/NBT; PG&E Solar Billing Plan EEC Price Sheet 2025.
EEC_HOURLY_CSV = CR_ROOT / "eec_hourly_2025.csv"
EEC_MIDAS_CSV = CR_ROOT / "LY2025 NBT Pricing Upload MIDAS.csv"

# Annual averages from eec_hourly_2025.csv (computed 2026-05-06):
EEC_ANNUAL_AVG = {
    "pge":   0.0968,   # $/kWh
    "sce":   0.0853,
    "sdge":  0.0782,
}

# Counterfactual export-rate scenarios for policy sensitivity.
# Use these only when overriding the hourly EEC file (e.g., NEM 2.0
# grandfathered customers, or "what-if" full retail).
NEM2_EXPORT_AVG = 0.32    # blended retail offset for NEM 2.0 customers
EXPORT_RATE_FLAT_SWEEP = [0.05, 0.07, 0.10, 0.15, 0.32]
EXPORT_REGIME = ("nbt_hourly", "nem2_retail", "flat_5c", "flat_15c")

# -----------------------------------------------------------------------------
# Sizing search grid (sizing_optimizer.py)
# -----------------------------------------------------------------------------
PV_KW_GRID = [0, 2, 4, 6, 8, 10, 12, 15]
BATT_KWH_GRID = [0, 5, 10, 15, 20, 30]


def utility_paths(utility: str) -> dict:
    return PIPELINE_OUTPUTS[utility.lower()]


def gas_price(utility: str) -> float:
    """Residential NG $/therm by utility."""
    key = {"pge": "pge", "sce": "socalgas", "sdge": "sdge"}.get(
        utility.lower(), "default")
    return NG_THERM_PRICE[key]


def assert_safe_out_dir(path) -> Path:
    """Refuse to write outside the EE data folder.

    Protects the parent california_rates pipeline outputs from accidental
    overwrite when this module is run from `california_rates/` (which it
    must be, for parent imports to work in sizing_optimizer_hourly.py).
    Any --out-dir argument that escapes EE_ROOT/data raises ValueError.
    """
    resolved = Path(path).resolve()
    expected = (EE_ROOT / "data").resolve()
    try:
        resolved.relative_to(expected)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside electrification_economics/data/. "
            f"Got: {resolved}\n"
            f"This guard exists to prevent accidental overwrite of the "
            f"parent california_rates pipeline outputs.") from exc
    return resolved
