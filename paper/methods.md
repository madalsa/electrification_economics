# Methods

## Scope
- **Population**: California IOU customers (PGE, SCE, SDGE), owner-occupied, non-EBD-eligible. ~3.4M households.
- **Excluded**:
  - POU territories (LADWP, SMUD, IID, etc.) — handled by PUMA-utility mapping.
  - Renters — capex decisions belong to property owners.
  - EBD-eligible (≤80% AMI in CEC priority CZs 8/9/10/13/14/15) — receive turnkey free retrofit; NPV not applicable.

## Representative buildings (`representative_buildings.py`)
- **2,541 medoids** from stratified k-means on ResStock CA, representing 3.37M households.
- **Strata**: utility × CEC CZ × heating fuel × building type × AMI bin × vintage decade.
- **Within-stratum features** (k ≤ 8): annual_kwh, annual_therms, summer/winter peak kW, cooling/HVAC/hot_water/plug_loads end-use shares, peakiness (peak ÷ mean), sqft.
- Each medoid carries `cluster_weight` for population aggregation, plus `income_category ∈ {Low, Medium, High}` and `is_care` (= Low) derived from `in.income`.

## Bundles (`bundles.py`)
- 8 combinations: `none`, `pv_bat`, `ev`, `hp`, `pv_bat_ev`, `pv_bat_hp`, `ev_hp`, `pv_bat_ev_hp`.
- **HP** = full Upgrade-11 substitution (HP space + HPWH + induction range + 200A panel).
- **EV** = single passenger vehicle (default VMT 12,000 mi/yr).
- **PV sizing grid**: 1× / 1.5× / 3× expanded annual load (CA solar yield ~1700 kWh/kW/yr).
- **Battery sizing grid**: 13.5 / 27 kWh (Tesla Powerwall 3 × {1, 2}).

## Rate scenarios (input — not derived in EE)
- **40 designed scenarios per utility** from parent `rate_designer.py`:
  - Fixed_Pct_TD ∈ {0, 25, 50, 75, 100}
  - Remove_Wildfire ∈ {0, 1}
  - ROE_Reduction ∈ {0, 0.5, 1.0, 1.5}
  - All revenue-neutral; income-graduated via `Fixed_CARE` / `Fixed_NonCARE` columns.
- **Per-utility EV-TOU opt-in** (PGE EV2-A / SCE TOU-D-PRIME / SDGE EV-TOU-5; see `ev_tou_schedules.py`).
- **NBT export-compensation sensitivity** via `--eec-multiplier 1.0 / 1.25 / 1.50` (status quo + CPUC softening).

## Bill methodology (`bill.py` — mirrors user's `*_baseline_bills.py`)
```
grid_in       = max(net_hourly_load, 0)
grid_out      = max(-net_hourly_load, 0)
vol_bill      = sum(grid_in × hourly_rate) - baseline_credit
if is_care:   vol_bill *= (1 - care_discount)
export_credit = sum(grid_out × hourly_EEC)        # PV bundles only
fixed_annual  = (Fixed_CARE if is_care else Fixed_NonCARE) × 12
total_bill    = vol_bill + fixed_annual - export_credit
```
- **baseline_credit**: PUMA-specific allowance from retail Excel (sheet `baseline_puma`); credit applies to within-allowance imports only.
- **care_discount** (utility-specific, from retail Excel `care_discount` column): PGE 35%, SCE 32.5%, SDGE 37%.

## Income tiers + CARE
- Source: `in.income` (ResStock dollar bands).
- Mapping: `<$50K → Low` (CARE), `$50-150K → Medium`, `$150K+ → High`.
- CARE adjustments:
  - Electricity: tier-specific `Fixed_CARE` from rate sheet + utility-specific volumetric discount applied to `vol_bill`.
  - Gas: 20% discount on $/therm (`config.GAS_CARE_DISCOUNT`).

## Per-utility TOU period definitions (matched to parent `<utility>_config.py`)
| Utility | Summer | Peak | Midpeak | Periods |
|---|---|---|---|---|
| PGE  | Jun-Sep | 16-21 | none | 4 |
| SCE  | Jun-Sep | 16-21 | winter only, 21-08 (overnight) | 5 |
| SDGE | Jun-Oct | 16-21 | 06-16 OR 21-22 (both seasons) | 6 |

A regression test (`tests/test_bill.py::test_period_masks_match_parent_config_if_present`) verifies our masks exactly equal those from `<utility>_config.build_*_period_masks` when the parent config modules are present.

## Subsidy regimes
Two regimes are computed for every cell, in parallel:
- **`2026_base`** (post-OBBB, default):
  - Federal: 25C / 25D / 30D = 0 (OBBB repeal, effective 1/1/2026 / 9/30/2025 for EV).
  - CA: TECH, SGIP, SGIP-HPWH, HOMES, Golden State, RENs, CC4A by air district, DCAP (≤300% FPL).
- **`2024_counterfactual`** (pre-OBBB):
  - Restores 30% PV/battery ITC, $2K 25C HP/HPWH, $600 panel, $7,500 30D EV, 30% 30C EVSE.
  - CA programs unchanged.

**The 2026-vs-2024 NPV gap per cell** is the central paper question: how much can rate design recover what OBBB took away?

## Fuel prices
- **Natural gas** (residential bundled non-CARE, Jan 2026; $/therm): PGE 2.92, SoCalGas (used for SCE) 2.08, SDGE 2.10. CARE customers: × (1 - 0.20).
- **Gasoline**: $4.90/gal default; sensitivity range $3.50-$6.50.
- **EV efficiency**: crossover 3.3 mi/kWh, sedan 4.0, default 3.5.
- **ICE MPG**: crossover 27, sedan 32, default 28.
- **VMT**: 12,000 mi/yr default; sensitivity grid {5K, 8K, 12K, 15K, 20K, 25K}.

## Financial framework
- **Horizon**: 20 years.
- **Discount rate**: 5% real (CPUC ACC customer perspective).
- **Bill escalator**: 2% real per year (Cal Advocates 2025 Rates Report).
- **PV inverter replacement**: $2,500 at year 13 (microinverters ignored).
- **Battery life**: 15 years (NREL ATB 2024b residential).

## Pipeline (`run_npv.py`)
For each `(medoid × bundle × sizing)`:
1. Load `baseline_hourly = Baseline_<U>/<bldg>-0.parquet` (15-min → 8760).
2. If HP bundle: add `upgrade11_delta = Upgrade11_<U>/<bldg>-11.parquet − baseline`.
3. If EV bundle: add `ev_hourly_load(VMT / EV_eff, 'smart_tou')`.
4. If PV bundle: **solve `battery_lp_dispatch` ONCE** using the actual baseline tariff (F0_WF0_ROE0 = E-TOU-C / TOU-D-4-9 / TOU-DR) — see "LP-once-and-reuse" below.

Then for each `(rate scenario × subsidy regime)`:

5. `bill_pre = compute_annual_bill(baseline_hourly, scenario)`.
6. `bill_post = compute_annual_bill(post_electrification_hourly, scenario, eec=eec_hourly)`.
   - For PV bundles, `post_electrification_hourly = grid_in − grid_out` from the single LP solve in step 4 (no re-optimization).
7. `annual_savings = (bill_pre - bill_post) + gas_savings + gasoline_savings`.
   - `gas_savings`: HP bundles only; baseline_therms × $/therm (CARE-discounted).
   - `gasoline_savings`: EV bundles only; VMT × $/gal ÷ MPG.
8. For each subsidy regime: `net_capex = gross_capex - subsidies`; `npv = payback_npv.npv(annual_savings, net_capex)`.
9. Write one parquet row.

### LP-once-and-reuse (battery dispatch)

Matches the user's prior paper methodology (`*_post_adoption.py`). The LP that picks battery charge / discharge / grid import / grid export is solved **once per (medoid, bundle, PV size, battery size) cell** using the actual baseline tariff (`F0_WF0_ROE0` in the rate designer's coordinates, which equals `E-TOU-C` / `TOU-D-4-9` / `TOU-DR` in the retail Excel). The resulting `grid_import` and `grid_export` 8760-arrays are reused for all 40 designed rate scenarios via simple dot products:

```
bill_post[scenario] = max(grid_in · rate[scenario] - baseline_credit
                          - grid_out · eec_hourly, 0)
                    + fixed_annual[is_care, scenario]
```

Justification: across the 40 designed scenarios, the rate designer scales all TOU periods proportionally as `Fixed_Pct_TD` varies (0% → 100%). The peak-to-offpeak **ratio** is constant to 4 decimals (PGE 1.2637, SCE 1.5316, SDGE 1.3333). Since the optimal battery dispatch is driven by that ratio (when to shift kWh from peak → offpeak), the LP solution is essentially identical across scenarios — only the absolute *cost* changes. Bias from reuse is below numerical precision.

Cuts LP count from `40 × (medoids × sizing cells)` to just `(medoids × sizing cells)` — a 40× speedup. Full-run compute drops from ~5-14 days to ~12-24 hours on the user's M1 Pro (8 workers).

### Bill clamp at zero (volumetric leg)

Volumetric leg of the bill is clamped at 0 before adding the fixed charge:

```
total_bill = max(vol_bill_after_care - export_credit, 0) + fixed_annual
```

Matches actual NBT billing convention: a customer's net bill before fixed charges cannot go below zero. Net Surplus Compensation (the small annual true-up payment at ~$0.05/kWh wholesale rate for true-up-cycle surplus) is **not** modeled. With the PV sizing grid capped at 1.25× annual load (within NBT interconnection eligibility), the clamp rarely binds; for the 1.25× tier the conservative bias is below 5% per cell.

**Compute estimate**: 2,541 medoids × 8 bundles × ~6 sizing cells per PV bundle ≈ **24 LP solves per medoid** (not per scenario). At ~0.94s per LP on M1 Pro → ~22.5 s/medoid → **~16 hr serial / ~2-3 hr with 8 workers** for the full sweep. Output rows: 2,541 × 8 bundles × 6 sizing × 40 scenarios × 2 subsidy regimes ≈ ~9.8M rows.

## Output schema (`data/npv_results.parquet`)
One row per `(medoid × rate × bundle × sizing)`:

| Column group | Columns |
|---|---|
| Identity | bldg_id, utility, cec_cz, cluster_weight |
| Income tier | income_category, is_care, ami_frac |
| Bundle | bundle, pv_kw, batt_kwh |
| Rate | rate_id, Fixed_Pct_TD, Remove_Wildfire, ROE_Reduction |
| Bills | bill_pre, bill_post, electric_savings |
| Non-electric | gas_savings, gasoline_savings, annual_therms |
| Total | annual_savings |
| Capex | net_capex_2026_base, net_capex_2024_counterfactual |
| **NPV** | **npv_2026_base, npv_2024_counterfactual** |

## Parent inputs (must be present)
- `CA_baseline_tmy_metadata_and_annual_results.parquet` — building metadata + annual results.
- `puma_utility_data.csv` — PUMA-to-utility mapping.
- `rate_scenarios_<u>_fresh.csv` — 40 designed scenarios per utility (from parent `rate_designer.py`).
- `retail_rates_data_<U>.xlsx` — actual tariff rows + baseline_puma sheet.
- `tou_weights_<u>.csv` — used by parent rate designer; not directly by EE.
- `eec_hourly_2025_wide.csv` — NBT hourly export compensation.
- `Baseline_<U>/<bldg_id>-0.parquet` per medoid — hourly baseline load.
- `Upgrade11_<U>/<bldg_id>-11.parquet` per medoid — hourly post-electrification load.

**Note on `Upgrade11_<U>/`**: only download parquets for medoid building IDs (2,541 files vs ~115K full population). Use `data/representative_buildings.parquet['bldg_id']` to filter.

## Usage
```bash
python -m src.preflight                       # validate inputs
python -m src.run_npv --limit 20              # smoke (20 medoids/utility)
python -m src.run_npv                         # full run
python -m src.run_npv --eec-multiplier 1.25   # NBT-softening sensitivity
```

## Out of scope
- Residential demand charges (`DC_5` / `DC_15` parked; future paper).
- Wholesale / FERC 2222 / CAISO DLAP export compensation.
- Dynamic / real-time pricing.
- Per-customer-class revenue neutrality (canonical-40 are utility-population-neutral).
- Behavioral demand response to any rate (load shape held fixed at ResStock baseline).
