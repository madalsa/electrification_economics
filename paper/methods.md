# Methods: Electrification Economics pipeline

Step-by-step description of the pipeline as implemented. Each stage is
independently re-runnable, reads only from prior stages or parent
`california_rates/` outputs, and writes only to
`electrification_economics/data/`.

The orchestrator (`src/run_economics.py`) sequences stages 0 → 5.
Preflight (`src/preflight.py`) validates prerequisites and prints a
run plan before any stage executes. Post-hoc decomposition
(`src/decompose.py`) consumes stage 5 outputs without re-running.

---

## 0. Population scope

The unit of analysis is the **household making a discretionary
electrification decision under a discount rate**. Three exclusions are
applied before any modeling:

1. **Investor-owned-utility (IOU) territories only** — PG&E, SCE, SDGE.
   POU customers (LADWP, SMUD, IID, etc.) are filtered out via the
   PUMA-utility mapping; their tariffs and incentive landscape differ
   materially.
2. **Owner-occupied only** — capex / payback decisions belong to
   property owners; renters cannot install PV or upgrade HVAC unilaterally.
3. **Non-EBD-eligible only** — Equitable Building Decarbonization
   direct-install (CEC, launched April 2026) provides turnkey free
   retrofit (HP + HPWH + panel + induction) to households at ≤80% AMI
   in CEC priority climate zones (8, 9, 10, 13, 14, 15). NPV is not a
   meaningful question for them; capex is zero by program design. They
   are reported in `population_excluded_summary.csv` as a population
   share but excluded from the optimization population.

Households with "Not Available" income tier are also dropped (cannot
classify against AMI).

The resulting population is **owner-occupied, IOU-served, non-EBD
households** — the cohort for whom the personal NPV question is live.

---

## 1. Representative buildings (stage 0)

**Source:** `src/representative_buildings.py`
**Output:** `data/representative_buildings.parquet`,
`data/population_excluded_summary.csv`

### 1.1 Sampling problem

The ResStock CA baseline has ~150k synthetic households. Running the
full sizing × rate × bundle sweep on all of them is intractable
(billions of cells) and unnecessary — most are statistically
indistinguishable from a nearby medoid. Stratified clustering picks
a representative sample with population weights so aggregate claims
remain valid.

### 1.2 Stratification axes

After scope filtering, households are stratified on the
cross-product of:

| Axis | Levels |
|---|---|
| utility | PGE, SCE, SDGE |
| CEC climate zone | 1–16 |
| heating fuel | natural gas / electricity / propane / fuel oil / none |
| building type | single-family detached / attached / mobile / multifamily |
| AMI bin | 0-30% / 30-60% / 60-80% / 80-100% / 100-120% / 120-150% / 150%+ |
| vintage decade | pre1940 / pre1960 / 1960-70s / 1980-90s / post2000 |

Empty strata are silently dropped. The orthogonal cuts capture the
demographic and physical drivers of electrification economics.

### 1.3 Within-stratum clustering

For each non-singleton stratum, k-means clustering on standardized
features produces ≤ k_cap medoids (the medoid = household closest
to its cluster centroid in standardized feature space). Clustering
features are explicitly chosen to span both *volume* (how much
energy) and *shape* (when it is used — relevant because rate-design
sensitivity is TOU-driven):

| Feature | What it captures |
|---|---|
| annual_kwh | Total electricity volume |
| annual_therms | Total natural-gas volume |
| sqft | Building size |
| summer_peak_kw, winter_peak_kw | Demand-charge exposure |
| cooling_share, hvac_share | Midday-peaking load share |
| hot_water_share | Year-round flat load with morning/evening peaks |
| plug_loads_share | Evening-peak-skewed load share |
| peakiness_summer = summer_peak_kw / (annual_kwh / 8760) | Spiky vs flat summer load |
| peakiness_winter = winter_peak_kw / (annual_kwh / 8760) | Spiky vs flat winter load |

`k_cap = 8`, default `--target 1500` (`avg_k = target // n_strata`
sets the typical within-stratum k; small strata stay at k=1). Each
medoid carries a `cluster_weight` equal to the sum of source
building weights in its cluster.

### 1.4 Validation

If end-use columns (plug_loads, hot_water) are absent from a
particular ResStock release, the affected share columns default to 0;
preflight raises a `WARN` so the user knows the shape-feature signal
is degraded. Tests verify the build_features function is robust to
missing optional columns and to zero-kWh inputs.

---

## 2. Extended rate design (stage 1)

**Source:** `src/rate_designer_extended.py`
**Output:** `data/rate_scenarios_extended_<utility>.csv`

The parent `california_rates` pipeline produces 40 designed rate
scenarios per utility along three policy axes: fixed-charge share
(`Fixed_NonCARE` ∈ {0%, 50%, 100%}), wildfire-cost socialization
(`Remove_Wildfire` ∈ {0, 1}), and ROE reduction
(`ROE_Reduction` ∈ {0, 1.0}). The paper's canonical-6 subset is:

| ID | Description |
|---|---|
| F0_WF0_ROE0     | Status quo (pre-IGFC baseline) |
| F0_WF0_ROE1.0   | ROE-only reduction |
| F50_WF0_ROE0    | 50% fixed-charge transition |
| F50_WF1_ROE0    | 50% fixed + wildfire socialized |
| F100_WF0_ROE0   | Full fixed-charge transition |
| F100_WF1_ROE0   | Full fixed + wildfire socialized |

Stage 1 re-emits all 40 in the extended schema and **augments** the
rate space with:

- **EV-only TOU** opt-in tariff(s) (`EV_TOU`) — per-utility analogues of
  PGE EV2-A / SCE TOU-EV-9-PRIME / SDGE EV-TOU-5. Applies to submetered
  EV load only; the rest of the household stays on its base rate. These
  are parallel opt-in tariffs whose revenue requirement is handled by
  the utility's own filing for that customer class — they are NOT
  revenue-neutralized against the canonical-6 set.
- **NBT export-regime overlays** (`EXPORT_NBT_HOURLY`,
  `EXPORT_NBT_SCALED_125`, `EXPORT_NBT_SCALED_150`). NBT hourly is the
  current law for new interconnections (post April 15, 2023); the
  scaled variants are CPUC-softening sensitivities. Each carries an
  `eec_multiplier` column (1.0 / 1.25 / 1.50) that
  `bundle_economics --eec-multiplier` applies at runtime to the
  annual-average EEC.

**Explicitly out of scope for this paper's headline rate set:**

- *Residential demand charges* (`DC_5`, `DC_15`). These are parked in
  the module (set `--include-demand-charges` to restore) for a follow-up
  paper focused on residential demand charges + electrification
  compatibility. Reasons: (i) residential DC is not currently on the
  CPUC table in CA, (ii) this pipeline takes load shape as fixed and so
  doesn't capture the shape-change goal that motivates DC, and (iii)
  Borenstein's existing peak-demand work argues residential DC is
  regressive without behavioral response, which the future paper would
  engage directly. Listed in §13.
- *NEM 2.0 retail / flat 5c / flat 15c counterfactuals* — removed.
  NEM 2.0 is grandfathered out and abstract flat-rate scenarios aren't
  policy-relevant. The NBT-scaled overlays span the realistic
  CPUC-action envelope.
- *Wholesale / FERC 2222 / CAISO DLAP export comp* — listed in §13 as
  future work. Tractable data path exists (CAISO OASIS DLAP hourly)
  but requires more careful treatment of aggregator economics and is
  scoped out of this paper.

Extended schema columns:

```
scenario_id, rate_type, fixed_monthly_dollars,
demand_charge_per_kw_mo, peak_window,
summer_peak, summer_midpeak, summer_offpeak,
winter_peak, winter_midpeak, winter_offpeak,
ev_super_offpeak, ev_on_peak,
export_regime, source_scenario, notes
```

`fixed_monthly_dollars` is approximated from `Fixed_NonCARE` × T&D
revenue share / 12 / residential customer count (FERC Form 1, 2024).

---

## 3. PV + battery sizing (stage 2)

**Source:** `src/sizing_optimizer.py` (heuristic, default)
**Source:** `src/sizing_optimizer_hourly.py` (LP refinement, optional)
**Output:** `data/sizing_results_<u>.parquet`,
`data/sizing_optimal_<u>.parquet`

### 3.1 TOU-aggregate heuristic (sizing_optimizer.py)

For each (building × rate × PV size × battery size) candidate, an
annual energy balance is computed at the TOU-period level:

1. **Load by period** = annual kWh × utility TOU weight share
   (from `tou_weights_<u>.csv`).
2. **PV generation** = pv_kw × 1700 kWh/kW/yr (NREL PVWatts central CA
   tilted-south), distributed across TOU periods per
   `PV_GEN_TOU_SHARE` (most generation lands in offpeak / midpeak;
   little in 4-9pm peak).
3. **Self-consumption per period** = min(load_period, gen_period).
4. **Export per period** = max(0, gen_period - load_period), valued
   at the utility's annual-average EEC ($/kWh; PG&E 0.097, SCE 0.085,
   SDGE 0.078).
5. **Battery arbitrage** = min(batt_kwh × 365 × 0.88 roundtrip eff,
   peak-period load, offpeak-to-peak capacity); shifts kWh from peak
   to offpeak.
6. Bill before / after, capex through `payback_npv.apply_capex_stack`,
   20-year NPV at 5% real discount with 2%/yr real bill escalator and
   inverter replacement at year 13 ($2,500) for PV systems.

Grid: PV kW ∈ {0, 2, 4, 6, 8, 10, 12, 15}; battery kWh ∈
{0, 5, 10, 15, 20, 30}. The (0, 0) combo is the baseline (excluded).

This is **first-order accurate** — it captures the main mechanism
(PV serves daytime / offpeak load, battery shifts excess into peak
hours) but underestimates LP-optimal NPV by ~10–20% because it
misses within-period arbitrage.

### 3.2 Hourly LP refinement (sizing_optimizer_hourly.py)

For ~10–20 per-CZ medoid-of-medoids archetypes, a full 8,760-hour
linear program is solved per (rate × PV size × battery size). Uses
`scipy.linprog` with `highs` solver; mirrors the
`<utility>_battery_lp` modules from the parent repo (battery
roundtrip eff 0.88, C-rate 0.4). Requires `Baseline_<u>/` parquets
locally — not in this repo. Output schema matches the heuristic so
figures can toggle between v1 and v2.

---

## 4. EV sensitivity sweep (stage 3)

**Source:** `src/vmt_sensitivity.py`
**Output:** `data/ev_sensitivity_<u>.parquet`,
`data/ev_sensitivity_summary.csv`

For each (building × rate × VMT × gas_price × charging_profile × EV
scenario):

1. **Effective $/kWh for charging** — a profile-weighted average of
   the rate's TOU prices. Three charging profiles:
   - `overnight_offpeak`: 95% of charging midnight–7am, 5% other.
   - `opportunistic`: flat across 24 hr (no smart scheduling).
   - `smart_tou`: shifts to lowest-price hours of host tariff
     (currently proxy = overnight; will refine with bill simulator).
2. **Annual fuel savings** = (VMT/MPG × gas_price) − (VMT/eff × eff_$/kWh).
   ICE MPG defaults to vehicle class (sedan 32, crossover 27);
   EV efficiency defaults (sedan 4.0, crossover 3.3 mi/kWh).
3. **Net EV premium** by EV scenario:
   - `new_new`: full premium ($5,800; KBB Mar 2026 EV vs ICE ATP).
   - `new_ev_dcap`: premium − $7,500 (DCAP, ≤300% FPL).
   - `scrap_replace_cc4a`: premium − Clean Cars 4 All rebate by air
     district (BAAQMD $9,500; SCAQMD $12,000; SJVAPCD $9,500;
     SMAQMD $9,500; San Diego runs DCAP only).
4. NPV at 5% real over 20 years; payback against net premium.

Grids: VMT ∈ {5k, 8k, 12k, 15k, 20k, 25k} mi/yr;
gas_price ∈ {$3.50, $4.50, $5.50, $6.50}/gal.

---

## 5. Heat-pump (Upgrade 11) economics (stage 4)

**Source:** `src/upgrade11_economics.py`
**Output:** `data/upgrade11_economics_<u>.parquet`

Two implementation paths:

- **PATH A** (when available): hourly Upgrade11_<u>/ parquets from the
  parent pipeline; computes per-period load + gas deltas. Runs only on
  a machine with those files.
- **PATH B** (default): annual-aggregate approximation from baseline
  metadata + COP/UEF assumptions. Sufficient for sensitivity sweeps.

Path B calculation per building:

```
delta_kwh_hp_space = baseline_gas_heating_therms × 29.3 / COP_space(cec_cz)
delta_kwh_hpwh     = baseline_gas_hot_water_therms × 29.3 / UEF_HPWH
delta_kwh_induction = baseline_gas_range_therms × 29.3 × 0.85
total_delta_kwh    = sum of the above
gas_displaced      = baseline_gas_heating + hot_water + range therms

elec_cost_increase = delta_hp_space × winter_avg_rate
                   + (delta_hpwh + delta_induction) × yearround_avg_rate
gas_savings        = gas_displaced × gas_$/therm (PGE 2.92, socalgas 2.08, sdge 2.10)
net_annual_savings = gas_savings - elec_cost_increase
```

COP per CEC CZ: 2.3 (CZ16, coldest) – 3.2 (CZ6/7/8, mildest); HPWH
UEF 3.0; induction efficiency 0.85.

Capex (HP + HPWH + induction range + 200A panel) goes through
`payback_npv.apply_capex_stack` with stacked rebates:
TECH + SGIP-HPWH + Golden State + REN (BayREN $400 or 3C-REN $5,000)
+ HOMES (whole-home retrofit, $4–8k depending on income tier).
Federal 25C/25D = 0 in 2026 (OBBB repeal).

---

## 6. Bundle economics (stage 5)

**Source:** `src/bundle_economics.py`
**Output:** `data/bundle_economics_<u>.parquet`,
`data/bundle_summary.csv`

### 6.1 Bundles considered

| Bundle | Components |
|---|---|
| `none` | Do-nothing baseline (reference) |
| `pv_bat` | PV + battery |
| `ev` | EV (single passenger vehicle) |
| `hp` | HP space + HPWH + induction + panel |
| `pv_bat_ev` | PV + battery + EV |
| `pv_bat_hp` | PV + battery + HP |
| `ev_hp` | EV + HP (no solar) |
| `pv_bat_ev_hp` | Full residential electrification |

### 6.2 Load-expansion and re-sizing

When a bundle adds EV charging or heat-pump load, the household's
electric load profile changes — and the optimal PV / battery size
changes with it. The module captures this synergy rather than
additively combining standalone NPVs:

1. **EV load** is distributed across the utility's TOU periods using
   the `smart_tou` charging profile (concentrated overnight).
2. **HP load** is distributed by component: space heating to winter
   periods only (proportional to baseline winter TOU share);
   HPWH + induction across all periods (year-round proportional).
3. **Expanded load** = baseline + EV + HP per period.
4. **PV + battery grid search** (when in bundle) is run against the
   *expanded* load, picking the (pv_kw, batt_kwh) that maximizes the
   PV+battery component NPV.

### 6.3 NPV decomposition

The output schema retains components separately rather than summing
into one annual_savings / one NPV. This is the central methodological
choice for the paper.

**Capex pieces** (all net of stacked rebates, all rate-independent):

- `capex_pv_bat` — PV + battery (post-SGIP)
- `capex_ev` — EV net premium after CC4A / DCAP if applicable
- `capex_hp` — HP + HPWH + induction + panel after TECH + SGIP-HPWH
  + Golden State + REN + HOMES
- `capex_total` = sum

**Annual savings pieces (year-1, real $):**

| Column | Rate-dependence | Linear in fuel price? |
|---|---|---|
| `bill_savings_pv_bat` | **rate-dep** | no |
| `gasoline_avoided` | rate-indep | yes (gas_price) |
| `ev_charging_cost` | **rate-dep** | no |
| `gas_avoided_value` | rate-indep | yes (therm_price) |
| `hp_elec_increase` | **rate-dep** | no |
| `annual_savings` = sum (with signs) |  |  |

**NPV per component** (20-yr real, sums to `npv`):

- `npv_pv_bat` = NPV(bill_savings_pv_bat, capex_pv_bat) with inverter
  replacement at year 13 if pv_kw > 0
- `npv_ev` = NPV(gasoline_avoided − ev_charging_cost, capex_ev)
- `npv_hp` = NPV(gas_avoided_value − hp_elec_increase, capex_hp)

The decomposition enables three things post-hoc without re-running
the optimizer (handled by `src/decompose.py`):

1. **Rate-design sensitivity:** within (bldg, bundle), the spread of
   `npv` across the 6 rate scenarios (and of just the rate-dependent
   savings components) tells you what fraction of total NPV is
   actually under CPUC's control.
2. **Fuel-price elasticity:** `gasoline_avoided` is linear in
   gas_price; `gas_avoided_value` is linear in therm_price. Scaling
   those columns by (alt_price / base_price) gives NPV under
   counterfactual fuel prices — no re-run.
3. **Source-of-NPV decomposition:** per bundle, the medians of each
   component yield the rate-indep share (fuel avoid), rate-dep share
   (electric-bill change), and capex share of total NPV.

---

## 7. Financial framework

All NPVs are computed in real (inflation-adjusted) dollars at a
**5% real discount rate** (CPUC ACC / DER cost-effectiveness customer
perspective; sensitivity range 3–8%). Horizon **20 years** (matches
NEM 2.0 grandfathering window and PV warranty; sensitivity 15 / 25
yr). Bill escalator **2% real per year** (CPUC GRC filings and Cal
Advocates 2025 Rates Report; sensitivity 0–4%).

Capex is paid at year 0; annual savings accrue years 1 through N
with the escalator applied. Simple payback is `capex / year-1
savings`; discounted payback is the year at which the cumulative
discounted cashflow equals capex.

---

## 8. Incentive stack (2026 base case)

**Federal (effectively zero):** OBBB (P.L. 119-21, July 2025) repealed
Sections 25C, 25D, 30D for installations after Dec 31, 2025 (vehicles
after Sept 30, 2025). Section 30C (EVSE) sunsets June 30, 2026.

**California programs still active in the 2026 base case:**

- **TECH Clean California** — $1,000/HP space ($2,000/home cap),
  $2,700 / $4,600 HPWH (market / equity).
- **SGIP-battery** — $200/kWh general (30 kWh cap),
  $850/kWh equity, $1,050/kWh equity-resilience (80 kWh cap).
- **SGIP-HPWH** — $3,800 standard, $4,885 low-income;
  +$1,500 low-GWP adder.
- **HOMES (IRA, CEC-administered)** — $4,000 market / $8,000
  low-income for whole-home retrofit (HP + HPWH at minimum).
- **Golden State Rebates** — $300 HPWH, $85 smart thermostat,
  stacks with TECH (through 12/31/2026).
- **Regional Energy Networks** — BayREN HPWH $250–$400,
  3C-REN $5,000 contractor incentive for SF HPWH.
- **DCAP** — $7,500 new ZEV ≤300% FPL; +$4,500 DAC bonus.
- **Clean Cars 4 All** — varies by air district (BAAQMD $9,500,
  SCAQMD $12,000, SJVAPCD/SMAQMD $9,500, SDAPCD does not run CC4A).
- **HEAR** — single-family fully reserved as of 2026-02-24;
  not available in base case (flag remains for multifamily / new
  appropriations).
- **EBD direct install** — turnkey free (replaces capex) for ≤80%
  AMI in CEC priority CZs; mutually exclusive with TECH/HEAR.
  *Population-filtered out of the optimization set.*

**Counterfactual:** `INCENTIVES_2024_COUNTERFACTUAL` restores the
pre-OBBB federal stack (25% ITC PV/battery, $2,000 25C HP/HPWH cap,
$600 panel, $7,500 30D EV, 30% 30C EVSE). Toggle via
`IncentiveContext(use_2024_counterfactual=True)`. Lets us measure
the NPV gap OBBB opened.

---

## 9. Safety guards

`config.assert_safe_out_dir(path)` refuses to write anywhere outside
`electrification_economics/data/`. Every CLI module passes its
`--out-dir` through the guard before any write. Static test
(`test_safety_guards.test_no_module_writes_outside_data_dir`) scans
all `src/*.py` for `to_parquet(` / `to_csv(` lines that reference
`CR_ROOT /` or `parents[2] /` (would escape EE_ROOT). Required
because EE is run from inside `california_rates/` so parent modules
are importable, but EE must never overwrite parent pipeline outputs
or shared-storage `Baseline_<u>/` symlinks.

---

## 10. Preflight validation

`src/preflight.py` runs before the orchestrator. Each check returns
PASS / WARN / FAIL with a human-readable line:

- **Output dir** writable + ≥2 GB free.
- **ResStock metadata** parquet present, has the 13 required columns
  (else stage 0 crashes); warns on absent optional columns
  (`plug_loads`, `hot_water`, NG end-uses — they default to zero but
  degrade signal).
- **PUMA-utility mapping** covers PGE/SCE/SDGE.
- **TOU weights** CSV per utility; required periods present.
- **Rate scenarios** fresh CSV per utility; warns if canonical-6
  subset is missing.
- **EEC hourly** present, ~8,760 rows per utility.
- **Existing outputs** — flags stages with partial outputs (so the
  user can decide whether to re-run from scratch or pick up).

Then prints a **run plan** with estimated row counts and rough
runtimes per stage, so the user sees what to expect before pressing
go. Exits non-zero on any FAIL; `--strict` also non-zero on WARN.

---

## 11. Orchestration

`src/run_economics.py --stage all [--utility pge sce sdge] [--test]`
runs stages 0–5 in sequence as separate subprocesses (so a failure
in one stage doesn't pollute later state). `--test` limits each
stage to 50 buildings for smoke validation. Stage outputs are
independently re-runnable; each stage reads only its declared
inputs.

Suggested workflow:

```bash
python -m electrification_economics.src.preflight        # validate
python -m electrification_economics.src.run_economics --test
python -m electrification_economics.src.run_economics    # full run
python -m electrification_economics.src.decompose        # post-hoc
```

---

## 12. Files produced

```
data/
  representative_buildings.parquet         (stage 0)
  population_excluded_summary.csv          (stage 0)
  rate_scenarios_extended_<u>.csv          (stage 1)
  sizing_results_<u>.parquet               (stage 2)
  sizing_optimal_<u>.parquet               (stage 2)
  sizing_results_hourly_<u>.parquet        (stage 2, optional LP)
  ev_sensitivity_<u>.parquet               (stage 3)
  ev_sensitivity_summary.csv               (stage 3)
  upgrade11_economics_<u>.parquet          (stage 4)
  bundle_economics_<u>.parquet             (stage 5)
  bundle_summary.csv                       (stage 5)
  bundle_decomposition_<u>.csv             (post-hoc)
  bundle_rate_sensitivity_<u>.csv          (post-hoc)
  bundle_fuel_elasticity_<u>.csv           (post-hoc)
```

Headline analytical outputs for the paper (post-stage-5):

- **`bundle_rate_sensitivity_<u>.csv`** — per (bldg, bundle), spread
  of total NPV across the 6 rate scenarios. The empirical answer to
  "how much does rate design move household NPV?"
- **`bundle_decomposition_<u>.csv`** — per-bundle median split of NPV
  into rate-independent (gas/gasoline avoid) vs rate-dependent
  (electric-bill change) vs capex shares.
- **`bundle_fuel_elasticity_<u>.csv`** — bundle median NPV at
  alternative gas prices × therm prices, computed analytically.

These are the inputs for the paper's three central figures.

---

## 13. Out of scope / future work

The pipeline is deliberately bounded to questions the current
implementation can answer credibly. The four extensions below are
plausible follow-up papers; each is signposted here so a reviewer
sees what's been left out *on purpose*.

### 13.1 Residential demand charges + electrification compatibility

The DC_5 / DC_15 scenarios remain in `rate_designer_extended.py`
behind an `--include-demand-charges` flag. The follow-up paper asks
whether residential demand charges — proposed but not yet adopted in
CA, in force in AZ and HI — are compatible with electrification
without behavioral response. Adding an EV charger spikes monthly peak
by 7-10 kW; under DC_15 that's ~$1,260/yr in incremental demand
charges, comparable to or larger than the EV's fuel savings under
many TOU tariffs. The Borenstein peak-demand-charge literature
(Borenstein 2016 *EJ*; Borenstein, Fowlie, Sallee 2021 NBER) argues
DC is regressive without behavioral response. Engaging that argument
directly requires:

  (a) a behavioral-response model for monthly peak under DC pricing,
      parametric across customer types (load factor, smart-load
      penetration, battery availability),
  (b) a proper per-customer revenue-neutrality check (rather than the
      population-level proxy the current calibration uses),
  (c) coupling the household battery dispatch model (we have the LP)
      to a DC objective so the battery shaves monthly peak.

Out of scope for this paper because the headline (rate reform under
post-OBBB capex) doesn't need it.

### 13.2 Wholesale / FERC Order 2222 / DLAP export compensation

Replacing the EEC hourly file with CAISO DLAP wholesale prices would
answer "what if DER export went to pure wholesale via aggregator
participation under FERC 2222." Tractable on the data side (CAISO
OASIS API, ~1 GB/yr × 3 DLAPs, hourly). Out of scope because:

  (a) the household's compensation depends on aggregator economics
      that aren't publicly observable (privately negotiated fee
      structures, performance penalty exposure), so modeling
      household DER value under FERC 2222 requires more assumptions
      than the data alone supports;
  (b) the NBT-scaled overlays (×1.25, ×1.50) already span the
      realistic CPUC-action envelope for this paper.

The DLAP path is right for a paper about aggregator participation
specifically; it's not needed for a paper about CPUC-level rate
reform.

### 13.3 Dynamic / real-time pricing (RTP)

CAISO has proposed and piloted RTP variants. A household on RTP sees
hourly-changing prices, which interacts strongly with PV/battery
dispatch. The pipeline could be extended to RTP via the same hourly
LP path used for sizing_optimizer_hourly. Out of scope because:

  (a) household RTP isn't a CPUC-table proposal at the residential
      level in CA in 2026,
  (b) modeling RTP rigorously would require a household-response
      assumption (do they read the prices? respond automatically?),
      which puts the analysis in the same parametric territory as DC.

### 13.4 Per-customer revenue-neutrality and bill-impact tests

The canonical-6 rates are revenue-neutral at the population level by
construction in the parent rate designer, but per-customer bill
impacts vary widely (a low-usage household pays more under high
fixed charge; a high-usage household pays less, or vice versa
depending on volumetric). The paper reports archetype-level NPVs but
does not enforce per-customer-class revenue neutrality. A follow-up
could:

  (a) decompose bill impacts by AMI tier × CZ to show who bears the
      revenue shift in each rate move,
  (b) compute counterfactual rates that hold per-AMI-tier revenue
      neutral (rather than population total), as a contrast against
      the existing utility-level neutrality.

Useful for an equity-focused paper; not needed for the rate-reform
question we're answering here.
