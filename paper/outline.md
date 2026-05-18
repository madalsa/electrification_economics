# Paper outline: Rate reform vs. the post-OBBB electrification gap

## Working title
Rate design as a substitute for federal capital subsidies: personal
electrification economics in California after OBBB and AB 205.

## Thesis

The One Big Beautiful Bill (OBBB, P.L. 119-21, July 2025) effectively
zeroed the federal residential capital-subsidy stack — Sections 25C,
25D, and 30D — for installations after Dec 31, 2025. Simultaneously,
California is implementing the AB 205 income-graduated fixed charge
(IGFC) at conservative levels and considering further moves on
wildfire-cost socialization and ROE reduction. Holding the post-OBBB
capex stack fixed, we quantify how much of the personal-NPV gap
opened by federal repeal each available CPUC rate-reform lever can
recover, for which households, and across which electrification
bundles. The answer is empirical and falls out of the data once the
pipeline runs.

## Research questions

1. **How big is the OBBB gap?**  For each archetype × bundle, what is
   the 2024-counterfactual NPV under status-quo rate minus the 2026
   NPV under status-quo rate?
2. **How much of the gap can rate reform close?**  Among the
   canonical-6 CPUC rate-reform scenarios (F0/F50/F100 × WF0/WF1 ×
   ROE0/ROE1, all revenue-neutral), what is the maximum NPV recovery
   under the 2026 capex stack?
3. **Is rate design a first-order or second-order lever?**  For each
   (archetype × bundle), what is the spread of NPV across the
   canonical-6 rate scenarios — both in absolute dollars and as a
   share of total NPV magnitude?
4. **Where does NPV come from?**  Bundle-by-bundle decomposition of
   NPV into (a) rate-independent fuel-avoidance (gas, gasoline),
   (b) rate-dependent electric-bill changes (PV self-cons + storage
   arbitrage + EV/HP added load), and (c) capex (after stacked rebates).
5. **Distributional incidence.**  Who is hurt most by OBBB; who is
   helped most by each rate-reform direction; which household profiles
   are stranded under all six rate scenarios?

## Methods

See `paper/methods.md` for the full pipeline description. Headline:

- **Population:** owner-occupied, IOU-served, non-EBD-eligible CA
  households. Represented by ~1,500 medoid archetypes (stratified
  k-means on 11 features incl. load-shape proxies) with population
  weights from ResStock CA × PUMA-utility mapping.
- **Rates:** canonical-6 designed-TOU + per-utility EV-TOU opt-in +
  NBT-scaled overlay sensitivity (×1.0 / ×1.25 / ×1.50). All
  canonical-6 are revenue-neutral by construction.
- **Bundles:** 8 combinations (none, pv_bat, ev, hp, pv_bat_ev,
  pv_bat_hp, ev_hp, pv_bat_ev_hp). PV+battery is re-sized to the
  bundle's *expanded* load (captures PV-EV / PV-HP synergy).
- **Capex stack:** post-OBBB 2026 (federal 25C/25D/30D = 0; CA
  programs intact: TECH, SGIP, SGIP-HPWH, HOMES, Golden State, RENs;
  CC4A by air district; DCAP for ≤300% FPL). 2024 counterfactual
  toggle exists for the gap-opening comparison.
- **NPV:** 20-yr real, 5% real discount, 2%/yr real bill escalator,
  inverter replacement at year 13 if PV.
- **Decomposition:** bundle output retains 11 component columns
  (3 capex pieces + 5 annual-savings pieces + 3 per-component NPVs)
  so rate sensitivity, fuel elasticity, and source-of-NPV can be
  computed post-hoc analytically without re-running the optimizer.

## Key figures (planned)

1. **NPV-gap map.** Per archetype × bundle, color = 2024-counterfactual
   NPV minus 2026-base NPV (status-quo rate). What did OBBB cost each
   profile?
2. **Rate-reform recovery.** Per archetype × bundle: best canonical-6
   NPV minus status-quo NPV under 2026 capex. What can CPUC do?
3. **Rate-design alpha.** Distribution of NPV spread across the 6
   rates, per bundle. Headline single number per bundle for the
   abstract.
4. **Source-of-NPV decomposition.** Per bundle, stacked bar of (gas
   avoided + gasoline avoided + electric-bill net) − capex. Shows
   which lever moves how much.
5. **Distributional heatmap.** Customer-type × rate → NPV under each
   bundle. Reveals who wins / loses under each rate direction.
6. **Fuel-price elasticity.** Per bundle, contour of NPV under
   alternative gas × therm prices. Falls out of the decomposition
   columns; computed analytically.
7. **NBT softening sensitivity.** Per bundle, NPV at NBT × 1.0 / 1.25
   / 1.50. Most relevant for PV-containing bundles.

## Policy framing

- **Who can do what:** rate moves are CPUC; capex subsidies are
  congressional. OBBB is sunk; AB 205 implementation is live.
- **The substitution question:** the paper provides the dollar
  arithmetic for CPUC commissioners considering how aggressive to be
  on F / WF / ROE. If rate reform recovers 60% of the OBBB gap for
  the median household, that is materially different from 10%.
- **Equity:** identifies customer profiles for whom *no* canonical-6
  rate restores 2024-with-subsidies NPV. These are the households for
  whom federal-policy reversal cannot be remedied at the rate-design
  layer.

## Datasets

- ResStock CA baseline + (PATH A) Upgrade 11 hourly parquets
- RASS 2019 cleaned survey
- Parent `california_rates` rate-designer fresh CSVs (40 designed
  scenarios per utility)
- TOU weights (parent pipeline) per utility
- CPUC EEC hourly (LY2025 NBT Pricing Upload MIDAS)
- PUMA-utility-CZ mapping
- IRA / OBBB statute summaries; CPUC R.22-07-005 docket; CEC EBD
  budget docs; TECH / SGIP / HOMES program tables (2026 stack)
- CC4A district-specific rebate schedules (CARB)
- KBB / Cox Auto Mar 2026 ATP for EV premium
- NREL ATB 2024b for PV / battery lifetime + replacement

## Literature anchors

- **Borenstein, Fowlie, Sallee (2021)** — designing electricity rates
  for an equitable energy transition; direct precedent for the IGFC
  equity question.
- **Borenstein & Bushnell (2022 AEJ:Policy)** — retail rate
  distortions vs externalities.
- **Davis & Hausman (2022 JEEM)** — EV vs ICE total cost of ownership;
  methodological cousin for the EV-bundle side.
- **Burlig, Bushnell, Rapson, Wolfram (2021 AER:I)** — EV electricity
  consumption from meter data; validation for VMT × charging-profile.
- **Borenstein (2017 JEPER)** — private net benefits of residential
  solar PV; methods precedent for personal-economics framing.
- **CPUC R.22-07-005** — AB 205 / IGFC rulemaking docket;
  institutional context.

## What this paper deliberately does NOT do

(Signposted in `methods.md` §13 as future work.)

- *Does not* model residential demand charges. DC is parked for a
  follow-up paper on the Borenstein peak-demand-charge angle.
- *Does not* model wholesale / FERC 2222 / DLAP export comp.
  CAISO DLAP data path exists but aggregator economics are out of
  scope here.
- *Does not* model dynamic / real-time pricing.
- *Does not* enforce per-customer-class revenue neutrality
  (canonical-6 are utility-population-neutral, which is the standard
  rate-design criterion; per-class would be a different paper).
- *Does not* model behavioral demand response to any rate. Load shape
  is taken as fixed at the ResStock baseline. The paper measures
  rate-induced *cost* changes, not rate-induced *behavior* changes.
