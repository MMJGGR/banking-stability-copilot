# Phase 4 first slice — interpretation and broad observable overlays

Date: 24 September 2026  
Branch: `research/broad-feature-phase4-overlays-2026-09-24`  
Status: **complete as retrospective development evidence for interpretation and continuous-observable overlays**  
Production status: unchanged

## Scope completed

This slice kept the 96-dimensional Phase 2 state and Phase 3 probability engine fixed. It delivered:

- an interpretation card for every internal state dimension;
- a post-hoc orthogonal display rotation that preserves the underlying state geometry;
- compact-theme clustering and stability diagnostics;
- country-level current-state and forecast-movement interpretation outputs;
- a data-led registry covering all sufficiently supported historical observables;
- time-ordered comparisons of no change, own history, state-only and own-history-plus-state forecasts;
- explicit separation of IMF WEO provider projections;
- a governed dependency record for the later crisis-event overlay.

No selected banking variable was privileged when constructing the outcome registry.

## State interpretation finding

All 96 dimensions have auditable cards showing their strongest positive and negative observable contributors, source concentration and display-rotation composition.

The best compact clustering candidate used six groups and was stable under feature resampling, but it was **not sufficiently separated or balanced**:

- silhouette: **0.0785**;
- bootstrap adjusted-Rand stability: **0.8917**;
- largest group: **86 of 96 dimensions**;
- singleton groups: **2**.

Therefore Phase 4 does **not** accept a small set of replacement pillars. The appropriate architecture is:

- retain the 96-dimensional engine internally;
- use dimension cards and observable contributors for auditability;
- use analyst-facing observable families as explanations;
- do not claim that the state naturally reduces to a handful of clean themes.

The post-hoc rotation is for display only and preserves state-space distances and forecasts.

## Broad observable registry

The outcome registry was built from the historical/observed lane across FSIC, FSIBSIS, MFS, WEO and WGI.

- registered feature-horizon cases: **3,713**;
- globally eligible after unit/support rules: **780**;
- executed in at least one registered development window: **639**;
- executed in both windows: **349**;
- executed in one window: **290**;
- eligible but without enough rows inside a registered evaluation window: **141**;
- provider-projection rows used: **0**.

Each executed case compared:

1. no change;
2. the observable's own exact-calendar history;
3. the fixed 96-dimensional state alone;
4. own history plus a regularized state correction.

The state and penalties were evaluated using earlier training/inner periods only. Negative results are retained.

## Main overlay result

Across all 639 executed feature-horizon cases:

- the hybrid state overlay beat own history on RMSE in **443** cases;
- own history beat no change in **372** cases;
- the hybrid overlay beat no change in **405** cases;
- the hybrid overlay beat the state-only model in **563** cases;
- median incremental RMSE improvement over own history was only **0.27%**.

This means the state contains useful incremental information, but it is **not a universal replacement for each observable's own history**. The preferred overlay architecture is hybrid and selective.

### Practical evidence tiers

To avoid calling tiny statistically detectable effects economically meaningful, a `material stable` case must:

- execute in both registered development windows;
- improve in both windows;
- reduce aggregate RMSE by at least 1% versus own history;
- beat own history on more than half of country-period rows;
- pass a 10% Benjamini-Hochberg false-discovery threshold.

Results:

| Evidence tier | Cases |
|---|---:|
| Material and stable state increment | **79** |
| Statistically supported but small/partial | 160 |
| Positive exploratory result | 204 |
| Own history only | 106 |
| No confirmed gain | 52 |
| State overlay materially harmful | 38 |

For the 79 material/stable cases:

- median incremental RMSE improvement over own history: **3.40%**;
- maximum improvement: **21.99%**.

The strongest stable improvements occur mainly in detailed MFS balance-sheet/monetary series, with selected FSIC and WEO outcomes. Results remain heterogeneous: some observables are harmed by adding the broad state.

## Architecture implication

Phase 4 changes the interpretation/product design in two ways.

1. **Do not replace production's two pillars with another arbitrary small pillar set.** The learned state is internally high-dimensional and not cleanly separable into six or another small number of themes.
2. **Use the state as an incremental information layer, not a universal observable forecaster.** For specific outcomes, own history remains the base model and the state is added only where time-ordered evidence supports it.

The emerging architecture is:

**broad observed data → missing-aware state → probabilistic state forecast → dimension/observable explanation → selectively validated observable and event overlays**

## Crisis/event overlay status

The crisis overlay was not fitted in this slice. It requires:

- verified systemic-event dates aligned to the same country-origin panel;
- identical right-censoring and cooldown rules;
- a comparable frozen-production-classifier benchmark on the same rows;
- a registered rare-event evaluation plan.

Guessing or reconstructing event dates from incomplete repository metadata would weaken the evidence. The production classifier remains byte-locked and unchanged.

## Provider projections

WEO 2026–2031 provider projections remain scenario-only. They were not used as:

- overlay outcomes;
- state inputs;
- model-selection observations;
- realized outcomes;
- historical validation data.

## Tests and boundaries

Local Phase 4 engineering tests: **4 passed**.

Verified boundaries:

- measurement state refitted: **false**;
- selected banking targets predeclared: **0**;
- provider-projection rows read: **0**;
- production files modified: **false**;
- production classifier retrained: **false**;
- final untouched confirmation evaluated: **false**;
- GitHub Actions used for iterative Phase 4 development: **0**.

## Limitations

- Results use latest-vintage retrospective history rather than historical source vintages.
- The overlay registry tests many outcomes; false-discovery control and practical thresholds reduce, but do not eliminate, research-selection risk.
- Only 349 cases had enough support in both registered development windows.
- Currency outcomes use causal own-history scaling and should be interpreted as normalized movement rather than raw currency-level forecasts.
- The first slice evaluates continuous observables; crisis and other rare events remain separately governed.
- Nothing in this evidence authorizes production integration.
