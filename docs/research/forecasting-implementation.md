# Broad-feature forecasting: execution status

Updated 24 September 2026. Active branch: `research/broad-feature-phase4-overlays-2026-09-24`. Parent research PR #28 remains draft. Production baseline `5957ca779dafa21f2e098c819bfb060f43243206` is unchanged.

## Governing requirements

The binding requirements are:

- original broad-feature PRD;
- source-grounded data contract v0.2;
- target-independent discovery plan v0.5;
- Phase 2 measurement/transition PRD v0.6;
- Phase 3 probabilistic PRD v0.7;
- provider-projection separation amendment v0.8;
- Phase 4 interpretation/overlay PRD v0.9.

No variable, source, feature family, target or outcome receives artificial importance merely because it appeared in an earlier model.

## Phase status

### Phase 0 — canonical data foundation

**Complete enough for research use.** Source identities, units, transformations, status, conflicts and exact annual endpoints are retained with explicit retrospective-vintage caveats.

### Phase 1 — target-independent structural discovery

**Complete.**

Key result: the information set is materially high-dimensional, while the original frozen-PCA geometry was too closely related to data coverage to use directly as the forecasting state.

### Phase 2A — missing-aware measurement state

**Complete as retrospective development evidence.**

- 14,652 eligible representations;
- 4,539,769 genuinely observed cells;
- no missing cells inserted into the fitting objective;
- 8,783 country-year states;
- selected state rank: 96;
- state-distance/coverage correlation reduced from 0.948 to 0.133;
- artificial-masking state-error/uncertainty correlation: 0.712.

### Phase 2B — target-independent state transitions

**Complete as retrospective development evidence.**

- one-year state RMSE improvement over no change: approximately 17.3%;
- two-year state RMSE improvement over no change: approximately 16.9%;
- historical analogues beat no change but remained weaker than regularized transition models.

### Phase 3 — probabilistic future-state forecasting

**Complete as retrospective development evidence.** Detailed report: `docs/research/phase3-completion-2026-09-24.md`.

Phase 3 kept the 96-dimensional state fixed and converted the registered point forecasts into calibrated joint future-state distributions.

- rolling historical forecast rows: 15,290;
- target years represented: 40;
- latest country states simulated: 213;
- one-year transition regions approximately 24.7% narrower at 80% and 26.0% narrower at 95% than calibrated no change;
- two-year transition regions approximately 19.0% narrower at 80% and 19.8% narrower at 95%;
- nominal 80% and 95% empirical coverage gates passed.

WEO 2026–2031 provider projections remain in a separate scenario-only lane and were not used in state fitting, transition fitting, calibration or realized outcomes.

### Phase 4A — interpretation

**First slice complete as retrospective development evidence.** Detailed report: `docs/research/phase4-first-slice-2026-09-24.md`.

All 96 state dimensions now have auditable interpretation cards. A post-hoc orthogonal rotation improves display concentration without changing state-space distances or forecasts.

A compact-theme test did not support replacing the engine with a small pillar set:

- best candidate groups: 6;
- silhouette: 0.0785;
- bootstrap adjusted-Rand stability: 0.8917;
- largest group: 86 of 96 dimensions;
- singleton groups: 2.

The interpretation layer should therefore use dimension cards and observable contributors rather than claiming that the state naturally reduces to a few clean themes.

### Phase 4B — broad continuous-observable overlays

**First slice complete as retrospective development evidence.**

The outcome registry was data-led and included all sufficiently supported historical observables. No named banking outcome was privileged.

- registered feature-horizon cases: 3,713;
- globally eligible: 780;
- executed in at least one registered development window: 639;
- executed in both windows: 349;
- provider-projection rows used: 0.

Each case compared no change, own history, state only and own history plus a regularized state correction.

Across the 639 executed cases:

- hybrid state overlay beat own history on RMSE in 443 cases;
- hybrid overlay beat state only in 563 cases;
- median incremental RMSE improvement versus own history was 0.27%;
- 79 cases met the material/stable threshold;
- material/stable cases had median incremental improvement of 3.40% and maximum improvement of 21.99%;
- 38 cases were materially harmed by adding the state.

The architectural conclusion is selective hybrid use: the state is an incremental information layer, not a universal substitute for each observable's own history.

### Phase 4C — crisis/event overlays

**Deferred pending verified alignment.**

Before fitting a challenger, the following must be aligned on identical country-origin rows:

- verified systemic-event dates;
- right-censoring, active-crisis and cooldown rules;
- the frozen production-classifier benchmark;
- rare-event calibration and false-alert metrics.

The production classifier remains byte-locked and unchanged. Event labels do not feed back into the state.

## Current architecture

The research architecture is now:

**broad observed data → missing-aware state → learned transition → calibrated future-state distribution → dimension/observable interpretation → selectively validated observable and event overlays**

The 96 dimensions remain internal. User-facing outputs should show evidence-backed observable implications, uncertainty, movement and separately governed overlays rather than 96 raw coordinates or a forced small pillar set.

## Validation and execution policy

Phase 4 targeted local engineering tests: 4 passed.

No GitHub Actions workflow was used for iterative Phase 4 development. The measurement state and Phase 3 engine were not refitted.

## Remaining work

The major remaining programme is:

1. align and execute the crisis/event overlay against the frozen production classifier;
2. obtain historical source vintages where feasible;
3. register a genuinely untouched confirmation design;
4. build analyst-facing product views from the Phase 3/4 contracts;
5. run the architecture in shadow beside production;
6. consider integration only after explicit owner approval.

## Production firewall

Phases 1–4 did not alter:

- the selected production crisis classifier;
- the serving risk model;
- the production pillar pipeline;
- production source caches;
- Streamlit production code;
- deployed country scores.

No merge, promotion or deployment is authorized.
