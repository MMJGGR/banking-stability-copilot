# Broad-feature forecasting: execution status

Updated 24 September 2026. Active branch: `research/broad-feature-phase5-shadow-confirmation-2026-09-24`. Parent research PR #28 remains draft. Production baseline `5957ca779dafa21f2e098c819bfb060f43243206` is unchanged.

## Governing requirements

The binding requirements are:

- original broad-feature PRD;
- source-grounded data contract v0.2;
- target-independent discovery plan v0.5;
- Phase 2 measurement/transition PRD v0.6;
- Phase 3 probabilistic PRD v0.7;
- provider-projection separation amendment v0.8;
- Phase 4 interpretation/overlay PRD v0.9;
- Phase 5 shadow/confirmation PRD v1.0.

No variable, source, feature family, target or outcome receives artificial importance merely because it appeared in an earlier model.

## Completed research phases

### Phase 0 — canonical data foundation

**Complete enough for research use.** Source identities, units, transformations, status, conflicts and exact annual endpoints are retained with explicit retrospective-vintage caveats.

### Phase 1 — target-independent structural discovery

**Complete.** The information set is materially high-dimensional, while the original frozen-PCA geometry was too closely related to data coverage to use directly as the forecasting state.

### Phase 2 — missing-aware state and state transitions

**Complete as retrospective development evidence.**

- 14,652 eligible representations and 4,539,769 genuinely observed cells;
- no missing cells inserted into the fitting objective;
- 8,783 country-year states and a selected 96-dimensional state;
- state-distance/coverage correlation reduced from 0.948 to 0.133;
- one-year state RMSE improvement over no change: approximately 17.3%;
- two-year state RMSE improvement over no change: approximately 16.9%.

### Phase 3 — probabilistic future-state forecasting

**Complete as retrospective development evidence.** Detailed report: `docs/research/phase3-completion-2026-09-24.md`.

- 15,290 rolling out-of-time forecasts and 30,580 leave-target-year-out calibration rows;
- 213 latest country states;
- one-year transition regions approximately 24.7% narrower at 80% and 26.0% narrower at 95% than calibrated no change;
- two-year transition regions approximately 19.0% narrower at 80% and 19.8% narrower at 95%;
- all registered calibration gates passed.

### Phase 4 — interpretation and separately governed overlays

**Complete as retrospective research/development evidence.** Detailed report: `docs/research/phase4-completion-2026-09-24.md`.

- an orthogonal interpretation rotation preserves state geometry and reconstruction;
- 4,924 annual level identities were registered and 1,847 observed identities evaluated;
- only 14 outcome/horizon evaluations across 13 identities passed both-window stability and false-discovery controls;
- the state is not a universal raw-indicator forecasting engine;
- state-only and state-plus-velocity crisis challengers underperformed the historical event-rate baseline on Brier score and log loss;
- no state-based crisis overlay advances;
- the production classifier remains locked and unchanged.

## Phase 5 — shadow integration and prospective confirmation

### Phase 5A — deterministic shadow-serving bundle

**Complete.** The first research batch is frozen as:

`shadow-1ef71a2f20bb20b625d5`

Freeze record: `docs/research/phase5-shadow-freeze-2026-09-24.json`.

The bundle contains:

- 213 research countries;
- 426 one-/two-year country-horizon records;
- current 96-dimensional states and state-information diagnostics;
- coordinate-level future quantiles;
- calibrated movement regions and future peer distributions;
- historical analogues;
- country-specific current-state and movement explanations;
- standardized observable implications and the Phase 4 stable-outcome catalog;
- an append-only 426-row prospective forecast ledger.

Population reconciliation is explicit:

- 200 countries overlap production and research;
- 13 are research-only;
- one production country (`SXM`) has no current research forecast.

Production score/category/crisis probability remain a separate read-only benchmark lane. They are not blended into the research state or forecast.

### Phase 5B — separate shadow viewer

**Engineering complete; not deployed.**

`research_shadow_app.py` is a separate, read-only Streamlit entry point. It:

- verifies bundle hashes;
- labels all output as research/not production;
- displays production and research in separate panels;
- exposes uncertainty, peer movement, analogues and explanations;
- suppresses the rejected Phase 4 research crisis probability;
- displays WEO availability only as a separate provider-scenario lane;
- makes no network writes and no production artifact writes.

### Phase 5C — prospective confirmation

**Open and necessarily incomplete.**

The 2027 and 2028 forecasts have been locked before eligible future realizations are available. The scoring harness is append-only and rejects provider projections as realized outcomes.

Prospective confirmation can only be evaluated when later source vintages contain verified observed data for the target years. The original forecast records and hashes must remain unchanged.

Local engineering validation completed:

- two independent bundle executions produced the same batch identifier and identical accepted file hashes;
- five focused unit tests passed;
- all 426 normalized rows and 426 nested country records reconciled;
- provider projections were rejected as realizations;
- duplicate realization records were rejected;
- shadow bundle hash validation passed.

No GitHub Actions workflow was used for iterative Phase 5 development.

## WEO provider projections

The current WEO response contains 46,388 provider-projection rows for 2026–2031. They remain in a separate provider lane and are prohibited from historical state fitting, targets, calibration, realized outcomes and model selection.

Phases 1–5 baseline evidence uses zero provider-projection rows. Optional WEO-conditioned scenarios remain separate from the model-only baseline.

## Supported architecture

The supported research architecture is:

**broad observed data → missing-aware state → learned transition → calibrated future-state distribution → coherent peer simulation → country-specific interpretation → frozen shadow forecast**

Exact observable forecasts and rare-event probabilities remain separately governed overlays. The current state-based crisis overlay is rejected.

## Next milestone

The architecture is now frozen for prospective observation rather than immediate redesign.

Next work should be limited to:

1. analyst review of the shadow interface and explanations;
2. operational monitoring of bundle integrity and source-vintage arrival;
3. generation of visibly separate WEO-conditioned scenarios if authorized;
4. prospective scoring when verified 2027/2028 observations become available;
5. comparison with unchanged production benchmarks;
6. explicit owner approval before any production substitution or deployment.

## Production firewall

Phases 1–5 did not alter:

- the selected production crisis classifier;
- the serving risk model;
- the production pillar pipeline;
- production source caches;
- production `app.py`;
- deployed country scores.

No merge, promotion or deployment is authorized.
