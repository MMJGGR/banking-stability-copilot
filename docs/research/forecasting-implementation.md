# Broad-feature forecasting: execution status

Updated 24 September 2026. Active branch: `research/broad-feature-phase3-probabilistic-2026-09-23`. Parent research PR #28 remains draft. Production baseline `5957ca779dafa21f2e098c819bfb060f43243206` is unchanged.

## Governing requirements

The binding requirements are:

- original broad-feature PRD;
- source-grounded data contract v0.2;
- target-independent discovery plan v0.5;
- Phase 2 measurement/transition PRD v0.6;
- Phase 3 probabilistic PRD v0.7;
- provider-projection separation amendment v0.8.

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

Phase 3 kept the 96-dimensional measurement state fixed and reproduced the Phase 2 point models with maximum absolute metric difference `1.607329824793169e-10`.

Execution evidence:

- rolling out-of-time forecast rows: 15,290;
- target years represented: 40;
- leave-target-year-out calibration rows: 30,580;
- latest country states simulated: 213;
- 800 joint draws per horizon;
- 250 peer-neighbour draws per horizon.

Calibration results:

- one-year transition coverage: 79.95% at the nominal 80% level and 94.95% at 95%;
- two-year transition coverage: 79.93% at 80% and 94.91% at 95%;
- compared with calibrated no change, one-year transition regions are approximately 24.7% narrower at 80% and 26.0% narrower at 95%;
- two-year transition regions are approximately 19.0% narrower at 80% and 19.8% narrower at 95%.

All registered Phase 3 calibration gates passed.

## Execution failure and correction

The first complete local Phase 3 attempt exceeded the execution window because BLAS/OpenMP libraries created excessive CPU-thread fan-out while holding the joint-simulation arrays. The model and dataset did not fail.

The identical 800-draw run completed in approximately 34.6 seconds after limiting numerical libraries to two threads. `scripts/run_phase3_research.sh` now applies those deterministic resource limits.

## WEO provider projections

Audit report: `docs/research/weo-projection-audit-2026-09-24.md`.

The current WEO response contains 46,388 provider-projection rows for 2026–2031, representing 204 entities and 145 indicators.

They are kept in a separate provider-projection lane and are prohibited from:

- historical state fitting;
- transition targets;
- transition calibration;
- realized outcomes;
- model selection;
- historical backtests using the current vintage.

Phase 1, Phase 2 and the Phase 3 baseline read zero provider-projection rows. Optional WEO-conditioned scenarios remain separate from the model-only baseline.

WEO observations at or before the cutoff, especially 2025, remain `historical_or_estimate_unverified` because the feed does not expose a complete actual/estimate boundary for every indicator.

## Current architecture

The research architecture is now:

**broad observed data → missing-aware state → learned transition → calibrated future-state distribution → coherent peer simulation**

Outputs include:

- one- and two-year point forecasts;
- calibrated 50%/80%/95% state regions;
- joint common-shock simulations;
- future peer-position distributions;
- historical analogue paths;
- standardized observable implications.

## Remaining work

The next major milestone is not another architecture-expansion phase. It is a clean confirmation and productization programme:

1. register a genuinely untouched confirmation design;
2. obtain historical source vintages where feasible;
3. build analyst-facing interpretation of the 96-dimensional state;
4. introduce supervised crisis/observable overlays only as separate layers;
5. run the new architecture in shadow beside production;
6. consider integration only after explicit owner approval.

## Production firewall

Phases 1–3 did not alter:

- the selected production crisis classifier;
- the serving risk model;
- the production pillar pipeline;
- production source caches;
- Streamlit production code;
- deployed country scores.

No merge, promotion or deployment is authorized.
