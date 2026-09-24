# Broad-feature forecasting: execution status

Updated 24 September 2026. Active branch: `research/broad-feature-phase4-interpretation-overlays-2026-09-24`. Parent research PR #28 remains draft. Production baseline `5957ca779dafa21f2e098c819bfb060f43243206` is unchanged.

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

**Complete.** The information set is materially high-dimensional, while the original frozen-PCA geometry was too closely related to data coverage to use directly as the forecasting state.

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

- rolling out-of-time forecast rows: 15,290;
- target years represented: 40;
- leave-target-year-out calibration rows: 30,580;
- latest country states simulated: 213;
- one-year transition regions are approximately 24.7% narrower at 80% and 26.0% narrower at 95% than calibrated no change;
- two-year transition regions are approximately 19.0% narrower at 80% and 19.8% narrower at 95%;
- all registered calibration gates passed.

### Phase 4A — interpretation

**Complete as retrospective research evidence.** Detailed report: `docs/research/phase4-completion-2026-09-24.md`.

A 96-dimensional orthogonal interpretation rotation preserves country distances and measurement reconstruction to numerical precision. It provides country-specific observable attribution without changing the state or forecasts.

The rotation does not justify publishing 96 fixed theme names: dimensions remain broad, overlapping and dominated by the largest banking balance-sheet source. Machine labels require analyst review.

### Phase 4B — broad observable validation

**Complete as retrospective development evidence.**

- registered annual level identities: 4,924;
- observed/evaluated identities: 1,847;
- observed historical cells: 692,406;
- outcome/horizon evaluations admitted in both windows: 1,061;
- stable outcome/horizon evaluations after both-window consistency and FDR control: 14;
- unique stable identities: 13;
- provider projections used: zero.

The state is useful for system-level condition and trajectory but is not a universal raw-indicator forecasting engine. Stable observable value is concentrated in selected GDP growth, profitability, primary-balance/output-gap, trade-volume and one monetary-balance-sheet outcome, mainly at two years.

### Phase 4C — systemic-crisis overlay

**Complete with a negative advancement decision.**

Official Laeven–Valencia systemic labels produced 6,980 eligible country-year rows and 378 positive onset rows after active-crisis, cooldown and right-censoring exclusions.

Aggregate later-window results:

- event-rate baseline Brier score: **0.0342**;
- state-only overlay Brier score: **0.0446**;
- state + velocity + uncertainty Brier score: **0.0512**.

Neither state challenger improved Brier score or log loss. No state-based crisis overlay advances, and no Phase 4 latest crisis probabilities are admissible. The production crisis classifier remains locked and unchanged.

## WEO provider projections

The current WEO response contains 46,388 provider-projection rows for 2026–2031. They remain in a separate provider-projection lane and are prohibited from historical state fitting, targets, calibration, realized outcomes and model selection.

Phases 1–4 used zero provider-projection rows in their baseline historical evidence. Optional WEO-conditioned scenarios remain separate from the model-only baseline.

## Current architecture

The supported research architecture is now:

**broad observed data → missing-aware state → learned transition → calibrated future-state distribution → coherent peer simulation → country-specific interpretation**

Exact observable forecasts and rare-event probabilities remain separately governed overlays. Phase 4 shows that neither should be assumed to work universally merely because the common state forecasts well.

## Next milestone

The next natural milestone is **shadow integration and prospective confirmation**, not another wholesale model redesign:

1. freeze and version the completed research stack;
2. define a research serving contract beside production;
3. expose current state, trajectory, probability regions and observable attribution in shadow mode;
4. preserve current production scores and crisis classifier as independent benchmarks;
5. evaluate future incoming source vintages against frozen forecasts;
6. maintain WEO-conditioned scenarios as a visibly separate provider lane;
7. require explicit owner approval before any production substitution.

## Production firewall

Phases 1–4 did not alter:

- the selected production crisis classifier;
- the serving risk model;
- the production pillar pipeline;
- production source caches;
- Streamlit production code;
- deployed country scores.

No merge, promotion or deployment is authorized.
