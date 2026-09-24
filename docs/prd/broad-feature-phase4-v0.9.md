# PRD v0.9 — Phase 4: interpretation and supervised overlays

Date: 2026-09-24  
Status: active research requirements  
Branch: `research/broad-feature-phase4-interpretation-overlays-2026-09-24`  
Parent Phase 3 head: `ad00ee3569d8bdb00b75ca7df37ced59a13f5126`  
Production changes: none authorized

## 1. Purpose

Phases 1–3 established a broad, missing-aware, 96-dimensional banking-system state; demonstrated target-independent one- and two-year state dynamics; and converted those forecasts into calibrated future-state distributions and coherent peer simulations.

Phase 4 makes that internal architecture useful to an analyst without allowing a small hand-picked outcome set to redefine the state.

It answers three separate questions:

1. **What does the learned state mean in observable economic and banking terms?**
2. **Which observable source series can the state forecast beyond simple persistence?**
3. **Does the state and its movement add value for separately governed event questions such as systemic banking-crisis onset?**

The state remains fixed. Supervised outcomes and events are overlays used to interpret or test the state; they do not select, refit, or constrain the Phase 2 measurement architecture.

## 2. Binding owner principle

No variable, indicator, source, feature family, target or outcome receives artificial emphasis or exclusion because it is familiar, appears in the current production model, or was used in an earlier experiment.

Phase 4 must inventory the complete eligible annual observable universe and apply only source-semantic, data-quality and minimum-evidence rules. Named outcomes may be highlighted only after the broad results are available and must be shown alongside the full audit population.

## 3. Fixed inputs

Phase 4 starts from the preserved Phase 2 and Phase 3 evidence:

- 96-dimensional fixed-loading measurement state;
- 8,783 country-year states;
- 14,652 eligible predictor representations;
- measurement loadings, robust feature scales and reliability diagnostics;
- one- and two-year fixed point-transition models;
- calibrated Phase 3 future-state distributions;
- source metadata and the immutable 16 September 2026 research retrieval.

Phase 4 must reproduce relevant upstream hashes and does not retune the measurement-state dimension or Phase 3 probability model.

## 4. Three separate data lanes

### 4.1 Historical or observed lane

Only retrospective historical observations that satisfy the source data contract may be used for observable-outcome evaluation, event overlays or realized outcomes.

### 4.2 Provider-projection lane

Current-vintage WEO values dated 2026–2031 remain provider projections. They are prohibited from:

- state construction;
- outcome targets;
- event targets;
- historical model selection;
- backtest outcomes;
- forecast-error calibration.

They may later be displayed as provider benchmarks or used in an explicitly labelled conditional scenario.

### 4.3 Copilot forecast lane

The Copilot baseline and its supervised overlays remain independent model outputs. No automatic averaging or blending with provider projections is permitted.

## 5. Phase 4A — interpretation of the learned state

### 5.1 Objective

Turn the 96-dimensional internal state into stable analyst-readable evidence while preserving the geometry and forecasts learned in Phases 2–3.

### 5.2 Orthogonal interpretation rotation

Apply an orthogonal sparse/simple-structure rotation to the fixed measurement loadings. The rotation may make dimensions easier to interpret but must preserve:

- pairwise country-state distances;
- total state information;
- measurement reconstruction under the same rotated state/loadings;
- Phase 2 point predictions after applying the same coordinate rotation;
- Phase 3 probability geometry after applying the same rotation.

The rotation is an interpretation coordinate system, not a refit of the state.

### 5.3 Interpretation evidence

For every rotated dimension export:

- largest positive and negative observable loadings;
- source and indicator metadata;
- contribution concentration and effective number of contributing representations;
- reliability and observation support;
- representation mix (level, lag, change);
- source concentration;
- stability caveats.

Machine-generated theme names must be labelled `descriptive_candidate_label` and require analyst review. No theme label changes the model.

### 5.4 Country explanation

For each latest country state and forecast horizon export:

- dimensions contributing most to current distinctness;
- dimensions contributing most to expected movement;
- top observable representations associated with those dimensions;
- distinction between state direction, relative-peer direction and supervised risk direction.

A latent dimension is not automatically a risk factor.

## 6. Phase 4B — broad observable-outcome validation

### 6.1 Outcome universe

Inventory every eligible annual level identity from FSIC, FSIBSIS, MFS, WEO and WGI.

An identity remains separate when source dimensions differ, including sector, transformation, unit, frequency, scale, consolidation or accounting basis.

All identities receive an outcome-admission state, including:

- eligible;
- provider projection only;
- unresolved unit/semantics;
- conflict quarantined;
- insufficient country/year support;
- constant or non-finite;
- no exact-calendar future pair.

There is no economic allowlist or top-k screen.

### 6.2 Outcome definition

For an eligible level identity, test the exact-calendar future change relative to the latest observable level available at the forecast origin.

Where the Phase 2 feature transformation is invertible only on a standardized scale, evaluate standardized outcome change rather than claiming exact raw-unit forecasts.

Domestic-currency amounts use the registered causal own-history normalization; raw incomparable currency magnitudes must not be compared across countries.

No interpolation or shortened calendar lag is permitted.

### 6.3 Forecasts being compared

On identical rows compare:

1. **Persistence/no change** — future standardized value equals current standardized value.
2. **State reconstruction forecast** — the fixed Phase 2/3 future-state change is translated through the relevant level-feature loading.
3. **Optional supervised state overlay** — a regularized model using current state and current observable level, fitted only when the identity has sufficient data.

The state-reconstruction forecast is the primary architecture test because it uses one common state-transition engine across the complete observable universe. The supervised overlay is a challenger and must not feed back into the state.

### 6.4 Time ordering

Use exact later observations and time-ordered development windows. All transformation and overlay fitting occurs inside the training period.

The first registered windows remain:

- 2016–2018;
- 2019–2021.

Later periods are not described as an untouched final confirmation merely because they were not used by a particular outcome identity.

### 6.5 Minimum evidence rules

Outcome admission may require minimum numbers of exact future pairs, countries, origins and positive within-series variation. These thresholds exist only to avoid unstable statistics and must be declared before reading model performance.

Every excluded identity remains in the audit ledger with its reason.

### 6.6 Multiple-comparison and stability control

Thousands of outcomes are evaluated. Phase 4 must not present isolated improvements as discoveries without controlling false findings.

Report:

- improvement versus persistence by horizon and window;
- bootstrap or fold uncertainty where feasible;
- sign consistency across registered windows;
- false-discovery-rate-adjusted evidence for broad claims;
- source/family aggregate distributions;
- complete result ledger, including negative results.

An identity is described as `stable_state_predictable` only when improvement is positive across both registered windows or satisfies a preregistered pooled-and-stability rule.

## 7. Phase 4C — systemic-crisis event overlay

### 7.1 Separation from the state

Systemic-crisis onset is a supervised event question. Crisis labels may enter only this overlay and may not refit or orient the Phase 2 state.

### 7.2 Labels and horizon

Use the pinned Laeven–Valencia 1970–2025 systemic episode table. Borderline episodes remain excluded from the primary target and may be tested only as a sensitivity.

Use a separately registered one-to-three-year onset horizon. Exclude active crisis years, post-crisis cooldown years and right-censored observations.

### 7.3 Models

Compare on identical eligible country-years:

- event-rate baseline;
- current-state-only regularized logistic/hazard model;
- current state plus velocity and uncertainty;
- existing production crisis classifier only where a matched historical prediction can be reconstructed without changing its locked artifact.

The production classifier remains byte-locked and is not retrained.

### 7.4 Metrics

Report calibration and rare-event usefulness, including:

- Brier score;
- log loss;
- precision-recall AUC;
- ROC AUC as secondary context;
- recall and false-alert burden at registered review thresholds;
- calibration by time period and data-coverage group.

A state-based event overlay advances only if it improves calibration or alert usefulness on later windows, not merely in-sample discrimination.

## 8. Interpretation versus causality

Loadings, state dimensions, outcome predictability and crisis associations are not causal effects.

Phase 4 outputs must distinguish:

- descriptive association;
- reconstruction relationship;
- forward predictive relationship;
- supervised event association;
- causal claim, which is not made by this phase.

## 9. Acceptance criteria

Phase 4 closes when:

1. the interpretation rotation is demonstrably geometry-preserving;
2. all 96 dimensions have complete loading/support diagnostics;
3. latest country and forecast explanations are exported;
4. the complete annual observable universe has an admission state;
5. broad observable forecasts are compared with persistence on exact later rows;
6. multiple-comparison and cross-window stability results are reported;
7. the crisis overlay is evaluated or explicitly stopped for insufficient matched evidence;
8. WEO 2026–2031 projections remain in the provider-scenario lane only;
9. production app, scores, pillars, source caches and locked classifier remain unchanged;
10. limitations and negative results are preserved.

Phase 4 completion remains retrospective research evidence. It does not authorize production integration.

## 10. Execution policy

Use local execution against the immutable Phase 2/3 evidence and source snapshot. Do not use GitHub Actions for iterative Phase 4 development.

A consolidated repository checkpoint may be considered only after the phase is closure-ready. No fresh IMF or World Bank retrieval is required for this phase.
