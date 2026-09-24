# PRD v0.9 — Phase 4: interpretation and supervised overlays

Date: 24 September 2026  
Status: active research requirements  
Branch: `research/broad-feature-phase4-overlays-2026-09-24`  
Parent Phase 3 head: `ad00ee3569d8bdb00b75ca7df37ced59a13f5126`  
Production changes: none authorized

## 1. Purpose

Phases 1–3 established a broad, missing-aware banking-system state, learned one- and two-year state transitions and produced calibrated future-state distributions.

Phase 4 makes that architecture useful to an analyst without allowing analyst labels or selected outcomes to redefine the underlying state.

Phase 4 has four linked objectives:

1. explain what the 96 internal state dimensions represent in observable terms;
2. determine which future observable outcomes the state improves, using a data-led outcome registry rather than a hand-picked list;
3. add separately governed event overlays, including systemic-crisis risk, without feeding those labels back into the state model;
4. define the analyst-facing contract that separates state, forecast, interpretation, provider scenarios and supervised risk outputs.

The completed Phase 2 measurement state and Phase 3 probability engine remain fixed inputs during this phase.

## 2. Binding architecture rule

No feature, indicator, source, family, target or outcome may receive artificial importance merely because it is familiar, appears in production, or was used in an earlier experiment.

Phase 4 may ask supervised questions of the state. It may not use the answers to reconstruct or retune the Phase 2 state during the same phase.

The architecture therefore remains:

**broad observed data → missing-aware state → state transition → probability distribution → interpretation and supervised overlays**

not:

**selected target → feature selection → target-shaped state**.

## 3. Fixed research inputs

Phase 4 starts from preserved Phase 2 and Phase 3 evidence:

- 96-dimensional fixed-loading country-year state;
- 8,783 country-year state observations;
- 14,652 model-eligible feature representations;
- measurement loadings and feature metadata;
- state-information and uncertainty diagnostics;
- calibrated one- and two-year future-state distributions;
- historical analogue outputs;
- standardized observable-change implications;
- provider-projection ledger kept outside the historical/realized lane.

Phase 4 does not refit the measurement state merely to improve overlay performance.

## 4. Phase 4A — state interpretation

### 4.1 Dimension cards

For every internal state dimension, export an auditable interpretation card containing:

- strongest positive and negative observable loadings;
- source, indicator, unit, sector, transformation and representation metadata;
- contribution concentration by source and indicator family;
- years and country-years supporting the relevant features;
- loading/reliability caveats;
- recent projected observable implications from Phase 3.

A state dimension is not assigned a human label unless its observable composition is sufficiently coherent and stable.

### 4.2 Data-derived themes

Create analyst-facing themes by clustering state dimensions according to their observable loading profiles and metadata composition.

Requirements:

- clustering is post-hoc and does not alter the 96-dimensional state;
- state-space distances and forecasts remain unchanged;
- the number of themes is selected from stability and separation diagnostics rather than fixed to the production pillars;
- every theme retains the full list of member dimensions and observable contributors;
- mixed or unstable groups are labelled `mixed / not reliably interpretable`, not forced into a familiar economic category;
- names are generated only after inspecting the data-derived members.

Candidate methods may include loading-profile cosine similarity, hierarchical clustering and bootstrap/stability checks. A sparse or rotated display representation may be used only if it is explicitly a display transform and preserves the original engine separately.

### 4.3 Country interpretation

For each latest country state and forecast, export:

- leading positive and negative state themes;
- the observed features that most strongly support those themes;
- forecasted theme movement at one and two years;
- uncertainty and data-support flags;
- distinction between absolute movement and relative peer movement;
- historical analogues as context, not causality.

The output must not imply that distance from the peer centre is inherently good or bad.

## 5. Phase 4B — data-led observable overlay registry

### 5.1 Outcome universe

Construct the candidate outcome registry from all historical/observed feature identities that satisfy data-contract and support requirements.

Provider projections are prohibited from realized outcomes, model selection and historical validation.

A candidate observable outcome may enter the registry only when it has:

- resolved identity and unit treatment;
- exact-calendar one- or two-year future observations;
- sufficient countries, years and transition rows;
- non-trivial variation;
- no unresolved source conflict for the target cell;
- an explicit status showing that it is historical/observed or historical-status-unverified, never `provider_projection`.

The eligibility thresholds are support rules, not economic importance rules. All excluded identities remain in an outcome ledger with the reason.

### 5.2 Forecast questions

For every admitted observable, compare on identical later rows:

1. **Own-history baseline:** current observable and its available exact-calendar lag/history.
2. **State-only overlay:** the fixed 96-dimensional current state.
3. **Hybrid overlay:** own history plus the fixed state.
4. **No-change baseline:** future observable equals current observable, where meaningful.

Outcomes should normally be evaluated both as future level and future change when the unit/semantics permit. The registry must record when only one formulation is economically valid.

### 5.3 Validation

Use time-ordered, nested development comparisons.

- transformations and penalties are fit on earlier data only;
- every model is evaluated on identical country-origin rows;
- no provider projection can be used as a target or feature in the baseline overlay run;
- no outcome may be selected using its final evaluation period;
- paired error improvements must be retained, including negative results;
- multiple-outcome discovery must report false-discovery-controlled evidence or a clearly separated exploratory tier;
- broad overlay performance must be reported by source, unit family, data coverage and era.

The purpose is to learn where the state adds value, not to declare every observable forecastable.

## 6. Phase 4C — event and crisis overlays

Systemic crisis and other discrete events are separately governed overlays.

### 6.1 Crisis challenger

Build a state-based discrete-time hazard challenger using:

- current state;
- recent state movement where available;
- state uncertainty/information quality;
- no selected raw indicator unless it is introduced as a separately registered challenger.

The crisis label does not enter the measurement state or Phase 3 probability engine.

### 6.2 Comparators

Compare against:

- a simple historical/base-rate hazard;
- the fixed production crisis classifier where exact comparable inputs and dates are available;
- a state-only regularized hazard;
- a state-plus-velocity hazard;
- any hybrid challenger registered before evaluation.

The selected production classifier remains byte-locked and is not retrained or replaced in Phase 4.

### 6.3 Metrics

Report at minimum:

- Brier score;
- log loss;
- calibration by risk band;
- precision-recall AUC;
- recall at declared false-alert burdens;
- event and country coverage;
- performance by era and information quality.

Rare-event performance and calibration matter more than headline accuracy.

## 7. Phase 4D — analyst-facing output contract

The eventual product must keep the following outputs distinct:

1. **Current state:** a descriptive estimate of the banking system.
2. **Future-state distribution:** the modelled one- and two-year state range.
3. **Interpretation themes:** post-hoc explanations of the state, not construction constraints.
4. **Observable overlays:** forecasts for specific indicators where validated.
5. **Event overlays:** separately calibrated crisis or stress probabilities.
6. **Provider scenarios:** IMF/WEO or other external projections, clearly labelled and never silently blended with the baseline.
7. **Policy/production score:** any governed final rating or action layer, which remains outside this research phase.

No single composite risk score is created in Phase 4 merely for presentation convenience.

## 8. Phase 4 first execution slice

The first executable slice will deliver:

1. state-dimension interpretation cards from the fixed Phase 2 loadings;
2. data-derived theme clustering with stability diagnostics;
3. latest country/theme summaries using Phase 3 forecast implications;
4. a complete observable-outcome admission ledger;
5. time-ordered own-history/state/hybrid comparisons for every sufficiently supported continuous observable;
6. explicit multiple-testing/exploratory labels;
7. crisis-overlay data-contract and target-firewall tests;
8. no production changes and no provider projections in the baseline overlay execution.

The crisis hazard may execute in a second Phase 4 slice if event-label alignment or comparable production-classifier evidence is not yet complete. That dependency must be reported rather than guessed.

## 9. Acceptance criteria

Phase 4 closes when:

1. all 96 state dimensions have auditable interpretation cards;
2. theme groupings have stability and mixed-theme diagnostics;
3. country-level interpretation separates evidence, uncertainty and forecast movement;
4. every eligible historical observable has an admission state;
5. overlay comparisons use fixed state inputs and time-ordered evaluation;
6. negative overlay results are retained;
7. provider projections are read only by explicitly conditional scenario code;
8. crisis/event labels do not alter the underlying state;
9. the fixed production classifier is preserved;
10. production app, scores, caches and serving artifacts are unchanged;
11. results remain retrospective research evidence and are not presented as production validation.

## 10. Execution policy

Do not use GitHub Actions for iterative Phase 4 development.

Develop and test locally against the preserved Phase 2/3 evidence. Use at most one consolidated repository checkpoint after the phase is closure-ready and only if needed.

No fresh IMF or World Bank retrieval is required for the first Phase 4 slice.