# PRD v1.1 — Phase 6: replacement risk-rating challenge

Date: 2026-09-25  
Status: active research requirements  
Branch: `research/broad-feature-phase6-replacement-challenge-2026-09-25`  
Parent Phase 5 head: `f07ee156a0e91b1a2103bc7e858e825302a80304`  
Production changes: none authorized

## 1. Purpose

The original programme objective is to build a better banking-system risk architecture that can replace the current production score **if, and only if, it wins a fair decision-focused comparison**.

Phases 1–5 built and froze a stronger analytical backbone:

- broad, missing-aware banking-system state;
- historical trajectory;
- one- and two-year state forecasts;
- calibrated forecast distributions;
- peer simulations;
- observable attribution;
- shadow-serving and prospective confirmation machinery.

What is still missing is the replacement **decision layer**.

Phase 6 must convert the frozen analytical backbone into a directly comparable risk-rating system organised around the owner's three required questions:

1. **Relative to other countries, how risky is this banking system?**
2. **Relative to its own history, how unusual or deteriorated is its current position?**
3. **How imminent is a banking or sovereign-debt crisis?**

The result must be a coherent risk rating, not another collection of unrelated model outputs.

## 2. Required outputs

For each supported country, the replacement challenger must produce:

- `replacement_risk_score` on the existing 1–10 scale;
- `replacement_risk_category` using the existing five category bands;
- `peer_risk_percentile`;
- `own_history_risk_percentile`;
- `banking_crisis_probability_1y`;
- `banking_crisis_probability_2y`;
- `banking_crisis_probability_3y`;
- `sovereign_distress_probability_1y`;
- `sovereign_distress_probability_2y`;
- `sovereign_distress_probability_3y`;
- `any_systemic_event_probability_1y`;
- `any_systemic_event_probability_2y`;
- `any_systemic_event_probability_3y`;
- confidence/information-quality diagnostics;
- dominant drivers and whether they arise from peer position, own-history deterioration, banking-event risk or sovereign-event risk;
- explicit comparison with the unchanged production score/category and production crisis probability.

The rating must remain interpretable enough to reconcile from these components.

## 3. Core architecture

Phase 6 uses a layered decision architecture.

### 3.1 Frozen analytical backbone

The Phase 2–5 state architecture remains fixed during the replacement challenge:

- 96-dimensional missing-aware state;
- exact country-year trajectory and velocity;
- Phase 3 uncertainty and future-state distribution;
- Phase 4 observable attribution;
- Phase 5 frozen research serving contract.

The state is not refit merely to improve the decision-layer results.

### 3.2 Peer-relative vulnerability model

Estimate a forward-looking risk index from:

- the fixed current state;
- state velocity and acceleration where observed;
- crisis-specific observed banking, macro, sovereign and external-liquidity features;
- information quality and state uncertainty;
- no provider projections.

Convert each forecast-origin year's model risk index into a percentile among countries scored in that same year.

This produces `peer_risk_percentile` and answers:

> How risky does this country look relative to the other countries that could have been assessed at the same date?

Cross-sectional percentiles must use only contemporaneous eligible countries. Future countries or future values cannot enter.

### 3.3 Own-history deterioration model

For every country-year, calculate the current risk index relative to that country's earlier risk-index history.

Inputs include:

- expanding-window own-history percentile;
- change in risk index;
- one- and three-year velocity where exact calendar observations exist;
- acceleration;
- distance from the country's earlier median state;
- uncertainty-aware support.

This produces `own_history_risk_percentile` and answers:

> Is this country in a materially worse position than is normal for itself?

A country with insufficient earlier history must receive an explicit `insufficient_own_history` status rather than a fabricated neutral percentile.

### 3.4 Banking-crisis hazard

Use official Laeven–Valencia 1970–2025 systemic banking-crisis onsets.

Compare, on identical time-ordered rows:

1. historical event-rate baseline;
2. production-style crisis-feature benchmark;
3. fixed state only;
4. state plus velocity/uncertainty;
5. production-style raw crisis features plus fixed state/trajectory;
6. a stacked challenger that may use the locked production classifier probability **only where matched historical inference can be reproduced without retraining or leakage**.

The state itself remains fixed. Banking labels may train only this separate hazard layer.

Outputs are calibrated one-, two- and three-year onset probabilities.

### 3.5 Sovereign-distress hazard

Build a separate sovereign-debt-distress onset model.

Primary public label source:

- Bank of Canada–Bank of England Sovereign Default Database, 2025 edition, covering 1960–2024;
- source definition: debt is in distress/default when scheduled debt service is interrupted, contractual terms are renegotiated, or both;
- the source file and its SHA-256 must be pinned in the evidence artifact.

A public World Bank 2026 debt-distress reproducibility extract may be used as a sensitivity or source cross-check. It may not silently replace the primary label definition.

An onset occurs when a country moves from no debt in default to positive debt in default after applying the registered episode-gap rule. Active-default years, the post-event cooldown and right-censored origins are excluded.

The model may use:

- fixed state and trajectory;
- government debt/revenue and interest/revenue measures;
- primary/fiscal balance;
- external debt service and public financing need;
- reserves/imports and reserves/current-account-payments;
- current account and external liabilities;
- exchange-rate/inflation pressure;
- governance/institutional measures;
- sovereign-bank exposure measures;
- information quality.

Current-vintage WEO values dated after the source cutoff remain provider scenarios and are prohibited from fitting, targets and historical validation.

Outputs are calibrated one-, two- and three-year sovereign-distress onset probabilities.

### 3.6 Any-systemic-event probability

Train a direct combined-event challenger where the target is:

- banking-crisis onset;
- sovereign-distress onset;
- or both;

within the registered horizon.

Also calculate the transparent probability union from the separate hazards. Because banking and sovereign crises are dependent, the simple independence formula may be reported only as a sensitivity. The primary `any_systemic_event_probability` must come from a directly calibrated combined-event model or a validated dependence adjustment.

## 4. Risk score construction

The replacement score has three visible components.

### 4.1 Relative structural position

For each country-year:

- `peer_component = peer_risk_percentile`;
- `history_component = own_history_risk_percentile` when supported.

The blend weight is selected inside training data only from a transparent grid:

`peer_weight ∈ {0.0, 0.1, ..., 1.0}`

The selected weight minimizes time-ordered combined-event Brier score, with ties choosing the more balanced weight closest to 0.5.

When own history is insufficient, the peer component is used and the missing history is disclosed.

### 4.2 Absolute imminence component

Map the calibrated three-year `any_systemic_event_probability` into its percentile in the training-period forecast distribution.

This creates an absolute-event-risk component on the same 0–1 scale while retaining the raw probability for users.

### 4.3 Final score

Define:

`relative_component = w * peer_component + (1 - w) * history_component`

`risk_intensity = max(relative_component, imminence_component)`

`replacement_risk_score = 1 + 9 * risk_intensity`

rounded to one decimal and bounded to `[1, 10]`.

The maximum is deliberate: a high calibrated crisis probability may not be averaged away by a benign peer or historical comparison.

The score categories remain compatible with production:

- `1–2`: Very Low Risk;
- `>2–4`: Low Risk;
- `>4–6`: Moderate Risk;
- `>6–8`: High Risk;
- `>8–10`: Very High Risk.

### 4.4 Confidence treatment

Information uncertainty must not mechanically make a country look safer.

Report score uncertainty and confidence separately. A low-information case may receive a conservative floor only if the floor improves time-ordered decision performance and is selected inside training data. No untested manual penalty may be introduced.

## 5. Outcome and label governance

### 5.1 Banking events

- official systemic episodes only;
- borderline events excluded in the primary specification;
- active event years excluded;
- three-year post-event cooldown;
- right-censored origins excluded;
- exact event onset and horizon recorded.

### 5.2 Sovereign events

- primary source file pinned and checksummed;
- positive debt-in-default values converted to annual default status;
- event onset is the start of a registered contiguous episode;
- small technical residual defaults are tested under materiality sensitivities rather than silently counted or discarded;
- domestic arrears and external default are reported separately where source structure permits;
- the primary outcome definition must be specified before model comparison.

### 5.3 Combined events

The event ledger must preserve:

- banking only;
- sovereign only;
- joint/overlapping event;
- no event;
- active-event exclusion;
- cooldown exclusion;
- right-censored exclusion.

## 6. Candidate model families

Begin with regularized, auditable models:

- logistic/discrete-time hazard regression;
- ridge logistic regression;
- elastic-net logistic regression;
- monotonic calibrated score blending;
- optional gradient-boosted challenger only after the regularized models are stable.

Deep neural networks are not the default. The historical event count is limited and Phase 4 showed that complexity does not guarantee better rare-event probabilities.

All preprocessing, feature selection, penalty tuning and calibration must be fit inside training windows only.

## 7. Time-ordered validation

Use expanding historical development windows with explicit purging for the forecast horizon.

At minimum:

- early development windows spanning pre-2008 episodes;
- 2008–2014 stress window;
- 2015–2022 later window;
- an untouched confirmation period registered separately if feasible.

No shuffled cross-validation.

Report metrics by:

- horizon;
- event type;
- era;
- region/income group where sample permits;
- data-coverage quartile;
- current risk category;
- event/non-event country.

## 8. Primary metrics

### 8.1 Probabilities

- Brier score;
- log loss;
- calibration intercept/slope;
- 50%/80%/95% reliability diagnostics where applicable;
- PR AUC;
- ROC AUC;
- event recall;
- precision;
- false alerts per true alert;
- lead time before onset.

### 8.2 Ranking and rating

- Spearman rank association with later adverse outcomes;
- category transition stability;
- event rates by score/category;
- monotonicity of future event rates across score bands;
- own-history warning lead time;
- peer-ranking stability under masking and source ablation;
- correlation with data coverage.

### 8.3 Production comparison

On matched latest countries:

- score/rank/category reconciliation;
- category changes;
- major divergence review;
- data-coverage sensitivity;
- explanation comparison.

On matched historical rows, where reproducible:

- production crisis probability versus challenger probabilities;
- production-style architecture benchmark versus challenger;
- identical outcome definitions and windows.

If matched historical production predictions cannot be reconstructed without changing the locked artifact, that limitation must remain explicit. Do not retrain the production classifier and call it the locked model.

## 9. Advancement gates

The replacement challenger cannot advance unless all applicable gates pass.

### Banking hazard

- improves Brier score and log loss versus event rate and production-style benchmark;
- does not reduce PR AUC materially;
- false-alert burden is not worse at the registered recall floor;
- calibration is acceptable in later windows.

### Sovereign hazard

- improves Brier score and log loss versus event rate and a transparent debt-vulnerability baseline;
- provides useful event recall at an acceptable false-alert burden;
- later-window calibration is acceptable.

### Combined decision score

- future event rates rise monotonically by risk category;
- High/Very High categories have materially higher later event incidence than Low/Very Low;
- improves combined-event Brier/log loss versus the best single-event and production-style benchmark;
- score is not dominated by data coverage;
- masking and source-ablation results are stable enough for governance;
- major current-country divergences are explainable.

Failure of any core gate means `do_not_replace_production`.

## 10. Phase 6 deliverables

### Deliverable 6.1 — decision contract and labels

- Phase 6 PRD;
- banking and sovereign event ledgers;
- source provenance and checksums;
- exclusion/censoring audit;
- outcome prevalence by horizon.

### Deliverable 6.2 — decision feature frame

- fixed state features;
- own-history features;
- crisis-specific raw features;
- sovereign-specific raw features;
- production benchmark fields;
- missingness/information diagnostics;
- provider-projection firewall.

### Deliverable 6.3 — hazard models

- banking hazard comparison;
- sovereign hazard comparison;
- combined-event comparison;
- time-ordered tuning and calibration evidence.

### Deliverable 6.4 — replacement score

- learned peer/history blend;
- probability-imminence mapping;
- 1–10 score and categories;
- latest country ratings and component bridge;
- divergence report versus production.

### Deliverable 6.5 — final challenge decision

One of:

- `replacement_challenger_passes_development_gates_shadow_only`;
- `partial_components_pass_keep_hybrid`;
- `do_not_replace_production`;
- `blocked_by_missing_authoritative_labels_or_matched_benchmark`.

No result authorizes deployment.

## 11. Execution policy

- work on the Phase 6 child branch;
- use local/synthetic tests during development;
- use one consolidated real-data checkpoint when closure-ready;
- avoid repeated GitHub Actions runs;
- no fresh five-source retrieval solely for model development;
- pin any new official label source and preserve its raw checksum in the evidence artifact;
- production files remain read-only.

## 12. Production firewall

Phase 6 may not modify, merge, promote, deploy or retrain:

- production `app.py`;
- production pillar pipeline;
- serving source caches;
- deployed country scores;
- production risk model;
- selected production crisis classifier.

The selected production classifier remains expected at:

- bytes: `52,580`;
- SHA-256: `054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee`.

Any production substitution requires a separate explicit owner authorization after the Phase 6 evidence is reviewed.
