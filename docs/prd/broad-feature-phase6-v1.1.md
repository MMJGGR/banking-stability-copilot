# PRD v1.1 — Phase 6: production replacement challenge

Date: 2026-09-24  
Status: active research requirements  
Branch: `research/broad-feature-phase6-replacement-challenge-2026-09-24`  
Parent Phase 5 head: `f07ee156a0e91b1a2103bc7e858e825302a80304`  
Production changes: none authorized

## 1. Purpose

The original programme objective is to determine whether the broad-data architecture can **replace** the current production risk architecture, not merely sit beside it permanently.

Phases 1–5 built and froze a stronger analytical backbone:

- broad missing-aware state;
- historical trajectory;
- one- and two-year state forecasts;
- calibrated uncertainty;
- joint peer simulation;
- country-specific explanation;
- deterministic shadow contract and prospective ledger.

That backbone has not yet produced and validated a complete replacement for production's core decision outputs:

- a 1–10 country risk score;
- five risk categories;
- a calibrated banking-system stress/crisis probability;
- an operating threshold and analyst-review policy.

Phase 6 builds that final decision layer and subjects it to a direct replacement contest. A challenger advances only if it beats the production architecture on the registered decision objectives and operational safeguards. Complementarity is the fallback outcome, not the assumed objective.

## 2. Production decision being challenged

The current production system provides:

1. a directionally constrained economic pillar;
2. a directionally constrained banking/industry pillar;
3. an equal pillar blend;
4. KNN imputation and confidence regression toward median risk;
5. risk floors and critical-field missingness penalties;
6. an upward-only legacy crisis-probability uplift;
7. a 1–10 score and five fixed categories.

The served classifier is locked but not cleanly revalidated. Its preservation is a governance requirement, not evidence that it is an unbeatable benchmark.

Phase 6 must distinguish:

- **served-production snapshot:** the actual current 2026 score and category;
- **production-architecture historical reconstruction:** the production pillar and crisis-overlay rules re-executed under time-ordered historical conditions;
- **replacement challenger:** the new broad-state decision layer.

The historical reconstruction is a benchmark for the architecture. It must not be described as a vintage-clean reproduction of what the deployed app displayed at every historical date.

## 3. Replacement outputs

The challenger must produce the following outputs on the same country-year rows used for evaluation.

### 3.1 Forward system-stress probability

`system_stress_probability_2y`

A calibrated probability that the banking system experiences material deterioration over the next two years.

### 3.2 Replacement risk score

`replacement_risk_score`

A 1–10 score derived from the calibrated forward stress probability against a frozen training-reference distribution. Higher means more risk.

The score may be relative, as production is relative, but the mapping must be learned only from the training period of each validation fold.

### 3.3 Replacement risk category

Use the existing production labels for operational comparability:

- 1–2: Very Low Risk;
- 3–4: Low Risk;
- 5–6: Moderate Risk;
- 7–8: High Risk;
- 9–10: Very High Risk.

The category mapping may not be altered after seeing an outer test period.

### 3.4 Crisis probability

`replacement_crisis_probability_1_3y`

A separately calibrated probability of official systemic banking-crisis onset in years 1–3. It is not automatically identical to the broader system-stress probability.

### 3.5 Review policy

A frozen analyst-review threshold with reported:

- event/stress recall;
- precision;
- false alerts per 100 country-years;
- false alerts per true alert;
- regional and data-coverage burden.

## 4. Material system-deterioration target

Production's structural score has no single directly observed ground-truth label. Phase 6 therefore registers a transparent forward decision target rather than fitting the new score to the existing production score.

A country-origin is positive for `material_system_deterioration_2y` when either:

1. an official systemic banking-crisis onset occurs in years 1–3; or
2. at least two evidence-backed observable families deteriorate materially within two years.

The observable families are taken from the Phase 4 broad screen, not selected before that screen:

- real activity: real GDP growth;
- banking earnings: ROA, ROE and net-income measures;
- fiscal position: primary balance and related validated fiscal measures;
- external activity: export/import volume growth.

Ambiguous-direction outcomes are not used to define the target merely because they passed an accuracy screen. They remain available for interpretation.

For every outer fold:

- family transformations, medians, dispersion and deterioration thresholds are estimated only from the outer training period;
- a family is materially worse when its signed two-year change falls below the registered training-tail threshold;
- the default tail is the worst 20% of training changes;
- at least two families must be observed to classify non-crisis deterioration;
- crisis onset remains positive regardless of observable coverage.

Mandatory sensitivities:

- crisis-only target;
- observable-deterioration-only target;
- one-family and three-family deterioration thresholds;
- 15%, 20% and 25% training-tail thresholds.

The default target advances only if its conclusion is not an artifact of one threshold choice.

## 5. Information available to the challenger

The full broad measurement state remains fixed. Crisis/stress labels do not refit or rotate the 96-dimensional state.

Candidate decision features may include:

- current 96-dimensional state;
- exact one-year state velocity;
- state acceleration where available;
- state-information uncertainty and observed share;
- registered one- and two-year point state changes;
- forecast movement radii and relative-peer movement probabilities;
- crisis-specific raw features constructed under the exact historical availability policy;
- raw-feature missingness/age indicators;
- country-independent global-state context.

No WEO 2026–2031 provider projection may enter baseline features, targets, calibration or realized outcomes. Provider-conditioned scenarios remain separate.

## 6. Candidate decision models

The initial ladder is deliberately limited and interpretable.

### A. State-only regularized logistic model

Tests whether the broad state, trajectory and uncertainty are sufficient.

### B. Crisis/raw-feature regularized logistic model

Uses the exact crisis-specific historical feature panel without the broad state.

### C. State-plus-raw regularized logistic model

Combines the broad state with crisis-specific raw information.

### D. Histogram gradient-boosting challenger

Tests nonlinear interactions without introducing a deep sequence model. It must use the same rows and outer folds as the regularized models.

### E. Production-architecture reconstruction

Rebuilds the constrained two-pillar score on each historical origin using only information available by that origin, then applies the locked production classifier where technically compatible.

The reconstruction must preserve the production rules:

- declared feature directions;
- constrained components;
- equal economic/industry blend;
- confidence adjustment;
- risk floors;
- critical-field penalty;
- upward-only crisis uplift.

It is the direct architecture benchmark, not a target for challenger training.

## 7. Validation design

### 7.1 Time ordering

Use expanding historical training and later outer test windows. The initial registered development windows are:

- 2014–2018;
- 2019–2022.

All features, target thresholds, preprocessing, calibration, probability-to-score mapping, category thresholds and alert thresholds are fitted inside the outer training period.

### 7.2 Country grouping

Inner tuning and calibration must avoid placing the same country on both sides where practical. Time ordering takes precedence; country-grouped sensitivity is reported separately.

### 7.3 Current-snapshot reconciliation

For the 2026 overlap population:

- compare challenger and served production score/category;
- report score deltas and category migrations;
- explain the largest changes using research and production attribution;
- do not use current production values to fit the challenger.

### 7.4 No final-confirmation claim

The Phase 6 retrospective contest remains development evidence because historical source vintages are incomplete and the broad state uses latest-vintage history.

The Phase 5 prospective batch remains the forward confirmation vehicle. Phase 6 may freeze a replacement challenger for later prospective evaluation but may not claim production victory from retrospective evidence alone.

## 8. Primary metrics

### 8.1 System-deterioration probability

- Brier score;
- log loss;
- PR-AUC;
- ROC-AUC;
- calibration slope/intercept;
- 50%/80%/95% reliability bands where practical;
- precision and recall at the frozen review threshold;
- false alerts per 100 country-years;
- event recall by deterioration family.

### 8.2 Risk score and category

- monotonic deterioration/event rate across the five categories;
- pairwise ranking concordance;
- category stability under source/missingness perturbation;
- category stability through adjacent annual origins;
- correlation with data coverage and uncertainty;
- regional and income-group concentration;
- score/category movement relative to production.

### 8.3 Crisis probability

- Brier score;
- log loss;
- PR-AUC;
- ROC-AUC;
- event recall;
- false-alert burden;
- calibration by crisis epoch and coverage group.

## 9. Replacement advancement gates

A challenger does not advance merely because it is the best challenger.

### 9.1 System-stress score gate

Against the production-architecture reconstruction on identical outer-test rows, the challenger must:

1. improve aggregate Brier score by at least 5%;
2. improve aggregate log loss by at least 5%;
3. not reduce PR-AUC by more than 2%;
4. show non-decreasing observed deterioration rates across ordered risk categories, allowing at most one adjacent statistical tie;
5. not increase false alerts per true deterioration by more than 10% at the registered recall floor;
6. retain the result in both registered outer windows;
7. retain the conclusion across the mandatory target sensitivities;
8. show materially lower dependence on data coverage than production.

### 9.2 Crisis-probability gate

The replacement crisis model must:

1. beat the historical event-rate baseline on Brier score and log loss;
2. beat the production-architecture crisis benchmark where matched historical predictions are technically valid;
3. preserve or improve PR-AUC;
4. meet the registered recall floor without an unusable false-alert burden;
5. pass both later outer windows.

If no crisis challenger passes, the production classifier remains separately locked and the structural replacement score may still be evaluated independently.

### 9.3 Operational gate

Before any promotion proposal:

- all outputs reproduce from immutable inputs;
- a rollback bundle exists;
- production and challenger populations reconcile;
- no provider projection entered baseline evidence;
- largest score migrations have analyst-readable explanations;
- owner approval is explicit.

## 10. Possible Phase 6 conclusions

Phase 6 is allowed to conclude any of the following:

1. **Full replacement candidate:** challenger score/category and crisis probability both pass.
2. **Structural-score replacement candidate:** score/category pass, crisis model does not; retain production crisis classifier temporarily.
3. **Partial component replacement:** only selected interpretation/forecast components pass.
4. **No replacement:** challenger does not beat production; retain research as complementary.

The conclusion is determined by evidence, not by the programme's desired destination.

## 11. Deliverables

- historical production-architecture reconstruction ledger;
- material-deterioration target ledger and sensitivity audit;
- challenger out-of-time probability ledger;
- risk-score/category ledger;
- calibration and alert-burden reports;
- current 2026 score/category migration report;
- segment diagnostics;
- replacement gate decision;
- frozen challenger bundle if and only if a gate passes;
- explicit rejected-output ledger otherwise.

## 12. Execution policy

Develop and test locally/synthetically where possible.

Do not use GitHub Actions as an iterative development loop. One consolidated real-data checkpoint is allowed only when the Phase 6 implementation is closure-ready.

No fresh source retrieval is required for the architecture challenge. Use the immutable research artifacts and current governed production artifacts.

## 13. Production firewall

Phase 6 may not alter or deploy:

- production `app.py`;
- current country scores;
- production source caches;
- the selected production classifier;
- the production inference pipeline;
- the served risk model.

No merge, promotion or deployment is authorized by this PRD.
