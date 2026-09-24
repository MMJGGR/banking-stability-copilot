# PRD v1.2 — Phase 6: replacement rating and production challenge

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

Phase 6 builds the missing decision layer. The new rating must answer three separate questions:

1. **Peer-relative risk:** how risky is the country compared with other countries at the same date?
2. **Own-history stress:** how abnormal or adverse is the country compared with its own prior history?
3. **Event imminence:** how likely is a systemic banking crisis or sovereign default/distress episode within the decision horizon?

Those components remain visible. The final score may combine them only through a registered, monotone policy. An imminent event may raise risk but may never be averaged away by an otherwise average peer position.

A challenger advances only if it beats the production architecture on registered decision objectives and operational safeguards. Complementarity is the fallback outcome, not the assumed objective.

## 2. Production decision being challenged

The current production system provides:

1. a directionally constrained economic pillar;
2. a directionally constrained banking/industry pillar;
3. an equal pillar blend;
4. KNN imputation and confidence regression toward median risk;
5. risk floors and critical-field missingness penalties;
6. an upward-only legacy banking-crisis probability uplift;
7. a 1–10 score and five fixed categories.

The served classifier is locked but not cleanly revalidated. Its preservation is a governance requirement, not evidence that it is an unbeatable benchmark.

Phase 6 must distinguish:

- **served-production snapshot:** the actual current 2026 score and category;
- **production-architecture historical reconstruction:** the production pillar and crisis-overlay rules re-executed under time-ordered historical conditions;
- **replacement challenger:** the new three-axis broad-state decision layer.

The historical reconstruction is an architecture benchmark. It must not be described as a vintage-clean reproduction of what the deployed app displayed at every historical date.

## 3. Replacement outputs

The challenger must produce the following on the same country-year rows used for evaluation.

### 3.1 Peer-relative risk

`peer_relative_risk_index`

A 0–1 same-year percentile of the challenger structural-stress signal. The ranking reference contains only the countries available at that forecast origin.

### 3.2 Own-history stress

`own_history_stress_index`

A 0–1 percentile of the same structural-stress signal against the country's **strictly prior** history. No current or future observation may enter its historical reference distribution.

The output must include:

- prior years available;
- first supported year;
- support status;
- recent velocity and acceleration diagnostics.

Sparse history affects confidence, not the risk score direction.

### 3.3 Banking-crisis imminence

`banking_crisis_probability_1_3y`

A separately calibrated probability of official systemic banking-crisis onset in years 1–3, using the governed Laeven–Valencia 1970–2025 episode artifact.

### 3.4 Sovereign-crisis imminence

`sovereign_crisis_probability_1_3y`

A separately calibrated probability of sovereign default or distressed restructuring onset in years 1–3.

The primary event source is the Bank of Canada–Bank of England Sovereign Default Database, current 2025 edition, which covers country-level sovereign obligations in default from 1960–2024. Before modelling, the source file must be:

- imported from the official Bank of Canada release;
- stored or referenced through a pinned checksum;
- transformed into an onset/episode ledger under a documented rule;
- reviewed for domestic arrears, external private, external official and local-currency scope;
- separated from provider projections and model inputs.

Until that governed episode ledger is present, the sovereign probability is **unavailable**, not zero, and a full replacement decision may not be approved.

A Global Macro Database sovereign-crisis series may be used only as a research sensitivity if its licence permits the intended use. It is not the governed production label source.

### 3.5 Combined event imminence

`banking_or_sovereign_event_probability_1_3y`

Preferred method: a directly calibrated composite-event model using the union of governed banking and sovereign event onsets.

Mandatory sensitivities:

- lower bound: `max(p_banking, p_sovereign)`;
- conditional-independence union: `1 - (1-p_banking)(1-p_sovereign)`.

The system may not silently assume independence. If only one governed event head is available, the combined output must be marked `partial_event_coverage`.

### 3.6 Replacement risk score and category

The final score retains operational comparability with production:

- 1–10 score, higher is riskier;
- 1–2 Very Low;
- 3–4 Low;
- 5–6 Moderate;
- 7–8 High;
- 9–10 Very High.

Its components must remain separately visible.

### 3.7 Evidence confidence

`rating_confidence` and `rating_support_status`

Confidence is reported separately from risk. Low coverage or short history may widen uncertainty or make a rating provisional; it may not automatically make a country safer or riskier unless a separately approved policy floor is applied after model validation.

## 4. Three-axis score policy

Let:

- `R_peer` be peer-relative risk in [0,1];
- `R_history` be own-history stress in [0,1];
- `R_event` be the training-reference percentile of the calibrated combined event probability.

The structural base is:

`R_structural = w_peer * R_peer + (1 - w_peer) * R_history`

The event overlay is monotone and upward-only:

`R_final = R_structural + alpha * max(R_event - R_structural, 0)`

where:

- `w_peer` is selected inside the training sample from a registered grid;
- `alpha` is selected inside the training sample from `{0.25, 0.50, 0.75, 1.00}`;
- `alpha = 1` makes event imminence a full floor;
- the outer test sample may not influence either choice.

The displayed score is:

`replacement_risk_score = 1 + 9 * R_final`

The event probabilities remain displayed as probabilities. The 1–10 score is a relative decision index, not a literal probability.

Mandatory alternatives:

- constrained monotone logistic stack;
- peer/history equal weighting;
- full event floor `max(R_structural, R_event)`;
- no-event structural score.

The simplest policy that meets the advancement gates is preferred.

## 5. Structural-stress signal

The state itself is neutral and must not be relabelled as risk after the fact. A supervised decision head estimates the probability of material system deterioration over two years.

A country-origin is positive for `material_system_deterioration_2y` when either:

1. a governed systemic banking-crisis or sovereign-crisis onset occurs in years 1–3; or
2. at least two evidence-backed observable families deteriorate materially within two years.

The observable families come from the Phase 4 broad screen:

- real activity: real GDP growth;
- banking earnings: ROA, ROE and net-income measures;
- fiscal position: primary balance and related validated fiscal measures;
- external activity: export/import volume growth.

Ambiguous-direction outcomes are not used merely because they passed an accuracy screen.

For every outer fold:

- transformations, medians, dispersion and deterioration thresholds are estimated only from outer training data;
- a family is materially worse when its signed two-year change falls below the registered training-tail threshold;
- the default tail is the worst 20% of training changes;
- at least two observable families must be present for a non-event deterioration label;
- a governed crisis onset remains positive regardless of observable coverage.

Mandatory sensitivities:

- event-only target;
- observable-deterioration-only target;
- one-family and three-family thresholds;
- 15%, 20% and 25% deterioration tails.

## 6. Model architecture

The fixed 96-dimensional measurement state remains neutral. Event and deterioration labels do not refit or rotate it.

### 6.1 Structural-stress head

Candidate features:

- current state;
- exact one-year state velocity;
- acceleration where available;
- state uncertainty and observed share;
- registered one-/two-year expected state movement;
- movement radii and relative-peer transition probabilities;
- global-state context.

### 6.2 Banking-crisis head

Candidate features:

- broad state and trajectory;
- banking asset quality, capital, liquidity, earnings and funding indicators;
- credit-cycle and property-cycle features;
- sovereign-bank exposure;
- raw-feature age, coverage and missingness.

### 6.3 Sovereign-crisis head

Candidate features:

- broad state and trajectory;
- government debt/GDP and debt/revenue;
- interest/revenue and debt-service burden;
- primary/fiscal balance;
- reserves/import and reserves/current-account-payment coverage;
- external debt and public external financing need;
- current account and exchange-rate stress;
- sovereign-bank exposure and banking claims on government;
- governance and institutional indicators;
- raw-feature age, coverage and missingness.

Debt/GDP alone is a mandatory simple benchmark. A more complex sovereign model may not advance unless it beats that benchmark on calibrated probability quality.

### 6.4 Candidate model ladder

For each head, compare on identical rows:

1. historical event/deterioration rate;
2. simple benchmark(s), including debt/GDP for sovereign risk;
3. regularized logistic regression;
4. state-only logistic model;
5. raw-feature-only logistic model;
6. state-plus-raw logistic model;
7. histogram gradient boosting for nonlinear interactions.

Deep sequence models are deferred unless this ladder leaves substantial, stable residual value.

## 7. Validation design

### 7.1 Time ordering

Use expanding historical training and later outer test windows. Initial registered development windows:

- 2014–2018;
- 2019–2022.

All target thresholds, preprocessing, calibration, probability-to-score mapping, peer/history reference distributions, category thresholds and review thresholds are fitted inside the outer training period.

### 7.2 Country grouping

Inner tuning and calibration must avoid placing the same country on both sides where practical. Time ordering takes precedence; country-grouped sensitivity is reported separately.

### 7.3 Event contamination

For each event head:

- active event years are excluded;
- post-event cooldown years are excluded;
- right-censored origins are excluded;
- borderline banking episodes are excluded by default;
- event definitions and source versions are carried in every ledger.

### 7.4 Current 2026 reconciliation

For overlapping countries:

- compare challenger and served production scores/categories;
- show all three challenger components;
- report score deltas and category migrations;
- explain the largest migrations;
- do not use production scores to fit the challenger.

### 7.5 No final-confirmation claim

Retrospective Phase 6 evidence remains development evidence because complete historical source vintages are unavailable. The frozen Phase 5/Phase 6 prospective ledgers remain the forward confirmation vehicle.

## 8. Primary metrics

### 8.1 Structural-stress probability

- Brier score;
- log loss;
- PR-AUC;
- ROC-AUC;
- calibration slope/intercept;
- precision/recall and false-alert burden;
- family-specific deterioration recall.

### 8.2 Three-axis rating

- monotonic event/deterioration rates across ordered categories;
- peer-ranking concordance;
- own-history lead/lag behavior before known events;
- category stability across annual origins;
- category stability under missingness and source perturbations;
- relationship with data coverage and uncertainty;
- region/income concentration;
- production migration analysis.

### 8.3 Banking and sovereign event heads

- Brier score;
- log loss;
- PR-AUC;
- ROC-AUC;
- event recall;
- false alerts per 100 country-years;
- false alerts per true event;
- calibration by era, region and coverage group.

## 9. Advancement gates

A model does not advance merely because it is the best challenger.

### 9.1 Structural rating gate

Against the production-architecture reconstruction on identical outer-test rows, the challenger must:

1. improve aggregate Brier score by at least 5%;
2. improve aggregate log loss by at least 5%;
3. not reduce PR-AUC by more than 2%;
4. show non-decreasing deterioration/event rates across ordered categories, allowing at most one adjacent statistical tie;
5. not increase false alerts per true deterioration by more than 10% at the registered recall floor;
6. retain the result in both outer windows;
7. retain the conclusion across target sensitivities;
8. show materially lower dependence on data coverage than production.

### 9.2 Banking-crisis gate

The banking head must beat the historical event-rate baseline on Brier score and log loss, preserve or improve PR-AUC, meet the recall floor with usable false-alert burden and pass both later windows.

### 9.3 Sovereign-crisis gate

The sovereign head must:

1. use the pinned governed sovereign event ledger;
2. beat the event-rate baseline;
3. beat debt/GDP alone on Brier score and log loss;
4. preserve or improve PR-AUC;
5. meet the recall floor with usable false-alert burden;
6. pass both later windows.

### 9.4 Combined-event gate

The direct composite head must outperform the mandatory max-probability and independence-union sensitivities on calibrated probability metrics. Otherwise the system reports separate banking and sovereign probabilities without claiming a superior combined probability.

### 9.5 Operational gate

Before any promotion proposal:

- outputs reproduce from immutable inputs;
- a rollback bundle exists;
- populations reconcile;
- no provider projection entered baseline evidence;
- largest score migrations have analyst-readable explanations;
- all unavailable event heads remain null rather than zero;
- owner approval is explicit.

## 10. Possible conclusions

1. **Full replacement candidate:** structural rating, banking head and sovereign head pass.
2. **Rating replacement candidate with partial event coverage:** structural rating passes but one event head remains unavailable or fails; retain the corresponding production/analyst process.
3. **Structural-score replacement candidate:** score/category pass, event heads do not.
4. **Partial component replacement:** only selected components pass.
5. **No replacement:** challenger does not beat production.

## 11. Deliverables

- peer-relative and own-history component ledger;
- governed banking and sovereign event ledgers;
- structural-deterioration target and sensitivity audit;
- candidate probability ledgers;
- three-axis score/category ledger;
- production reconstruction and migration report;
- calibration/alert-burden/segment diagnostics;
- replacement gate decision;
- frozen challenger bundle only if a gate passes;
- explicit rejected/unavailable output ledger otherwise.

## 12. Execution policy

Develop and test locally/synthetically where possible. Do not use GitHub Actions as an iterative development loop. One consolidated real-data checkpoint is allowed only when closure-ready.

A fresh broad-source retrieval is not required for development. The official sovereign-default source acquisition is a separately logged data-governance step.

## 13. Production firewall

Phase 6 may not alter or deploy:

- production `app.py`;
- current country scores;
- production source caches;
- selected production classifier;
- production inference pipeline;
- served risk model.

No merge, promotion or deployment is authorized by this PRD.
