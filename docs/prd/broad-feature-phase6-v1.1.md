# PRD v1.1 — Phase 6: replacement country-risk score

Date: 2026-09-25  
Status: active replacement-challenge requirements  
Branch: `research/broad-feature-phase6-replacement-score-2026-09-25`  
Parent Phase 5 head: `f07ee156a0e91b1a2103bc7e858e825302a80304`  
Production changes: none authorized

## 1. Objective

Phase 6 returns to the original project objective: build a decision architecture that can replace the current production score **only if it wins an apples-to-apples challenge**.

The replacement must answer three questions for every sufficiently supported country:

1. **Peer position:** how risky is the country relative to other countries at the same date?
2. **Own-history position:** how risky is the country relative to its own prior history?
3. **Event imminence:** what is the calibrated probability of a systemic banking crisis, sovereign default/distress onset, or either event within one to three years?

The score is not another unsupervised pillar average. It is a supervised, time-ordered decision layer on top of the fixed broad state and crisis-specific observed indicators.

## 2. Fixed analytical backbone

Phase 6 does not refit Phases 2–5 merely to improve the replacement score.

Fixed inputs include:

- the 96-dimensional missing-aware state;
- exact one-year state velocity where available;
- state-information and uncertainty diagnostics;
- the accepted transition/probability architecture;
- governed historical banking-crisis labels;
- crisis-specific observed macro, fiscal, external and banking indicators;
- the frozen production score/classifier as benchmarks where matched historical predictions are available.

The broad state remains neutral. Event labels orient only the separate decision layer.

## 3. Governed event targets

### 3.1 Banking-crisis onset

Use the official Laeven–Valencia 1970–2025 systemic episode artifact already pinned in the repository.

Primary target:

`systemic banking-crisis onset in forecast-origin year +1 through +3`.

Exclude:

- active-crisis years;
- the registered three-year post-crisis cooldown;
- right-censored origins;
- the three officially borderline episodes from the primary target.

### 3.2 Sovereign-default onset

Use the 2025 Bank of Canada–Bank of England Sovereign Default Database, covering default stocks through 2024.

Primary default state includes positive government obligations in default to external official/private creditors or on local-currency debt. Domestic fiscal arrears are retained as a sensitivity and are not silently merged into the primary label.

Primary target:

`entry into sovereign default in forecast-origin year +1 through +3`.

The source workbook must be pinned by SHA-256 and converted into a country-year status/episode artifact with parsing diagnostics. If ingestion or identity checks fail, sovereign probabilities remain unavailable; they are never inferred from unsupported labels.

Current WEO 2026–2031 projections are prohibited from event targets, historical features, calibration outcomes and realizations.

### 3.3 Either-event target

Define a direct union target:

`banking-crisis onset OR sovereign-default onset within +1 to +3 years`.

Train this target directly. Do not manufacture the probability by assuming independence between banking and sovereign events.

Separate banking and sovereign probabilities remain visible.

## 4. Candidate decision models

Every candidate uses later-time validation and training-only preprocessing.

### A. State benchmark

Inputs:

- fixed 96-dimensional state;
- state velocity;
- state uncertainty and coverage.

This is retained as a benchmark even though Phase 4 showed that state-only banking-crisis prediction was inadequate.

### B. Crisis-specific observed benchmark

Inputs:

- governed as-of macro/fiscal/external indicators;
- banking capital, asset-quality, earnings, funding and liquidity indicators where observed;
- missingness/age flags;
- no future observations or provider projections.

This represents the core logic of a conventional early-warning model.

### C. Replacement challenger

Inputs:

- fixed broad state;
- state velocity and uncertainty;
- crisis-specific observed indicators;
- own-history deviations of the model signal;
- no production score as a predictor.

Primary estimator family:

- regularized logistic regression with calibrated probabilities;
- elastic-net/logistic challenger where sample size supports it;
- nonlinear models only if they provide stable incremental value in later windows.

Simple models remain preferred when performance is statistically indistinguishable.

## 5. Three score components

All components are scaled from 0 to 1, where 1 is highest risk.

### 5.1 Peer-risk percentile

For each forecast origin, rank the candidate’s pre-calibration risk signal across countries observed at that date.

This answers: `How risky is this country compared with peers now?`

### 5.2 Own-history stress percentile

For each country and date, compare the current risk signal only with that country’s earlier risk-signal history.

Use an expanding empirical distribution. Shrink toward 0.5 when history is short; do not allow future values into the historical percentile.

This answers: `How unusual or stressed is this country compared with itself?`

### 5.3 Absolute event imminence

Use the calibrated direct either-event probability.

Also report separately:

- banking-crisis probability;
- sovereign-default probability;
- either-event probability.

This answers: `How likely is a material banking/sovereign event within one to three years?`

## 6. Composite replacement score

The final index is a non-negative weighted combination of:

- peer-risk percentile;
- own-history percentile;
- absolute either-event probability.

Weights must sum to one and are selected inside time-ordered training data to minimise probability loss for the either-event target, subject to:

- no negative weight;
- minimum representation of each of the three owner-required dimensions unless later-window evidence demonstrates no incremental value;
- stability across folds;
- no use of final confirmation outcomes.

The weighted index is monotonically calibrated to the either-event target. The 1–10 score is:

`1 + 9 × calibrated replacement-risk index`.

Higher is riskier.

The familiar categories remain for comparability:

- 1–2: Very Low Risk;
- 3–4: Low Risk;
- 5–6: Moderate Risk;
- 7–8: High Risk;
- 9–10: Very High Risk.

Category thresholds may only change after a separately reported calibration/stability test.

## 7. Information quality

Missingness is not itself treated as economic distress.

For poorly supported countries:

- retain a model estimate if technically possible;
- report uncertainty and coverage separately;
- mark the rating `provisional` or `insufficient_information` under registered support rules;
- do not automatically worsen the economic score merely because information is missing.

This replaces the production model’s practice of imposing mechanical risk floors for low coverage.

## 8. Validation design

### 8.1 Time ordering

Use expanding historical outer windows. Every feature, transform, calibration map, score weight and threshold is fitted only on earlier observations.

### 8.2 Primary metrics

For banking, sovereign and either-event probabilities:

- Brier score;
- log loss;
- precision-recall AUC;
- ROC AUC;
- calibration by probability band;
- recall and false-alert burden at registered review thresholds.

For the 1–10 score:

- monotonic realized event rates by score/category;
- rank correlation with later adverse events;
- category migration/stability;
- performance by region, development group, coverage and era;
- matched comparison with production outputs where available.

### 8.3 Replacement gate

The challenger does not replace production unless it:

1. improves calibrated probability quality for the primary either-event target;
2. does not materially worsen banking-crisis performance;
3. produces monotonic and stable score/category event rates;
4. controls false-alert burden at the chosen recall level;
5. remains robust by coverage and era;
6. passes prospective shadow confirmation;
7. receives explicit owner approval.

A richer output set alone is not sufficient.

## 9. Piecemeal deliverables

### 6.1 — Outcome and source contract

- this PRD;
- production-score audit;
- banking/sovereign/either-event definitions;
- sovereign source provenance and fail-closed rules.

### 6.2 — Label and panel engineering

- sovereign workbook ingestion and pinned source hash;
- banking and sovereign onset ledgers;
- exact as-of feature panel;
- state/velocity/information joins;
- contamination and censoring audit.

### 6.3 — Candidate models

- state benchmark;
- crisis-specific observed benchmark;
- state + observed replacement challenger;
- calibrated banking, sovereign and either-event probabilities.

### 6.4 — Replacement score

- peer percentile;
- own-history percentile;
- absolute imminence;
- time-selected non-negative weights;
- 1–10 score and categories;
- information-quality status.

### 6.5 — Decision report

- later-window metrics and score monotonicity;
- direct production comparison where technically valid;
- go/no-go result;
- unchanged production-artifact verification;
- shadow-only serving output if the gate is not met.

## 10. Production firewall

Phase 6 may not modify, merge, promote or deploy:

- production `app.py`;
- current country scores/categories;
- the pillar pipeline;
- production source caches;
- the selected crisis classifier.

The production classifier remains locked at 52,580 bytes and SHA-256:

`054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee`

No production substitution occurs without the complete replacement gate and explicit authorization.
