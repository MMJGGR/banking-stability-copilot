# Phase 6 completion — replacement challenge closed as a negative result

Date: 28 September 2026
Branch: `research/phase6-closeout` (from `research/broad-feature-phase6-replacement-score-2026-09-25` at `9379dcd`)
Status: **closed; replacement not advanced**
Production status: unchanged; no merge, promotion or deployment

## Decision

The Phase 6 replacement challenger does not replace the production score and is not developed further. The blended replacement score is retired. The broad-state supervised crisis layer is retired. The evidence below is kept as the record.

What carries forward from the research programme is the Phase 3 transition engine, as a nowcast for countries with stale data, and the governed sovereign-default label work. Event modelling continues in the product plan with a small pre-specified predictor set, pooled estimation and events that occur in low-income economies (see the BankEnv Model Review, 28 Sep 2026).

## Crash fixed

The final CI run (`36123402026`) failed after validation had finished, in `fit_latest_probability`: `CalibratedClassifierCV(cv=forward_splits, ensemble=False)` calls `cross_val_predict`, which requires splits that partition every row, and forward time splits never test the earliest years.

The fix mirrors validation. The model is fitted on origins whose one-to-three-year labels resolved before the last forward block, and a sigmoid calibrator is fitted on that block (`cv="prefit"`). A regression test (`test_latest_probability_fits_with_forward_splits`) reproduces the failing condition. Validation code and metrics are unchanged by the fix.

## Result: the challenger loses to the base rate

Registered later-time windows 1995–2000, 2001–07, 2008–14 and 2015–21; 2,265 matched test rows.

| Target | Base-rate Brier | State | Observed | Combined (challenger) |
|---|---:|---:|---:|---:|
| Banking crisis | **0.0437** | 0.0635 | 0.0460 | 0.0647 |
| Sovereign default | 0.0752 | **0.0746** | 0.0859 | 0.1388 |
| Either event | 0.1052 | **0.1050** | 0.1181 | 0.1387 |

The combined challenger is worse than the base rate on every target and worse than both of its parts, which is overfitting and miscalibration: 274 features against 103–269 positive rows, under one event per candidate variable. The development gate fails on every check.

The blended score's weight search chose the minimum allowed weight for peer and own-history components (0.1 / 0.1 / 0.8) in all four folds: the data rejected the blend. The blend was also never calibrated after weighting, contrary to PRD v1.1 §6.

## Why state-only banking PR-AUC looked strong (0.36 vs 0.034 in Phase 4)

All of it comes from one window and one event cluster.

| State-only banking, test window | ROC-AUC | PR-AUC |
|---|---:|---:|
| 1995–2000 | 0.744 | 0.219 |
| 2001–2007 | 0.854 | 0.566 |
| 2008–2014 | 0.224 | 0.015 |
| 2015–2021 | 0.427 | 0.005 |

60 of the 103 positive rows are 2005–07 origins of the 2007–09 crisis. The model gives France, Spain, Germany, Portugal, Ireland, Italy, the Netherlands, Luxembourg, Belgium, Austria and Greece probabilities of about 0.999 at those origins: it recognises advanced European economies just before 2008. The fixed state was fitted on 1981–2026 data, so its geometry already reflects what happened after 2008.

Removing 2004–07 origins:

| Banking target | ROC-AUC all | ROC-AUC excl. 2004–07 | PR-AUC excl. 2004–07 | Prevalence excl. |
|---|---:|---:|---:|---:|
| State | 0.724 | 0.477 | 0.021 | 0.022 |
| Observed | 0.615 | 0.578 | 0.028 | 0.022 |
| Combined | 0.737 | 0.489 | 0.028 | 0.022 |

Without the cluster the state has no banking-crisis ranking skill, consistent with Phase 4. Phase 4's windows (2016–21) excluded the cluster, which is why it reported 0.034.

## One positive finding

The observed-indicator model ranks **sovereign default onsets** consistently: ROC-AUC 0.782 and PR-AUC 0.201 against a prevalence of 0.081, and 0.769 / 0.197 with 2004–07 removed. Its Brier score (0.0859) is worse than the base rate, so its probabilities need recalibration, but the ranking signal is real and not driven by the 2008 cluster. This supports building the product's event reading on sovereign distress and observed indicators.

## Limits

- The fixed Phase 2 state was fitted on the full 1981–2026 sample; all state-based results carry look-ahead.
- History is latest-vintage, not real-time.
- The banking and sovereign panels were inner-joined, reducing the banking population from about 6,980 to 3,165 rows and excluding country-years in sovereign default or its cooldown.
- No matched historical comparison with the locked production classifier is possible; its historical predictions were not preserved.

## Production firewall

No production path, cache, score or classifier changed. The locked classifier remains 52,580 bytes, SHA-256 `054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee`.
