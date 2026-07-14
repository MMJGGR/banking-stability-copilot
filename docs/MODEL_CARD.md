# Banking Stability Risk Model Card

## Release Status

| Field | Current position |
|---|---|
| Serving snapshot | `2026-06-30`, generated `2026-07-12` |
| Countries scored | 201 |
| Manifest status | `verified` and actively served; no recorded transition to the governance lifecycle state `approved` |
| Source mode | `local_cached_sources` |
| Governance policy | Approved by `@MMJGGR` on 2026-07-10 |
| Crisis-validation status | `invalid_superseded`; legacy summary metrics and confusion matrix are withheld in the app |
| Archived checkpoints | `artifacts/snapshots/2025-12-31`, `artifacts/snapshots/2026-06-30`, `artifacts/snapshots/2026-06-30-official-api` |

`artifacts/data_manifest.json` is the authority for the artifact currently
served. It records a 2026-06-30 cutoff, a 2026-07-12 build, 201 scored
countries, zero failed model smoke checks, and locally cached IMF/World Bank
source inputs. The snapshot is serving successfully, but `verified` proves
artifact integrity and recorded checks; it is not evidence of owner approval
or external model validation.

The serving artifact persists its feature values and PCA loading maps. The
separate inference-pipeline artifact contains the fitted imputation values,
scaling, constrained components, direction rules, and reference distributions
needed to score a fixed snapshot consistently.

## Intended Use

The model is intended for:

- Cross-country banking-system risk screening.
- Monitoring changes in macroeconomic and banking-sector indicators.
- Prioritizing countries for deeper analyst review.
- Scenario exploration and transparent peer comparison.

It is not intended for:

- Automatic investment, lending, supervisory, or policy decisions.
- Institution-level solvency assessment.
- Precise crisis timing.
- Replacement of country experts or supervisory information.
- Causal claims about individual indicators.

## Current Serving Structure

The served score combines:

- A directionally constrained economic pillar.
- A directionally constrained banking/industry pillar.
- Confidence weighting, risk floors, and a bounded critical-field missingness
  penalty.
- A legacy supervised systemic-banking-crisis overlay.

The base pillar score gives equal weight to the economic and industry
components. The crisis component is not a simple 90/10 average. It can only
raise risk and is calculated as:

`max(0, 0.1 x ((1 + 9 x P(crisis)) - pillar score))`

The crisis probability and final risk score are different measures; neither
is a literal probability of bank failure.

## Crisis Labels and Validation Truth

The governed reference artifact is extracted from Appendix I, Table A1 of
Laeven and Valencia (2026), *Systemic Banking Crises Database: 1970-2025*, IMF
Working Paper WP/26/94. It contains exactly 164 published rows: 161 systemic
episodes and three episodes the authors explicitly classify as borderline.
The borderline Nicaragua 2018, Vietnam 2022, and Sri Lanka 2023 rows are
excluded from default positive targets.

The served classifier predates that exact reconciliation and was trained on
the earlier incomplete label implementation. Its packaged grouped/holdout
summary also selected and reported the operating threshold on the same
holdout. Consequently, its precision, recall, F1, alert count, and confusion
matrix are not clean external-test evidence. The app now fails closed and does
not show them as current validation results.

The repository contains a replacement research foundation with exact labels,
nested grouped and forward validation, cross-fitted calibration, frozen
inner-fold thresholds, and out-of-fold ledgers. Tested challengers failed the
pre-declared 2014-2018 forward holdout (approximately 0.37-0.56 ROC-AUC across
the tested specifications with an unusable alert burden). No replacement
classifier, hazard probability, or alert policy was promoted. Active country
scores remain unchanged by that research work.

## Interpretation Limitations

- Pillar scores are relative to the scored country universe.
- Countries with missing data can depend materially on imputation and policy
  floors.
- Some banking-sector features have less than 50% direct coverage.
- The legacy crisis overlay is not a validated stand-alone early-warning
  model and should not drive decisions.
- Relative percentile scores can move when the country universe or source
  coverage changes.
- The model has not completed formal external validation.

The current sensitivity evidence is
`artifacts/model_policy_audit.json`. It records the exact WP/26/94 label counts
and policy sensitivities. Directional constraints make score orientation
deterministic; GDP per capita remains an input whose inclusion sensitivity is
reported rather than an uncontrolled sign anchor.

## Validation and Promotion Standard

A future classifier or score release must include:

- Exact WP/26/94 label reconciliation and explicit borderline treatment.
- Country-grouped and expanding out-of-time evaluation.
- Preprocessing, calibration, and threshold selection fitted without outer
  test leakage.
- A final confirmation period untouched by model, feature, calibration, and
  threshold selection.
- ROC-AUC, PR-AUC, Brier score, calibration, precision, recall, specificity,
  false alerts per 100 country-years, event recall, and alert burden.
- Results by region, income group, crisis epoch, regime, and data coverage.
- Baseline comparison, confidence intervals, and sensitivity to imputation and
  country-universe changes.
- Challenger-versus-production score movement, named approval, checksums, and
  a rollback artifact.

Validation metrics may appear in the Methodology tab only when the packaged
summary explicitly carries `validation_status: validated_clean`,
`clean_validation: true`, and `display_metrics: true`. Historical README or
artifact metrics without that evidence gate are not approved results.
