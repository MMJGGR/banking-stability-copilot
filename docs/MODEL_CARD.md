# Banking Stability Risk Model Card

## Release Status

| Field | Value |
|---|---|
| Current committed artifact | Official API-sourced snapshot dated `2026-06-30` |
| Countries scored | 201 |
| Artifact status | `verified` manifest status; policy approval still pending |
| Active snapshot cutoff | `2026-06-30` |
| Archived checkpoints | `artifacts/snapshots/2025-12-31`, `artifacts/snapshots/2026-06-30`, `artifacts/snapshots/2026-06-30-official-api` |
| Production-validation status | Local model checks passed; final promotion requires governance approval |

The active artifact was rebuilt from official IMF SDMX and World Bank API
retrievals on 2026-07-10. It is a serving-ready candidate, not a formal
production approval. The mid-2026 snapshot uses WEO dataflow
`IMF.RES:WEO(9.0.0)`, FSIC `IMF.STA:FSIC(13.0.1)`, MFS
`IMF.STA:MFS_DC(8.0.0)`, FSIBSIS `IMF.STA:FSIBSIS(18.0.0)`, and WGI through
2024. The active manifest records official retrieval checksums and source
versions.

New candidate models persist a separate inference-pipeline artifact containing
training-time imputation values, scaling, PCA transforms and orientation, and
the reference distributions used for comparable percentile scoring. The crisis
classifier artifact likewise persists training-time fill values and its
calibrated estimator.

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

## Model Structure

The current architecture combines:

- Economic pillar PCA.
- Industry pillar PCA.
- A supervised systemic-banking-crisis classifier.
- Data-coverage and confidence adjustments.

The supervised target uses the May 2026 Laeven-Valencia systemic banking crisis
database through 2025. Events explicitly classified as borderline by the source
are excluded from primary training and reserved for sensitivity analysis.

The current final score uses a 90% pillar component and a 10% crisis-probability
component. Scores are bounded from 1 to 10.

## Interpretation Limitations

- Pillar scores are relative to the scored country universe.
- A score is not a literal probability of bank failure.
- The crisis probability and composite score are different measures.
- Countries with missing data can depend materially on imputation.
- Confidence floors are policy rules rather than learned model parameters.
- GDP per capita affects PCA construction and direction and requires bias
  sensitivity review.

The current machine-readable sensitivity audit is
`artifacts/model_policy_audit.json`. GDP orientation remains a material policy
choice because it determines whether higher GDP per capita anchors lower risk
or whether the PCA sign is allowed to drift. This must remain explicitly
reviewed before production promotion.

## Validation Standard

Future approved releases must include:

- Country-grouped validation.
- Out-of-time or rolling-origin evaluation.
- ROC-AUC and PR-AUC.
- Recall, precision, false-positive rate, and calibration.
- Results by region, income group, crisis epoch, and data coverage.
- Baseline comparisons.
- Leakage checks.
- Sensitivity to imputation, confidence floors, and country-universe changes.

Historical README metrics are not approved until reproduced under this
standard.

## Known Open Risks

- Crisis-label completeness requires independent review.
- Relative percentile scoring can change when the country universe changes.
- Some banking-sector features have less than 50% direct coverage.
- The model has not completed a formal external validation.

## Release Requirements

An approved release requires:

1. Versioned source and feature manifests.
2. A fixed snapshot cutoff.
3. Complete preprocessing-pipeline serialization.
4. Grouped and out-of-time validation.
5. Challenger-versus-production comparison.
6. Material score-movement review.
7. Artifact checksums.
8. Named approval and rollback artifact.
