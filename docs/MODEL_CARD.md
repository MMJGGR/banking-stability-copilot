# Banking Stability Risk Model Card

## Release Status

| Field | Value |
|---|---|
| Current committed artifact | February 2026 legacy model |
| Countries scored | 201 |
| Artifact status | `legacy_model_unverified_cutoff` |
| Approved snapshot cutoff | Not available in the legacy artifact |
| Production-validation status | Pending retraining under grouped and out-of-time validation |

The legacy artifact is retained for application continuity. It is not an
approved YE2025 or mid-2026 model release.

New candidate models persist a separate inference-pipeline artifact containing
training-time imputation values, scaling, PCA transforms and orientation, and
the reference distributions used for comparable percentile scoring. The crisis
classifier artifact likewise persists training-time fill values and its
calibrated estimator. These controls do not retroactively make the legacy model
comparable; a verified candidate rebuild is still required.

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

The legacy final score uses a 90% pillar component and a 10% crisis-probability
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
`artifacts/model_policy_audit.json`. On the existing 201-country feature matrix,
removing GDP from PCA inputs changes 14 countries by at least one risk point,
while removing GDP-based PCA orientation changes 140. Confidence regression
changes 23 countries by at least one point and risk floors change 11. These are
material policy sensitivities and require approval before model promotion.

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

- The current production artifact predates explicit snapshot metadata.
- Crisis-label completeness requires independent review.
- The complete preprocessing and PCA inference objects are not yet preserved
  in the legacy artifact.
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
