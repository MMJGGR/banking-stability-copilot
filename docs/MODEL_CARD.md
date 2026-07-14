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

## Hierarchical Early-Warning Challenger

The repository also contains a research architecture that does **not** replace
the active score unless it passes forward-time validation and is explicitly
approved. It separates outputs that the active blended score can otherwise
conflate:

- A one-year conditional systemic-crisis onset hazard.
- Incremental onset risk in years two to three and cumulative three-year risk.
- A historical-core expert for long, broad macro-financial histories and a
  modern-full expert when direct banking-system evidence meets its coverage
  contract.
- Seven mechanism-evidence families: credit/property cycle; bank solvency and
  asset quality; bank funding and liquidity; sovereign liquidity and market
  access; external/FX stress; macro, commodity and global triggers; and
  structural resilience.
- Evidence coverage reported separately from risk. Missing data cannot be
  interpreted as low risk or statistical certainty.
- A proposed recall-oriented Amber review tier and a higher-conviction Red tier
  that requires stronger hazard plus corroborating or persistent evidence. Both
  tiers remain disabled until validation identifies usable operating points.

Mechanism outputs are normalized evidence composites, not seven independent
crisis probabilities. Conditional severity is deliberately not active: it
requires a separately observed and governed loss/severity target rather than a
proxy inferred from crisis onset.

The Streamlit app reads only a compact, build-time JSON snapshot. It does not
train models or retrieve BIS/IMF/World Bank data at runtime. Research status is
shown explicitly in Country, Global, Explorer, and Methodology views.

The corrected strict forward test uses horizon-specific label-availability
embargoes: the training origins end in 2007, 2006 and 2005 for the one-, two-
and three-year hazards; threshold validation begins in 2009; and the untouched
forecast-origin test is 2014-2022. It did not pass promotion. One-year ROC-AUC
is 0.367 and average precision is 0.0056; years-two-to-three ROC-AUC is 0.246
and average precision is 0.0038; cumulative three-year ROC-AUC is 0.291 and
average precision is 0.0076. The test contains 1,635 country-years, 196
countries and nine unique one-year crisis events. Validation found no usable
Amber or distinct Red operating point, so both tiers are disabled. Because the
promotion gate failed, the app suppresses country hazard probabilities and
issues no alerts. These results are retained as evidence rather than tuned
away.

This checkpoint evaluates a predeclared transparent estimator; it does not use
cross-validation for model or hyperparameter selection. A bounded candidate
set must be compared with expanding outcome-year folds and forward-only
calibration before the locked confirmation period is used again. The existing
2014-2022 results have already informed development and therefore are not a
pristine final confirmation sample for a future selected model.

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
- Embargoed expanding outcome-year cross-validation followed by an untouched
  forward confirmation period.
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
- The hierarchical challenger must remain research-only if its untouched
  forward test, calibration, specificity, precision, unique-event recall, or
  alert burden fails the documented gate.
- The historical-core and modern-full experts are not yet cross-calibrated;
  expert routing must not be treated as model certainty.

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
