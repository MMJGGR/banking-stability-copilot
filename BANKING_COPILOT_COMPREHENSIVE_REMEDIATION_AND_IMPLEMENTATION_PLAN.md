# Banking Stability Copilot: Comprehensive Remediation and Implementation Plan

## 1. Purpose

This document is the project-wide remediation, modernization, and implementation plan for the Banking System Stability Copilot. It covers all material issues identified during the repository, model, data, runtime, and hosting review—not only data refresh and snapshot functionality.

The target system must:

- Correct known runtime and model-training defects.
- Establish reproducible model validation and governance.
- Produce comparable quarterly risk snapshots and optional monthly monitoring views.
- Preserve historical source vintages and model versions.
- Distinguish actual, estimated, projected, carried-forward, and imputed values.
- Continue operating when an external API is unavailable or changes schema.
- Keep data ingestion and model training outside the Streamlit application.
- Remain practical within GitHub and Streamlit Community Cloud constraints.
- Have deterministic dependencies, automated tests, and CI controls.
- Replace stale or contradictory documentation with generated evidence.
- Resolve whether “copilot” means an analytical dashboard or an interactive AI product.
- Provide production-readiness, security, monitoring, rollback, and ownership controls.

### 1.1 Implementation Status

Completed in the initial priority tranche:

- [x] Fixed the missing classifier ROC AUC import and added a clear failure when no CV fold is valid.
- [x] Fixed the application model-load return contract.
- [x] Fixed the undefined correlation-diagnostic feature lists.
- [x] Corrected training-date propagation to the UI and model metadata.
- [x] Added IMF annual, quarterly, and monthly period parsing.
- [x] Added WEO actual, estimate, and projection status preservation for newly built caches.
- [x] Added an explicit default YE cutoff and `MODEL_AS_OF_DATE` override.
- [x] Added FSIBSIS quarterly and monthly support.
- [x] Replaced row-level validation with country-grouped validation.
- [x] Relabelled deployment-model epoch output as in-sample diagnostics.
- [x] Added derived FSIBSIS and WGI Parquet caches.
- [x] Added a lightweight Streamlit model-artifact loader to avoid importing the training stack.
- [x] Added serving-manifest generation with source cutoffs, artifact checksums, and explicit legacy cutoff status.
- [x] Added checksum validation before the default Streamlit model artifact is unpickled.
- [x] Added configurable source adapters with API, bulk-download, and validated local fallbacks.
- [x] Added a scheduled source-availability workflow that does not download Git LFS data.
- [x] Added Python 3.11 configuration, development requirements, automated tests, and a CI workflow.
- [x] Verified the Streamlit serving path and on-demand FSIBSIS loading in a browser.

Checkpoint evidence:

- Local validation: 22 tests passed; Python compile check and `git diff --check` passed.
- Browser validation: application rendered the snapshot status, initially deferred
  FSIBSIS loading, and loaded 83 United States balance-sheet indicators on demand
  without browser console errors.
- Current artifact status: the February 2026 model remains explicitly marked
  `legacy_model_unverified_cutoff`; it has not been relabelled as a verified
  YE2025 or mid-2026 release.

Remaining work is tracked in the delivery plan and immediate next actions below.

## 2. Comprehensive Issue Register

### 2.1 Immediate Code and Runtime Defects

| Issue | Impact | Priority |
|---|---|---:|
| `auc()` is used by classifier validation but is not imported | Fresh classifier training can fail during cross-validation | Critical |
| The model-training correlation plot references undefined `economic_cols` and `industry_cols` | Diagnostic output silently fails inside a broad exception handler | High |
| The application model-load failure path returns seven values while the caller expects six | A primary load failure can trigger a secondary unpacking failure | Critical |
| The UI reads the training date from `pca_info`, while the model stores it separately | Dashboard displays an unknown training date | Medium |
| Broad and bare exception handlers suppress actionable errors | Failures can be hidden and debugging becomes unreliable | High |
| Runtime paths assume execution from the repository root | Alternate execution contexts can fail to find README and image assets | Medium |
| Pickle artifacts are loaded directly | Artifacts are Python-version sensitive and unsafe if their provenance is not controlled | High |

### 2.2 Model Methodology and Validation Issues

| Issue | Impact | Priority |
|---|---|---:|
| Random row-level train/test and cross-validation splits can place the same country in training and validation epochs | Reported performance can be inflated by country leakage | Critical |
| Validation is not a genuine out-of-time backtest | Forward-looking performance is not adequately demonstrated | Critical |
| Non-crisis observations are deduplicated differently from crisis observations | Training sample construction can distort class and country representation | High |
| PCA scores are percentile ranks across the current country universe | Scores can change when countries enter or leave, even without underlying country changes | High |
| GDP per capita is used in PCA and to orient both pillars | Potential structural wealth bias requires explicit validation and policy approval | High |
| Quarterly comparability is impossible if PCA is refitted for every snapshot | Apparent score movements can reflect model changes rather than data changes | Critical |
| Current model artifacts do not preserve the complete transform pipeline | New observations cannot be reliably transformed on an identical basis | Critical |
| Classifier calibration and performance claims are not stored in an auditable model card | Published metrics cannot be reliably tied to the deployed model | High |
| Crisis labels and event coverage require independent review | Missing or disputed crisis events can bias training and evaluation | High |
| Confidence floors and imputation penalties are policy choices embedded in code | Their effects require sensitivity analysis and governance | Medium |
| Relative risk ranks are presented similarly to absolute probability measures | Users can misinterpret the score’s meaning | High |

### 2.3 Data Correctness and Freshness Issues

| Issue | Impact | Priority |
|---|---|---:|
| WEO extraction treats years through the current calendar year as historical | 2026 projections can enter a YE2025 model | Critical |
| Actual, estimate, and projection statuses are not preserved in the normalized cache | Users cannot distinguish observed from forecast values | Critical |
| IMF monthly and quarterly formats are not consistently normalized | Available observations can be excluded or misdated | Critical |
| FSIBSIS feature extraction uses annual columns and ignores available quarterly data | Banking balance-sheet features remain at 2024 despite 2025 quarterly observations | Critical |
| The model is a latest cross-section rather than a country-period feature store | Historical and quarterly snapshots are difficult to reproduce | High |
| `tot_deterioration_3yr` has no coverage | A declared model feature provides no information | High |
| Multiple banking features have less than 50% raw coverage | Results depend heavily on imputation and carry-forward assumptions | High |
| Source-specific stale observations are collapsed into a single overall coverage measure | Data age risk is understated | High |
| Raw source versions and transformation manifests are incomplete | Historical scores cannot be fully reconstructed | High |

### 2.4 Application and Hosting Issues

| Issue | Impact | Priority |
|---|---|---:|
| Streamlit reparses the large FSIBSIS CSV and WGI workbook during cold startup | Cold start is approximately 90–100 seconds locally | Critical |
| Model, data loading, and presentation are combined in one startup resource | Cache behavior and failure isolation are poor | High |
| Raw datasets are shipped with the application deployment | Deployment size, memory, startup time, and LFS bandwidth are unnecessarily high | High |
| Approximately 444 MB of Git LFS objects are downloaded for a complete checkout/deployment | Frequent deployments can exhaust included bandwidth | High |
| Streamlit Community Cloud is being treated as a possible processing environment | CPU, memory, hibernation, and ephemeral runtime behavior make this unsuitable | High |
| The application has no last-known-good artifact fallback | A bad refresh or missing artifact can make the application unavailable | Critical |
| No explicit application health, freshness, or degraded-mode status is presented | Operational problems are not visible to users | Medium |

### 2.5 Software Engineering and Delivery Issues

| Issue | Impact | Priority |
|---|---|---:|
| No automated test suite | Core transformations and scoring can regress undetected | Critical |
| No CI workflow | Broken code or artifacts can be merged and deployed | Critical |
| Dependencies use broad minimum versions without a lock file | Builds are not deterministic | High |
| `imbalanced-learn` is absent while SMOTE is requested by the classifier | Training silently changes behavior depending on environment | High |
| Current local environment contains dependency conflicts | Local success does not guarantee deployment reproducibility | High |
| No supported Python-version and artifact compatibility policy | Pickles and compiled dependencies can fail after upgrades | Medium |
| Diagnostic scripts are numerous and inconsistently structured | Maintenance and ownership are unclear | Medium |
| Replication outputs are duplicated in multiple folders | Repository size and source-of-truth ambiguity increase | Medium |
| Logging is primarily print-based and inconsistently structured | Automation cannot reliably classify warnings and failures | Medium |
| Configuration and policy thresholds are distributed across modules | Changes are difficult to audit and test | Medium |

### 2.6 Product and User-Experience Issues

| Issue | Impact | Priority |
|---|---|---:|
| The product is named a “copilot,” but the deployed application is an analytical dashboard | Product expectations and actual capabilities do not match | High |
| Insight, report, trend, and explainability modules are not consistently connected to the UI | Existing capabilities are incomplete or unused | Medium |
| No conversational workflow or LLM integration exists | “Copilot” functionality is not implemented | Product decision |
| No formal user personas or decision workflows are defined | Feature priorities and acceptable risk explanations are unclear | High |
| Risk score provenance and movement explanations are limited | Users cannot easily audit why a score changed | High |
| No access-control or data-classification decision is documented | Future private or licensed data could be exposed incorrectly | High |
| Accessibility, responsive behavior, and user testing are not documented | UI quality is not systematically validated | Medium |

### 2.7 Documentation and Governance Issues

| Issue | Impact | Priority |
|---|---|---:|
| README, replication documentation, training logs, and model artifacts report different metrics | Users cannot determine which claims are current | Critical |
| Country and feature counts are inconsistent across artifacts | Reproducibility claims are weakened | High |
| Model limitations and intended use are not clearly stated | Risk of inappropriate decision use | High |
| No model card, data card, change log, or release process exists | Governance and auditability are insufficient | High |
| No named owners or review requirements are encoded in repository controls | Updates and model promotions lack accountability | High |

## 3. Program Scope

### 3.1 In Scope

- All issues in the comprehensive issue register.
- Immediate runtime and training stabilization.
- Dependency and environment reproducibility.
- Automated tests and CI.
- Model-validation redesign.
- Model and data governance.
- Explicit product-definition decision for “copilot.”
- Dynamic data ingestion and historical snapshots.
- Streamlit performance and resilience.
- Artifact storage and release management.
- Documentation reconciliation.
- Security, monitoring, recovery, and ownership.

### 3.2 Out of Scope for Initial Remediation

- Institution-level confidential supervisory data.
- Intraday or real-time risk scores.
- A migration away from Streamlit before the optimized architecture is measured.
- Fully autonomous model promotion.
- Material financial decisions without qualified human review.
- Enterprise identity and authorization until user and data classifications require them.

## 4. Stabilization Workstream

### 4.1 Critical Bug Fixes

- Import and test `sklearn.metrics.auc`.
- Correct the application load failure return contract.
- Replace undefined correlation-plot variables with a shared feature contract.
- Correct training-date propagation to the UI.
- Replace bare exceptions with scoped exceptions and structured error messages.
- Resolve paths from `BASE_DIR` or `pathlib.Path(__file__)`.
- Fail training when mandatory diagnostics or artifacts cannot be produced.

### 4.2 Dependency Baseline

- Select and document a supported Python version.
- Add every optional dependency that is required for production behavior.
- Decide whether SMOTE remains part of the approved method.
- Pin direct dependencies.
- Generate a reproducible lock or constraints file.
- Build and test in a clean environment.
- Add an artifact compatibility version to every serialized model.

### 4.3 Minimum Test Harness

Add:

- Unit tests for period parsing and cutoff logic.
- Unit tests for score categories and confidence floors.
- Unit tests for source adapters and fallbacks.
- Contract tests for model artifact loading.
- Integration test for a small end-to-end training fixture.
- Streamlit startup smoke test.
- Regression fixture for selected countries and scores.

## 5. Model Validation and Governance

### 5.1 Validation Redesign

- Replace row-level random splitting with country-grouped validation.
- Add rolling-origin or epoch-based out-of-time backtests.
- Ensure a country does not appear in both training and validation for the same evaluation.
- Report ROC-AUC, PR-AUC, recall, precision, false-positive rate, calibration, and uncertainty.
- Report results by income group, region, data coverage, and crisis epoch.
- Benchmark against transparent baselines.
- Quantify score sensitivity to imputation, coverage floors, and GDP anchoring.
- Test stability when countries enter or leave the scoring universe.

### 5.2 Model Governance Artifacts

Every production model release requires:

- Model card.
- Data card.
- Training manifest.
- Feature contract.
- Validation report.
- Known limitations.
- Intended and prohibited uses.
- Bias and subgroup analysis.
- Challenger-versus-production comparison.
- Approval record.
- Rollback artifact.

### 5.3 Score Interpretation

The UI and documentation must distinguish:

- Relative percentile-based pillar scores.
- Calibrated crisis probabilities.
- Composite policy scores.
- Data confidence and freshness.

## 6. Product Definition

### 6.1 Required Product Decision

Choose one of two explicit product directions:

1. **Banking Stability Analytics Dashboard**
   Rename the product and focus on transparent risk analytics, comparisons, reports, and monitoring.

2. **Banking Stability Copilot**
   Retain the name and add a governed conversational layer that answers questions using model outputs, source metadata, methodology, and citations.

### 6.2 If Copilot Scope Is Approved

Implement only after the analytical foundation is stable:

- Curated question types and user personas.
- Retrieval from approved model/data artifacts only.
- Source citations and snapshot awareness.
- No unsupported causal or investment conclusions.
- Prompt-injection and data-exfiltration controls.
- Conversation logging policy.
- Cost and rate controls.
- Human-readable uncertainty and limitation statements.
- Evaluation set for factuality, grounding, and harmful overclaiming.

### 6.3 Existing Module Integration

Review and either integrate, rewrite, or remove:

- Insight generation.
- Report generation.
- Trend analysis.
- Explainability.
- Alternative risk-scorer modules.
- Unused dashboard components and diagnostic scripts.

## 7. Snapshot Definitions

### 7.1 Production Snapshot Cadence

| Snapshot | Cutoff | Intended status |
|---|---:|---|
| YE snapshot | 31 December | Final after agreed reporting lag |
| Q1 snapshot | 31 March | Final or near-final |
| H1/Q2 snapshot | 30 June | Provisional initially, then final |
| Q3 snapshot | 30 September | Provisional initially, then final |
| Monthly monitor | Month end | Provisional monitoring only |

### 7.2 Initial Snapshot Set

- `2025-YE`: reproducible baseline using information available for the 2025 cutoff.
- `2026-Q1`: first comparable quarterly refresh.
- `2026-H1-PROVISIONAL`: latest available information through 30 June 2026.
- `LATEST`: alias to the latest successfully published snapshot.

### 7.3 Observation Status

Every feature value must have one of the following statuses:

- `actual`
- `estimate`
- `projection`
- `carried_forward`
- `imputed`
- `missing`

The system must not silently treat estimates or projections as actual observations.

### 7.4 Snapshot Record

Each country-snapshot record should include:

```text
country_code
snapshot_date
snapshot_status
model_version
risk_score
economic_pillar
industry_pillar
crisis_probability
data_coverage
freshness_score
imputation_rate
oldest_feature_date
actual_feature_count
estimated_feature_count
projected_feature_count
source_vintages
generated_at
```

## 8. Target Architecture

```text
Official APIs / Bulk Downloads
              |
              v
      Source Adapters
              |
              v
 Raw Versioned Snapshot Storage
              |
              v
 Normalization and Period Parsing
              |
              v
 Data Quality and Freshness Gates
              |
              v
 Country-Period Feature Store
              |
              +----------------------+
              |                      |
              v                      v
 Fixed-Model Inference       Controlled Retraining
              |                      |
              +----------+-----------+
                         |
                         v
              Versioned Serving Artifacts
                         |
                         v
                    Streamlit UI
```

Streamlit remains the presentation and exploration layer. It must not become the scheduler, durable data store, raw-data processor, or model-training environment.

## 9. Source Strategy

### 9.1 Common Source Adapter Contract

Create source-specific adapters under:

```text
src/sources/
├── base.py
├── imf_weo.py
├── imf_fsic.py
├── imf_mfs.py
├── imf_fsibsis.py
└── world_bank_wgi.py
```

Each adapter will implement:

```python
check_version()
fetch_api()
fetch_bulk()
validate_schema()
normalize()
get_freshness()
build_manifest_entry()
```

### 9.2 Retrieval Fallback Order

1. Official API, when available and stable.
2. Official bulk CSV or Excel download.
3. Most recent validated raw snapshot.
4. Fail without publishing a replacement.

The last known-good production snapshot must remain available when a source fails.

### 9.3 Weak API Controls

- Explicit connection and read timeouts.
- Bounded retries with exponential backoff.
- Authentication through GitHub Secrets.
- Response-size and content-type checks.
- Schema-version detection.
- ETag, Last-Modified, release-version, or checksum comparison.
- Bulk-download fallback.
- Source-specific mappings isolated from model code.
- Raw response retention for audit and replay.
- Alerts when the fallback snapshot exceeds its freshness SLA.

### 9.4 Source Cadence

| Source | Version check | Processing cadence | Notes |
|---|---:|---:|---|
| FSIC | Weekly | Monthly | Quarterly/monthly availability varies by country |
| MFS | Weekly | Monthly | Preserve monthly and quarterly observations |
| FSIBSIS | Weekly | Quarterly | Use quarterly observations when annual data is unavailable |
| WEO | Weekly metadata check | April and October | January/July updates are partial |
| WGI | Monthly metadata check | Annual | Structural annual input |
| Crisis labels | Quarterly review | Annual/model release | Requires controlled governance |

## 10. Data Model

### 10.1 Normalized Observation Table

```text
source
dataset_version
country_code
indicator_code
period
frequency
value
observation_status
publication_date
retrieved_at
source_url
raw_snapshot_id
```

### 10.2 Country-Period Feature Table

```text
country_code
snapshot_date
feature_name
feature_value
source
source_period
source_vintage
observation_status
is_imputed
imputation_method
freshness_days
confidence
```

### 10.3 Manifest

Each published snapshot requires a JSON manifest containing:

- Snapshot identifier and cutoff date.
- Generation timestamp.
- Model and feature-schema versions.
- Source dataset versions.
- Source retrieval timestamps and URLs.
- Raw and processed checksums.
- Row, country, and indicator counts.
- Latest observation period by source.
- Actual, estimate, projection, and imputation counts.
- Validation results.
- Previous production snapshot.

## 11. Model Operating Model

### 11.1 Frequent Inference Refresh

Run monthly or quarterly:

1. Ingest new source observations.
2. Build a dated feature matrix.
3. Apply the approved imputer and transformations.
4. Apply the approved scalers and PCA models.
5. Generate classifier probabilities.
6. Produce risk scores.
7. Compare with the previous snapshot.
8. Publish only after validation.

### 11.2 Controlled Retraining

Run annually or manually:

1. Freeze a training cutoff.
2. Build a time-indexed training panel.
3. Fit imputers, scalers, PCA, and classifier.
4. Use country-grouped and out-of-time validation.
5. Compare the challenger model with production.
6. Review score and tier changes.
7. Approve and version the model.
8. Re-score historical snapshots only as a separately labelled restated series.

### 11.3 Comparability Rule

Quarterly snapshots must use the same approved model pipeline. PCA must not be refitted for each quarterly snapshot because that would change the score basis and reduce comparability.

Persist the full inference pipeline:

- Feature contract and ordering.
- Imputer.
- Transformations.
- Scalers.
- PCA models.
- Classifier.
- Calibration logic.
- Thresholds.
- Training cutoff and version metadata.

## 12. Storage and GitHub Strategy

### 12.1 Repository Contents

Keep in Git:

- Application and pipeline code.
- Configuration.
- Schemas and mappings.
- Tests.
- Small serving artifacts.
- Snapshot manifests.
- Data quality summaries.

Do not routinely commit:

- Full raw IMF CSV files.
- Full WGI workbooks.
- Frequently changing large Parquet files.
- Duplicate replication outputs.

### 12.2 Initial Artifact Storage

Use GitHub Releases for:

- Raw source snapshots.
- Processed historical datasets.
- Model release packages.
- Large diagnostic bundles.

Move to object storage if update frequency, access patterns, security, or retention requirements exceed what GitHub Releases can manage cleanly.

### 12.3 Serving Artifacts

Streamlit should load only compact artifacts:

```text
artifacts/
├── model_scores.parquet
├── snapshot_index.parquet
├── latest_features.parquet
├── explorer_timeseries.parquet
├── model_pipeline.pkl
└── data_manifest.json
```

The model artifact should ultimately use a safer and more portable serialization approach where practical. Until then, it must only be loaded from a controlled, trusted release.

## 13. GitHub Actions Workflows

### 13.1 Source Check

File: `.github/workflows/source-check.yml`

Schedule: weekly at a non-zero minute.

Responsibilities:

- Check source metadata and versions.
- Compare checksums or headers.
- Record detected changes.
- Open or update an issue when a source changes.
- Avoid downloading unchanged large files.

### 13.2 Data Refresh

File: `.github/workflows/refresh-data.yml`

Triggers:

- Monthly schedule.
- Manual dispatch.
- Confirmed source update.

Responsibilities:

- Retrieve changed sources.
- Validate source files.
- Normalize observations.
- Build features and snapshot artifacts.
- Score with the current production model.
- Generate comparison and quality reports.
- Upload raw and processed artifacts.
- Open an automated pull request.

### 13.3 Model Retraining

File: `.github/workflows/retrain-model.yml`

Triggers:

- Annual schedule.
- Manual dispatch.

Responsibilities:

- Build the training panel.
- Train a challenger model.
- Run grouped and out-of-time validation.
- Produce diagnostics and change analysis.
- Upload a versioned model candidate.
- Open a model-release pull request.

The workflow must not automatically promote the challenger model.

### 13.4 Quality Checks

File: `.github/workflows/quality-check.yml`

Runs on pull requests:

- Unit tests.
- Period parser tests.
- Schema-contract tests.
- Coverage and freshness gates.
- Snapshot reproducibility test.
- Model-score sanity tests.
- Streamlit startup smoke test.
- Artifact-size checks.

## 14. Data Quality and Publication Gates

Block publication when:

- Required columns or indicators disappear.
- Source schema changes without an approved mapping.
- Row, country, or indicator counts decline beyond tolerance.
- Observation periods move backwards.
- Duplicate country-period-indicator records are introduced.
- Values outside valid economic or banking ranges increase materially.
- Projections enter an actual-only snapshot.
- Imputation or carry-forward rates exceed thresholds.
- Feature coverage falls beyond tolerance.
- Risk score or tier changes exceed review thresholds.
- Artifact generation is incomplete.
- The new snapshot cannot be reproduced from its manifest.

The update report must include:

- Added and removed countries.
- Source revisions.
- Coverage changes.
- Feature-vintage changes.
- Actual, estimated, projected, carried-forward, and imputed counts.
- Largest country score movements.
- Risk-tier changes.
- Model and source versions.
- Failed or degraded source adapters.

## 15. Streamlit Changes

### 15.1 Runtime Responsibilities

Streamlit will:

- Load the approved model as a cached resource.
- Load compact Parquet data as cached data.
- Present country, regional, and global outputs.
- Compare snapshots.
- Display freshness, coverage, and confidence.
- Continue serving the last known-good snapshot after refresh failures.

Streamlit will not:

- Download full source datasets during startup.
- Parse large CSV or Excel files.
- Train models.
- Run scheduled update jobs.
- Persist authoritative data on local runtime storage.

### 15.2 Snapshot User Experience

Add:

- Snapshot selector.
- Final/provisional/restated badge.
- Model-version selector where historical model versions are retained.
- Current versus prior score delta.
- Risk-tier movement.
- Feature-level movement attribution.
- Data freshness and coverage panel.
- Source-vintage disclosure.
- Warning when a country relies heavily on stale, carried-forward, or imputed data.

### 15.3 Performance Target

- Cold start under 15 seconds under normal hosting conditions.
- No raw file larger than the compact serving artifacts loaded at startup.
- No repeated dataset parsing during user interactions.

## 16. Implementation Workstreams

### Workstream A: Snapshot Semantics and Period Handling

- Add `as_of_date`.
- Normalize annual, quarterly, and monthly periods.
- Add observation-status handling.
- Correct WEO actual/estimate/projection selection.
- Add FSIBSIS quarterly extraction.
- Create period and cutoff tests.

### Workstream B: Source Adapters

- Create the common adapter interface.
- Implement WEO and FSIBSIS first.
- Add FSIC, MFS, and WGI.
- Add API credentials and bulk fallbacks.
- Add version and schema detection.

### Workstream C: Feature Store and Model Pipeline

- Create country-period feature representation.
- Persist source period and status per feature.
- Separate inference from training.
- Persist the complete inference pipeline.
- Add historical snapshot scoring.

### Workstream D: Quality and Governance

- Define schema contracts.
- Define coverage and freshness SLAs.
- Add validation reports.
- Add score-change thresholds.
- Define model approval and rollback processes.

### Workstream E: Storage and Automation

- Move raw snapshots out of routine Git history.
- Configure GitHub Releases or object storage.
- Add scheduled workflows.
- Add automated update pull requests.
- Add retention and cleanup policies.

### Workstream F: Streamlit Refactor

- Remove raw-source parsing.
- Load serving artifacts only.
- Add snapshot and freshness controls.
- Add graceful degradation.
- Verify memory and cold-start performance.

### Workstream G: Immediate Stabilization

- Fix classifier validation, application loading, metadata, and path defects.
- Replace silent failures with explicit errors.
- Establish a clean, reproducible environment.
- Add the minimum automated test harness.

### Workstream H: Model Validation and Governance

- Rebuild validation around country-grouped and out-of-time testing.
- Review crisis labels and sample construction.
- Quantify structural bias and score sensitivity.
- Produce model cards, data cards, and release approval records.

### Workstream I: Product and Copilot Definition

- Decide whether to rename the product or implement a governed copilot layer.
- Define user personas, decisions, and acceptable outputs.
- Integrate or remove unused insight, reporting, trend, and explanation modules.
- Add grounded-answer evaluation if conversational functionality is approved.

### Workstream J: Documentation, Security, and Operations

- Reconcile README and replication claims with generated artifacts.
- Add change logs, runbooks, ownership, and release procedures.
- Define artifact trust, secrets, access, and data-classification controls.
- Add operational health, freshness, alerting, and rollback procedures.

## 17. Delivery Plan

### Sprint 0: Critical Stabilization

Deliverables:

- Classifier AUC fix.
- Application load-contract fix.
- Correlation diagnostic fix.
- Correct training metadata display.
- Scoped exception handling.
- Path normalization.

Exit criteria:

- Application and committed model load successfully.
- A clean classifier training smoke test reaches validation.
- Mandatory diagnostics fail loudly rather than being silently skipped.

### Sprint 1: Engineering Baseline

Deliverables:

- Supported Python version.
- Pinned dependencies and reproducible lock/constraints file.
- Complete production dependency list.
- Unit, integration, artifact-load, and startup tests.
- Pull-request CI workflow.
- Structured logging baseline.

Exit criteria:

- A clean environment can install and run the test suite.
- CI blocks syntax, test, dependency, and artifact-contract failures.

### Sprint 2: Data Correctness Foundation

Deliverables:

- Explicit snapshot cutoff.
- Observation-status model.
- Period parser repairs.
- FSIBSIS quarterly support.
- WEO projection controls.
- Unit tests.
- Reproducible `2025-YE` feature matrix.

Exit criteria:

- No post-cutoff observation enters the snapshot.
- Every value has a source period and status.
- Monthly and quarterly parser tests pass.

### Sprint 3: Source Resilience

Deliverables:

- Source adapter framework.
- WEO and FSIBSIS adapters.
- API/bulk/last-good fallback.
- Snapshot manifest.
- Raw snapshot retention.

Exit criteria:

- A simulated API failure successfully uses the bulk fallback.
- A simulated full source failure preserves the previous production snapshot.

### Sprint 4: Full Ingestion Coverage

Deliverables:

- FSIC, MFS, and WGI adapters.
- Normalized observation table.
- Schema and freshness controls.
- Country-period feature table.

Exit criteria:

- All five sources produce normalized output.
- Source version, cutoff, and coverage are recorded automatically.

### Sprint 5: Comparable Scoring

Deliverables:

- Persisted inference pipeline.
- Inference-only refresh command.
- `2026-Q1` and provisional `2026-H1` snapshots.
- Snapshot comparison report.

Exit criteria:

- Snapshots use the same model version.
- Score changes can be traced to feature changes.
- Repeated runs from identical inputs produce identical outputs.

### Sprint 6: Automation and Hosting

Deliverables:

- GitHub Actions workflows.
- Release-based artifact storage.
- Automated update pull requests.
- Streamlit serving-only refactor.
- Snapshot selector and freshness panel.

Exit criteria:

- A scheduled workflow can produce a reviewable update PR.
- Streamlit starts without parsing raw IMF or WGI files.
- Cold-start and memory targets are met.

### Sprint 7: Model Validation and Governance

Deliverables:

- Time-indexed training panel.
- Grouped and out-of-time validation.
- Challenger-versus-production report.
- Model approval and rollback procedure.
- Annual retraining workflow.

Exit criteria:

- Model promotion requires explicit approval.
- Historical scores remain tied to their original model version.
- Restated histories are separately labelled.

### Sprint 8: Product Completion and Documentation

Deliverables:

- Approved dashboard-versus-copilot product decision.
- Integrated or retired auxiliary modules.
- Snapshot-aware explanation and provenance experience.
- Reconciled README and replication package.
- Model card and data card.
- Operations runbook and release notes.
- Accessibility and user-acceptance review.

Exit criteria:

- The product name matches delivered functionality.
- Published metrics are generated from the deployed artifacts.
- Users can understand score meaning, provenance, freshness, and limitations.
- Operations and rollback can be performed from documented procedures.

## 18. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| Data pipeline owner | Source adapters, normalization, manifests, data quality |
| Model owner | Features, inference pipeline, training, validation |
| Application owner | Streamlit serving layer and snapshot UX |
| Reviewer | Data changes, score changes, model promotion |
| Repository administrator | Actions, secrets, Releases, access and retention |

For a small team, one person may hold multiple roles, but production publication and model promotion should still have an explicit review step.

## 19. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| IMF API authentication or availability changes | Bulk-download adapter and last-good snapshot |
| Source schema changes | Contract tests and quarantine rather than automatic publication |
| Incomplete current-quarter reporting | Provisional status, carry-forward flag and confidence penalty |
| Model scores become incomparable | Freeze model pipeline between controlled model releases |
| Git LFS bandwidth and storage growth | Store raw snapshots in Releases/object storage |
| Streamlit resource limits | Load only compact serving artifacts |
| Scheduled workflow delay | Manual dispatch and non-zero cron minute |
| Silent historical revisions | Preserve raw vintages and generate revision reports |
| Excessive imputation | Coverage gates and country-level confidence warnings |
| Automated update produces implausible movements | Score-change thresholds and reviewable PR |
| Runtime or training defects regress | Required CI tests and clean-environment smoke tests |
| Model validation remains optimistic | Country-grouped and out-of-time evaluation gates |
| Product overclaims “copilot” behavior | Explicit product decision and grounded evaluation |
| Documentation diverges from artifacts | Generate reported metrics from release manifests |
| Untrusted model artifact is loaded | Controlled artifact provenance and checksum validation |

## 20. Success Measures

- No future-period observation enters a historical snapshot.
- Every score is linked to a model version and data manifest.
- Every feature is linked to its source period and observation status.
- Quarterly snapshots are comparable under a fixed model.
- Routine source refreshes require no manual data preparation.
- Failed source refreshes do not break the live application.
- Streamlit does not parse full raw source files.
- Cold-start time is below the agreed target.
- Snapshot generation is reproducible.
- Material data or score changes are reviewed before publication.
- Fresh training and validation complete without suppressed exceptions.
- CI passes in a clean, supported environment.
- Published performance metrics are reproducible from the production release.
- Validation prevents country and temporal leakage.
- Product naming and functionality are aligned.
- Model, data, application, and operational limitations are documented.

## 21. Immediate Next Actions

Work is executed in this priority order. A checked item has been implemented and
locally verified; it does not imply that a candidate model has been approved for
production.

1. [x] Fix the critical classifier and application defects.
2. [x] Establish a supported Python 3.11 dependency baseline.
3. [x] Add the minimum automated test suite and pull-request CI.
4. [ ] Approve snapshot definitions and provisional/final rules.
5. [x] Implement period parsing and actual/estimate/projection controls.
6. [ ] Rebuild `2025-YE` as the verified baseline. Blocked until official source
   endpoints or approved source files are configured.
7. [x] Redesign validation around country-grouped and out-of-time tests.
8. [ ] Audit crisis labels, GDP anchoring, confidence floors, and associated
   sensitivity. Full country-epoch sample preservation is complete. The
   training-relevant recent label window now uses the May 2026 IMF update
   through 2025 and excludes borderline cases by default; full historical
   reconciliation and policy approval remain outstanding. A reproducible audit
   now quantifies GDP orientation/input, confidence regression, and risk-floor
   sensitivity in `artifacts/model_policy_audit.json`.
9. [x] Persist the complete fitted inference pipeline. New candidate artifacts
   preserve classifier fill values and calibration plus pillar imputation,
   scaling, PCA orientation, and fixed reference distributions. The legacy
   committed model predates this contract and still requires a verified rebuild.
10. [x] Implement WEO, FSIBSIS, FSIC, MFS, and WGI source adapters.
11. [ ] Define and approve freshness, coverage, imputation, and score-change
    thresholds.
12. [ ] Produce `2026-Q1` and provisional `2026-H1` snapshots. Blocked by items
    4, 6, 9, and 11.
13. [ ] Move raw datasets out of routine Git LFS versioning. Candidate workflow
    artifacts and `data/raw/` exclusion are implemented; migration of existing
    repository history is a separate controlled operation.
14. [ ] Complete scheduled refresh, retraining, quality, promotion, and release
    workflows. Quality, source-check, and manual candidate-refresh workflows are
    implemented; promotion and release automation remain outstanding.
15. [ ] Complete the Streamlit serving-only refactor. Lightweight model loading,
    derived caches, manifest display, and on-demand FSIBSIS loading are complete;
    snapshot selection, health/freshness UI, and last-known-good fallback remain.
16. [x] Reconcile current documentation warnings and add model/data cards and an
    operations runbook. Release-specific generated metrics remain tied to a
    future approved candidate.
17. [ ] Decide whether to rename the dashboard or complete the governed copilot
    capability. Product-owner decision required.
18. [ ] Complete monitoring, rollback, ownership, and release procedures. The
    initial operations runbook is present; named ownership and automated health
    controls remain outstanding.

### 21.1 GitHub Checkpoints

- [x] Checkpoint 1: priority stabilization, data correctness foundation,
  grouped validation, source-adapter framework, candidate workflow, serving
  manifest, documentation baseline, and tests.
  - Published branch: `agent/priority-remediation`
  - Initial checkpoint commit: `197ce58`
  - Draft pull request:
    [#1](https://github.com/MMJGGR/banking-stability-copilot/pull/1)
  - Verification: 22 tests passed, compile and diff checks passed, and the
    Streamlit browser smoke test completed without console errors.
- [x] Checkpoint 2: deterministic inference transforms and calibrated classifier
  persistence.
  - Published commit: `4c4d7ca`
  - Pull request: draft
    [#1](https://github.com/MMJGGR/banking-stability-copilot/pull/1)
  - Verification: 26 tests passed; the persisted pillar pipeline is
    batch-invariant and exactly reproduces legacy pillar scores on the current
    201-country engineered feature matrix.
  - Open methodology gate: absolute score correlation with data coverage is
    `0.51`, above the current `0.40` threshold. The legacy and persisted
    implementations produce the identical failure, so this remains an explicit
    confidence-policy audit item rather than being waived.
- [x] Checkpoint 3: May 2026 IMF crisis-label provenance and recent-window
  corrections, including explicit borderline-case handling.
  - Published commit: `6bdfbdb`
  - Pull request: draft
    [#1](https://github.com/MMJGGR/banking-stability-copilot/pull/1)
  - Verification: 28 tests passed. Unsupported recent currency/sovereign-stress
    labels were removed, official recent systemic episodes were added, and
    source-designated borderline cases are excluded from training by default.
- [ ] Checkpoint 4: reproducible model-policy sensitivity audit covering GDP,
  confidence regression, risk floors, coverage correlation, and crisis-label
  composition. Local verification complete with 29 tests; mark checked after
  publication to draft pull request #1.
