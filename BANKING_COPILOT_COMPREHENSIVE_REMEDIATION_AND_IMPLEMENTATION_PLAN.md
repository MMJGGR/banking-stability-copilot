# BankEnv: Comprehensive Remediation and Implementation Plan

## 1. Purpose

This document is the project-wide remediation, modernization, and implementation plan for BankEnv. It covers all material issues identified during the repository, model, data, runtime, and hosting review—not only data refresh and snapshot functionality.

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
- [x] Split serving and training requirements so Streamlit Community Cloud
  (which ignores `runtime.txt` and builds with Python 3.13/3.14) installs
  binary wheels only; training keeps exact pins in `requirements-dev.txt`
  (2026-07-10).
- [x] Fixed the official-refresh regression that blanked all country display
  names: names now derive from country codes at SDMX normalization, at
  training, and defensively at app load (`src/country_names.py`) (2026-07-10).
- [x] Data Explorer now surfaces history at native source periodicity
  (monthly/quarterly/annual selector) and marks the boundary between reported
  actuals and IMF estimates/projections on WEO charts (2026-07-10).
- [x] Mapped TLS/WBG/KOS/SXM/CUW to real continents so small territories no
  longer surface as an "Other" highest-risk region (2026-07-10).
- [x] Built the directionally constrained challenger model: declared
  credit-risk direction per pillar feature, constrained non-negative PCA
  loadings shrunk toward equal weights, upward-only crisis overlay, and
  critical-field missingness penalty. Archived as
  `artifacts/snapshots/2026-06-30-challenger-directional` with a full
  production-vs-challenger comparison report; NOT promoted (re-ranking
  exceeds every review threshold) (2026-07-10, session 3).
- [x] Added last-known-good artifact fallback, a user-visible System Health
  panel (serving mode, snapshot age, per-source freshness vs SLA), the
  per-country score driver table script, crisis-label provenance
  verification harness, governance policy draft, CODEOWNERS, and release
  checklist (2026-07-10, session 3).

Checkpoint evidence:

- Local validation: 31 tests passed; Python compile check passed.
- Browser validation: application rendered the snapshot status, initially deferred
  FSIBSIS loading, and loaded 83 United States balance-sheet indicators on demand
  without browser console errors.
- Current artifact status: the active local serving artifact is verified for
  snapshot cutoff `2026-06-30`; `2025-12-31` and `2026-06-30` checkpoint
  bundles are archived under `artifacts/snapshots/`.
- Current data position: the active mid-2026 cutoff snapshot is sourced from
  official IMF SDMX and World Bank API retrievals. The active manifest records
  WEO through 2025-12-31, FSIC through 2026-04-30, MFS through 2026-05-31,
  FSIBSIS through 2026-M04, and WGI through 2024-12-31.

Current open issues are tracked only in section 21.5, **Current Open Issues
Backlog**. Earlier issue-register tables are retained as an audit trail of
identified findings that have either been fixed, superseded, or explicitly
carried forward into section 21.5.

## 2. Historical Issue Register (Cleared, Superseded, or Carried Forward)

This section is not the current open backlog. It records the original findings
that drove the remediation work. Items that remain unresolved after the latest
checkpoints are repeated in section 21.5 with current priority and next action.
Items not repeated in section 21.5 should be treated as cleared, superseded, or
historical context.

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
| No conversational workflow or LLM integration exists | Earlier “Copilot” positioning implied functionality that is not implemented | Product decision |
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

### 2.8 Findings from the July 2026 Independent Code Review

An independent full-repository review (2026-07-09) verified the checked items in
this plan against the code and identified the following additional issues. They
are registered here and folded into the immediate next actions in section 21.

| Issue | Impact | Priority |
|---|---|---:|
| Crisis-label epoch windows (1990/1995/2000/2005/2015 with a 3-year horizon) never produce positive labels for crises confined to 2009–2015 (Cyprus, Slovenia, Ukraine 2014, Bulgaria) or starting 2019+ (Lebanon 2019–2023) | Part of the May 2026 label update is structurally invisible to training; the small positive class is smaller than it needs to be | Critical |
| Cutoff and observation-status filtering is enforced only in `extract_weo_features`; `compute_literature_gap_features`, `compute_credit_to_gdp_gap`, `compute_sovereign_bank_nexus`, and `compute_external_risk_metrics` ignore `observation_status`, and the literature-gap and deployment lag features use `datetime.now()` instead of the snapshot cutoff | WEO estimates/projections dated within the cutoff can enter roughly half the engineered feature set | Critical |
| The `credit_to_gdp_gap` feature is a cross-country median deviation, not the BIS one-sided HP-filter trend gap the README and docstrings claim | The flagship "literature-validated" feature is misdescribed; documentation overstates methodology | High |
| Hybrid-weight documentation mismatch: module docstring and unused `HybridRiskScorer` state 0.4/0.4/0.2 while the implemented blend is 0.9 pillar / 0.1 classifier | Reviewers and users are misled about model structure; the unused class invites accidental reuse | High |
| Silent data heuristics are untracked: countries dropped when credit/GDP falls outside 5–500% (unit-mismatch guess), `sovereign_exposure_ratio` values below 2% overwritten with `securities_to_assets`, `replace(0, 1.0)` denominators clipped to ±100, FX exposure hard-imputed to 0 for a hand-picked currency list | Country scores can move or vanish without any flag, log record, or sensitivity coverage | High |
| Crisis labels are a hand-transcribed dictionary citing IMF WP/26/94; the transcription cannot be verified from the repository | Label provenance is unauditable; transcription errors are undetectable | High |
| Path normalization incomplete: `app.py` still opens `README.md` and `cache/eda/...` via relative paths | Alternate execution contexts break the Methodology tab | Medium |
| No dependency lock file and no interpreter guard; the suite fails collection on Python 3.8 with an unrelated `TypeError` | Wrong-environment failures are confusing; builds are not deterministic | High |
| Approximately 1,900 lines of dead modules (`risk_scorer`, `insight_generator`, `report_generator`, `trend_analyzer`, `explainability`, `ui_components`, `src/styles`) plus duplicated `replication/outputs` trees and 24 ad-hoc diagnostic scripts | Two competing risk-scoring and styling implementations confuse review and maintenance | Medium |
| The Methodology tab renders the README, including the unapproved historical AUC table, directly into the product UI | End users see disclaimed metrics presented as current documentation | Medium |
| Data Explorer time series do not display WEO actual/estimate/projection status | Research users cannot distinguish observed from forecast values in the browsing surface | Medium |

The review also confirmed strengths worth preserving: the persisted pillar
inference pipeline with fixed reference distributions, grouped and out-of-time
validation with explicit leakage assertions, checksum-validated serving
artifacts, the source-adapter fallback chain, and the candour of the model and
data cards. It further confirmed that the silent feature heuristics above do
not contaminate the raw Data Explorer views, which serve unmodified melted
observations.

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

**Owner clarification (2026-07-09):** the product serves two purposes and both
are in scope:

1. **Predictive screening** — the hybrid risk model producing cross-country
   risk scores and crisis probabilities.
2. **Research data utility** — the Data Explorer, cached IMF/WGI panels, and
   replication package used directly for the owner's research.

Readiness is therefore assessed per purpose. The research-utility surface is
usable now (raw explorer views are unmodified by model heuristics; serving
artifacts are checksum-validated). The predictive surface remains a candidate,
not production, until the section 2.8 critical items and the validation
standard in the model card are satisfied.

**Naming decision update (2026-07-11, owner):** the external app name is
**BankEnv**. The name reflects the banking operating-environment focus and
avoids similarity to rating-agency product names. The current release remains
an analytics app, not a conversational agent; adding a conversational layer
would require separate governance.

### 6.2 If Conversational Scope Is Approved

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

### 9.5 Official Retrieval Pipeline (SDMX 3.0) — Design Added 2026-07-09

Verified facts (probed 2026-07-09):

- The IMF Data portal exposes a public SDMX 3.0 REST API at
  `https://api.imf.org/external/sdmx/3.0` with no API key required.
- Data URL pattern:
  `/data/dataflow/{agency}/{dataflow}/{version}/{key}` with
  `Accept: application/vnd.sdmx.data+csv` returning CSV. `+` selects the
  latest dataflow version.
- The repository's existing exports map exactly to official dataflows:
  `IMF.RES:WEO(9.0.0)`, `IMF.STA:FSIC(13.0.1)`, `IMF.STA:MFS_DC(8.0.0)`,
  `IMF.STA:FSIBSIS(18.0.0)`. The returned `STRUCTURE_ID` column carries the
  dataflow version, which satisfies the manifest's source-version requirement
  automatically.
- Probe example (returns data):
  `https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.RES/WEO/+/USA.NGDP_RPCH.A`
- Vintage caveat: as of 2026-07-09 the live WEO dataflow still resolves to
  version 9.0.0 (October 2025 vintage, country update 2025-09-30, horizon
  2031). FSIC, MFS, and FSIBSIS are updated continuously and are expected to
  carry 2026 observations. The pipeline must therefore record and compare
  dataflow versions rather than assume freshness.
- The API returns long-format observations (one row per
  country-indicator-period), which maps directly onto the normalized
  observation table in section 10.1 — cleaner than reproducing the wide
  portal-export shape and melting it.
- WGI: retrieve from the World Bank API
  (`https://api.worldbank.org/v2/country/all/indicator/{code}`; percentile-rank
  codes such as `VA.PER.RNK` correspond to the 0–100 governance scores used by
  the model) or fall back to the published `wgidataset` workbook. Exact
  indicator codes to be confirmed during implementation.

Efficiency and cadence decisions (2026-07-09, revised same day after live
API testing):

- **API-first transport (revised after full workflow test).** Live testing
  reversed the earlier bulk-first assumption on two facts. First,
  `data.imf.org` returns 403 to every non-browser client, so portal bulk
  downloads cannot be automated at all — they remain a manual/local path
  only. Second, the current SDMX 3.0 client reaches the official IMF API and
  returns long-format CSV for every required IMF source, but the
  `c[TIME_PERIOD]=ge:` filter was not respected in the full workflow test.
  WEO, FSIC, MFS, and FSIBSIS probes requested recent slices but sampled
  historical periods as old as 1980, 2005, 2001, and 2005 respectively.
  Automated refreshes must therefore fix period filtering or use explicit
  source/country/indicator chunking before this path is trusted for scheduled
  app refreshes. Wildcard syntax is `*` per dimension (SDMX 3.0), not empty
  segments.
- **Discovered key structures (action 29 complete).**
  - `IMF.RES/WEO`: `COUNTRY.INDICATOR.FREQUENCY`
  - `IMF.STA/FSIC` (DSD_FSIC 13.0.0): `COUNTRY.SECTOR.INDICATOR.FREQUENCY`
  - `IMF.STA/MFS_DC` (DSD_MFS_DCS 8.0.0):
    `COUNTRY.INDICATOR.TYPE_OF_TRANSFORMATION.FREQUENCY`
  - `IMF.STA/FSIBSIS` (DSD_FSIBSIS 18.0.0):
    `COUNTRY.SECTOR.INDICATOR.FREQUENCY`
- **Upstream freshness confirmed (2026-07-09).** FSIC carries 2025-Q4,
  2026-Q1, and monthly observations through April 2026; MFS_DC through May
  2026; FSIBSIS through April 2026. The local January 2026 caches lag by
  roughly two to three quarters of banking data. WEO still resolves to the
  October 2025 vintage (9.0.0) upstream.
- **Change detection is poll-and-diff.** The API does not advertise an update
  schedule in machine-readable form. Available signals: dataflow version in
  `STRUCTURE_ID` (bumps on new vintages), WEO's per-country
  `COUNTRY_UPDATE_DATE` attribute, and self-computed content checksums. The
  weekly source check polls these and flags changes; the section 9.4 cadence
  table (WEO April/October with January/July partials; FSIC, FSIBSIS, and MFS
  rolling roughly monthly; WGI annual) remains the authoritative expectation
  for when changes should appear.
- **Refresh triggers.** Current state: `refresh-data.yml` is manual-dispatch
  only (candidate artifacts uploaded for review, never auto-committed);
  `source-check.yml` runs weekly but no-ops while the source URL secrets are
  empty. Target state: weekly source check detects a version or checksum
  change and opens an issue; a monthly scheduled refresh (plus manual
  dispatch) builds a candidate snapshot; publication remains a reviewed,
  manual step per section 14.

Implementation phases:

1. **Key discovery.** For each dataflow, fetch the data structure definition
   from `/structure/dataflow/{agency}/{dataflow}/+` and record the dimension
   order and codelists (the FSIC key structure differs from WEO's
   `COUNTRY.INDICATOR.FREQUENCY`; a naive filter probe returned 404).
   Persist the discovered key template per source in configuration, not code.
2. **SDMX fetch mode in the adapter.** Extend `SourceAdapter` with a
   `fetch_sdmx()` method: chunked requests (by country block or indicator
   block for MFS-scale flows), explicit timeouts, bounded retries with
   backoff, `ETag`/`Last-Modified` comparison, and raw CSV retention under
   `data/raw/{source}/{retrieved_date}/` for audit and replay. Retrieval
   order becomes SDMX API → configured bulk URL → validated local fallback,
   preserving the existing chain.
3. **Long-format normalization.** Add a loader path that consumes the SDMX
   long CSV directly into the section 10.1 observation table (source,
   dataset_version from `STRUCTURE_ID`, country, indicator, period,
   frequency, value, observation_status, retrieved_at), bypassing the
   wide-to-long melt. Map WEO estimate/projection boundaries from the
   dataflow's estimates-start attribute so observation status is populated at
   ingestion rather than inferred later.
4. **Version-check integration.** Point the existing `source-check.yml`
   workflow at the structure endpoints to detect dataflow version bumps
   (e.g., WEO 9.0.0 → next vintage) and open an issue when a source changes,
   without downloading data.
5. **Refresh integration.** Wire `refresh_data.py` to the SDMX fetch mode so
   `python -m src.scripts.refresh_data --as-of <cutoff>` performs genuine
   retrieval; record per-source dataflow version, retrieval timestamp, and
   checksums in the snapshot manifest. The quality gates in section 14 apply
   unchanged before any candidate is published.
6. **Equivalence test.** Before switching the model to API-sourced data,
   reproduce the current cached feature matrix from an API pull restricted to
   the same vintage where possible, and quantify any differences as a
   reviewed data-revision report rather than a silent swap.

### 9.6 SDMX Workflow Test — 2026-07-09

Test artifact: `artifacts/sdmx_workflow_test_report.json`.

Scope: live `check_version()` calls and live fetch probes for WEO, FSIC, MFS,
FSIBSIS, and WGI into a temporary directory. Raw temporary downloads were
removed after the test; the repository retains only the compact JSON report.
After the workflow probe, the local repository test suite passed:
`python -m pytest -q` → 31 passed, 1 warning.

| Source | Version/freshness result | Fetch probe result | App-ready status |
|---|---|---|---|
| WEO | `IMF.RES:WEO(9.0.0)` | 36.7 MB CSV; required long columns present; sample includes 1980–2031 despite requesting 2025+ | Not app-ready; period filtering must be fixed and estimate/projection status must be mapped |
| FSIC | `IMF.STA:FSIC(13.0.1)` | 145.6 MB CSV; required long columns present; sample includes 2005–2026-Q1 despite requesting 2025+ | Not app-ready; current fetch is too broad for scheduled refresh |
| MFS | `IMF.STA:MFS_DC(8.0.0)` | 478.0 MB CSV; required long columns present; sample includes 2001–2026-Q1 despite requesting 2026+ | Not app-ready; must be chunked and filtered before workflow use |
| FSIBSIS | `IMF.STA:FSIBSIS(18.0.0)` | 137.0 MB CSV; required long columns present; sample includes 2005–2026-Q1 despite requesting 2026+ | Not app-ready; must be chunked and filtered before workflow use |
| WGI | World Bank API returned 2024 data, but `check_version()` incorrectly reports latest year as 2017 | 81 KB CSV; 1,281 rows; 207 countries; six governance indicators; 2024-only filter respected | Partially app-ready; freshness detection bug must be fixed |

Workflow integration check:

- `src/scripts/refresh_data.py` exists but does not reference SDMX or
  `build_sdmx_sources()`.
- `src/scripts/check_sources.py` exists but does not reference SDMX or
  `build_sdmx_sources()`.
- `.github/workflows/refresh-data.yml` exists but does not call the SDMX
  client.
- `.github/workflows/source-check.yml` exists but does not call the SDMX
  version/structure checks.

Conclusion: the raw retrieval clients are a useful foundation, but the full
data workflow does **not** yet retrieve and rebuild the complete app dataset.
The workflow currently passes official-source connectivity and raw-format
tests, and fails end-to-end automation, period filtering, normalization, and
GitHub workflow integration.

### 9.7 Official API End-to-End Refresh — 2026-07-10

Status: completed locally and active in the serving artifacts.

Command path tested:

```bash
python -m src.scripts.refresh_data --as-of 2026-06-30
python -m src.scripts.refresh_data --as-of 2026-06-30 \
  --download-dir data/raw/official_refresh_20260710_001908 \
  --reuse-downloads
```

The first command downloaded all official source files. The initial
normalization run exposed a WEO period-parsing defect (`TIME_PERIOD` was read
as `1999.0`, causing all WEO rows to be dropped). The parser was fixed and the
second command reused the retained raw downloads, rebuilt caches, ran the model
pipeline, wrote the manifest, and passed validation.

Official retrievals used for the active snapshot:

| Source | Version/freshness | Raw bytes | Normalized rows | Coverage endpoint |
|---|---:|---:|---:|---|
| WEO | `IMF.RES:WEO(9.0.0)` | 36,738,338 | 361,733 | 208 countries; 1980–2031 horizon |
| FSIC | `IMF.STA:FSIC(13.0.1)` | 145,626,208 | 1,258,107 | 155 countries; through 2026-04 |
| MFS | `IMF.STA:MFS_DC(8.0.0)` | 478,046,242 | 4,509,993 | 180 countries; through 2026-05 |
| FSIBSIS | `IMF.STA:FSIBSIS(18.0.0)` | 136,980,774 | 11,459 | 141 countries; through 2026-M04 |
| WGI | World Bank API, latest year 2024 | 2,060,170 | 5,301 | 207 countries; 1996–2024 |

Model/app output:

- Active manifest: `snapshot_id=2026-06-30`,
  `snapshot_status=verified`, `source_mode=official_api_sdmx_worldbank`.
- Model artifact: `model.cutoff_verified=true`, `countries_trained=201`.
- Feature matrix: 214 countries, 73 columns.
- Model validation: 3 passed, 0 failed.
- Test suite: `python -m pytest -q` → 34 passed, 1 warning.
- Local Streamlit smoke: restarted on port 8514 and returned HTTP 200.
- Archived bundle: `artifacts/snapshots/2026-06-30-official-api`.

Remaining caveat: this updates the local repo and branch artifacts. The public
Streamlit app updates only after the committed changes are pushed to the branch
that Streamlit Cloud deploys, or after PR merge if the deployment tracks
`main`.

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

### Workstream I: Product Definition

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

## 21. Delivery Action Tracker and Current Backlog

This section records work execution history and the current backlog. The
checkbox lists below are delivery history; section 21.5 is the authoritative
current open-issues list. A checked item has been implemented and locally
verified; it does not imply that a candidate model has been approved for
production.

1. [x] Fix the critical classifier and application defects.
2. [x] Establish a supported Python 3.11 dependency baseline.
3. [x] Add the minimum automated test suite and pull-request CI.
4. [ ] Approve snapshot definitions and provisional/final rules.
5. [x] Implement period parsing and actual/estimate/projection controls.
6. [x] Rebuild `2025-YE` as the verified local cached-source baseline.
   Archived bundle: `artifacts/snapshots/2025-12-31`. Official endpoint-based
   reproduction remains pending source configuration.
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
   scaling, PCA orientation, and fixed reference distributions. The active
   `2026-06-30` artifact now includes the sidecar inference pipeline and a
   verified manifest.
10. [x] Implement WEO, FSIBSIS, FSIC, MFS, and WGI source adapters.
11. [ ] Define and approve freshness, coverage, imputation, and score-change
    thresholds.
12. [x] Produce a provisional `2026-H1`/mid-2026 local cached-source snapshot.
    Archived bundle: `artifacts/snapshots/2026-06-30`; active Streamlit
    manifest now displays `Snapshot 2026-06-30 | verified`. `2026-Q1` remains
    optional if a separate quarter-end checkpoint is required.
13. [ ] Move raw datasets out of routine Git LFS versioning. Candidate workflow
    artifacts and `data/raw/` exclusion are implemented; migration of existing
    repository history is a separate controlled operation.
14. [~] Promotion automation added 2026-07-10: `promote-snapshot.yml`
    downloads a candidate bundle, re-runs the serving-artifact smoke tests
    (`src/scripts/smoke_test_artifacts.py`, also gating `refresh-data.yml`),
    commits artifacts via Git LFS, and opens a reviewed promotion pull
    request. Final merge remains a human decision; release tagging remains
    outstanding.
15. [ ] Complete the Streamlit serving-only refactor. Lightweight model loading,
    derived caches, manifest display, on-demand country-sliced WEO/FSIC/MFS
    history, on-demand FSIBSIS loading, theme-safe controls, customizable
    peer sets, and cross-country indicator comparison are complete; active
    verified snapshot display is confirmed in browser; last-known-good
    fallback completed 2026-07-10 (session 3); health/freshness diagnostics
    and snapshot selection are hidden from the default frontend and gated
    behind `SHOW_ADMIN_DIAGNOSTICS=true`.
16. [x] Reconcile current documentation warnings and add model/data cards and an
    operations runbook. Release-specific generated metrics remain tied to a
    future approved candidate.
17. [x] Updated 2026-07-11 (owner): external app name standardized as
    **BankEnv** (page title, favicon, README, and working plan). Rejected
    names and vendor-adjacent naming patterns should not be reintroduced.
18. [~] Monitoring, rollback, ownership, and release procedures substantially
    complete 2026-07-10 (session 3): automatic last-known-good fallback,
    admin diagnostics, executable rollback runbook, CODEOWNERS, release
    checklist, and the proposed governance thresholds document. Outstanding:
    owner approval of `docs/GOVERNANCE.md` and post-deploy live health
    verification (backlog ranks 25-26, 30-31).

### 21.1a Refresh-Workflow Test Results (2026-07-09)

An end-to-end test of the refresh workflow produced these findings:

- **Cloud trigger blocked.** `refresh-data.yml` and `source-check.yml` exist
  only on `agent/priority-remediation`; GitHub returns 404 on dispatch until
  they reach the default branch. The refresh workflow is untriggerable until
  PR #1 merges.
- **No repository secrets configured.** All ten source-URL secrets are empty,
  so the weekly source check is currently a no-op.
- **Cloud fallback would fail anyway.** The workflow checks out with
  `lfs: false`, so the local-fallback path would find LFS pointer stubs, not
  data. Additionally, `data.imf.org` returns 403 to all non-browser clients
  (verified), so portal URLs cannot be fetched from Actions; only
  `api.imf.org` accepts plain clients. This confirms the section 9.5
  bulk-vs-API split must treat portal downloads as a manual/local path.
- **Full retrain path defect (fixed).** `refresh_data.py` with full
  retraining aborted: `external_debt_gdp` mapped to WEO `D_NGDPD`, which has
  aggregate-only coverage (G20, G40, G50, G60, G90 — zero real countries) in
  the current vintage. Committed snapshots never hit this because
  `build_local_snapshot.py` defaults to `retrain_classifier=False`, meaning
  the full-retrain path had likely never been executed end to end. Fix
  applied 2026-07-09: the `D_NGDPD` mapping was removed from both WEO
  extraction paths; the cached classifier remains loadable via its persisted
  fill values.
- **Pickle portability risk demonstrated.** Loading the cached classifier in
  a scikit-learn 1.8 environment raises `InconsistentVersionWarning`
  (artifact trained under 1.5.2) and an XGBoost serialization warning,
  confirming the issue-register entry on Python/library-version-sensitive
  pickles.
- **Positive side effect.** Rebuilding caches from raw CSVs populated the
  previously missing `observation_status` column in the WEO cache
  (actual/estimate/projection now distinguishable; action 35).
- Local rerun after the fix is in progress; results to be recorded here.

### 21.2 Actions Added by the July 2026 Independent Review

Ordered by leverage; items 19–21 block any production approval of the
predictive surface. See section 2.8 for full detail.

19. [x] Done 2026-07-10 (exceeded scope): coverage analysis showed 34 labeled
    episodes invisible to the original five epochs (1988-90, 1994-95,
    1999-2000, 2009-15, 2019+). The panel now has 12 epochs (added 1987,
    1993, 1998, 2008, 2011, 2014, 2018) covering every labeled start year
    from 1988 onward. Grouped and out-of-time validation re-run; honest
    leakage-free metrics (grouped CV AUC ~0.57-0.59) recorded for governance
    review — materially below the historical 0.84 claim.
20. [~] Partially done 2026-07-10 (session 3): automated reconciliation
    harness and checksum registry added
    (`src/scripts/verify_crisis_labels.py`, `data/reference/README.md`,
    `tests/test_verify_crisis_labels.py`). The parser, checksum pinning, and
    dictionary reconciliation are implemented and unit-tested; once the
    published WP/26/94 workbook is manually downloaded into
    `data/reference/` (IMF web hosts return 403 to all non-browser clients,
    the same block documented in section 9.5), every test run re-verifies the
    transcription against the pinned file. The one-step manual download is
    the only remaining action.
21. [x] Done 2026-07-10: shared `_enforce_cutoff` helper applied in
    `extract_fsic_features`, `compute_literature_gap_features`,
    `compute_credit_to_gdp_relative`, `compute_sovereign_bank_nexus`, and
    `compute_external_risk_metrics`; `datetime.now()` removed from the
    literature-gap and deployment-lag paths (cutoff threaded from
    `train_crisis_model(as_of_date=...)`).
22. [x] Done 2026-07-10 (rename branch): feature renamed to
    `credit_to_gdp_relative` across the pipeline with legacy aliases for
    cached classifiers and pre-rename serving artifacts; README, docstrings,
    and prints now state the cross-country median-deviation computation and
    explicitly disclaim the BIS HP-filter gap (implementing the genuine BIS
    gap remains a registered follow-up).
23. [x] Done 2026-07-10: module docstring corrected to 0.9 pillar / 0.1
    classifier, unused `HybridRiskScorer` removed.
24. [x] Done 2026-07-10: per-country heuristic flags collected during
    feature engineering and persisted to `artifacts/feature_heuristics.json`
    (shipped with candidate bundles). The first flagged build immediately
    exposed a critical defect: the millions/billions unit assumption dropped
    credit features for ~175 countries under official SDMX data (see 21.4).
25. [x] Done 2026-07-10: `constraints-dev.txt` generated (uv, Python 3.11)
    and used by the CI install; `src/config.py` now fails fast on
    interpreters older than 3.10.
26. [~] Partially done 2026-07-10: the seven dead modules are removed
    (verified unimported). `replication/outputs` deduplication and diagnostic
    script triage remain outstanding.
27. [x] Done 2026-07-10: README and architecture-diagram loads resolve from
    the app directory; the unapproved historical AUC block in the README is
    fenced with UNAPPROVED-METRICS markers; the Methodology tab now renders
    current manifest-backed model/data-card content instead of the stale README
    narrative.
28. [x] Display WEO actual/estimate/projection status in the Data Explorer
    time series to support research use. Implemented 2026-07-10: charts plot
    at native periodicity and mark the last reported actual with a dashed
    boundary and caption.

### 21.3 Retrieval Pipeline Actions (Section 9.5, Added 2026-07-09)

29. [x] Discover and persist the SDMX key structure (dimension order and
    codelists) for WEO, FSIC, MFS_DC, and FSIBSIS from the structure
    endpoints. Completed 2026-07-09; results recorded in section 9.5.
    Codelist retrieval remains part of action 30 implementation.
30. [x] Implement `fetch_sdmx()` as the primary automated transport
    (API-first per revised section 9.5): time-filtered incremental pulls,
    period-chunked backfills, retries, and raw-response retention; retrieval
    order becomes SDMX API → validated local fallback, with portal bulk
    downloads as a manual-only path (data.imf.org blocks non-browser
    clients). Completed 2026-07-10 for direct official API retrieval; period
    filtering is enforced client-side because IMF ignored tested server-side
    period parameters.
31. [x] Add the long-format SDMX loader path that populates the normalized
    observation table directly, including observation status at ingestion.
    Completed 2026-07-10 via `src/sources/sdmx_normalize.py`.
31a. [x] Add the monthly scheduled trigger to `refresh-data.yml` (keeping
    manual dispatch and reviewed publication) and make the weekly source
    check diff dataflow versions and checksums so refreshes are triggered by
    detected change, not blind schedule alone. Completed 2026-07-10; monthly
    scheduled candidate refresh now resolves a dynamic cutoff and the weekly
    source check uses official SDMX/WGI freshness checks.
32. [x] Confirm World Bank WGI retrieval codes (percentile-rank series) and
    add the WGI adapter's API mode. Completed 2026-07-10; freshness detection
    now reports latest year 2024.
33. [x] Point `source-check.yml` at the SDMX structure endpoints for
    version-bump detection. Completed through `src/scripts/check_sources.py`;
    workflow command defaults to official mode.
34. [x] Wire `refresh_data.py` to SDMX retrieval and record dataflow
    versions in the snapshot manifest; run the first genuine API-sourced
    refresh and compare against the January 2026 cached baseline as a
    reviewed revision report. Completed 2026-07-10; see section 9.7.
35. [x] Rebuild the WEO cache with observation status populated (fixes the
    committed cache's missing `observation_status` column noted in section
    2.8). Completed 2026-07-10.
36. [x] Fix the SDMX period filter/chunking defect identified in the
    2026-07-09 workflow test before enabling scheduled data refreshes.
    Completed 2026-07-10 by enforcing client-side period filtering and adding
    raw-download reuse for downstream reruns.

### 21.4 Fixes Landed 2026-07-10 (Session 2) and New Findings

Owner-reported symptom: Tanzania showed heavy imputation for industry data
despite excellent FSIC source coverage. Diagnosis traced four distinct
workflow gaps, all fixed:

1. **Stale imputed sidecar.** `imputed_features.parquet` was only written by
   the legacy pillar path; every build since the pipeline switch shipped the
   same stale file, so the dashboard flagged year-old numbers as "imputed"
   even where fresh raw data existed. The current pipeline now regenerates
   the sidecar on every build (`PillarInferencePipeline.impute`), and the
   smoke test asserts the sidecar preserves every observed raw value.
2. **Official-name pattern breaks.** FSIC extraction regexes written for the
   legacy CSV names missed the official SDMX names ('short-term' hyphen,
   'Foreign-currency-denominated loans'), silently losing
   `liquid_assets_st_liab` and `fx_loan_exposure`. Patterns now match both
   vocabularies; Tanzania regains observed values for both.
3. **Orphaned metadata after redundancy drops.** The correlation-based
   feature dropper removed value columns (roe, tier1_capital) but left their
   `_year` columns, and the dashboard treated any missing model column as
   imputable from the (stale) sidecar. Year companions are now dropped with
   their features and the dashboard only falls back to the sidecar for
   features the current model actually carries.
4. **Credit features nuked by a unit assumption (critical, new).** The
   heuristic flags added under item 24 revealed that the legacy
   millions/billions unit conversion produced ~0.0% credit/GDP ratios under
   official SDMX raw-unit data, so the 5-500% sanity gate silently dropped
   credit features for ~175 countries — every official-API artifact shipped
   without `credit_to_gdp_relative`, a Tier-1 classifier feature. The
   computation now tries both unit conventions per country and records a
   flagged scale adjustment when the non-default one is used.

Additional automation: candidate builds and promotions are now gated by
`src/scripts/smoke_test_artifacts.py` (manifest verified, >=150 countries,
scores in range, display names present, raw/imputed coherence), and
`promote-snapshot.yml` turns a candidate run into a reviewed promotion pull
request.

Governance note: the retrained 12-epoch classifier reports honest grouped-CV
AUC around 0.57-0.59 (out-of-time holdout 0.56-0.80 depending on fold),
replacing the leakage-inflated historical 0.84. Promotion of retrained
artifacts requires owner review under items 4, 8, and 11.

### 21.5 Current Open Issues Backlog (Updated 2026-07-10)

This section is the current working backlog. Older issue-register items above
remain useful history, but this table reflects what is still open after the
latest `master` checkpoints through `improve methodology cards`. If an older
unchecked action or historical issue is not listed here, do not treat it as a
current open issue without re-triage.

#### 21.5.1 Sprint 1: Model/Data-Science Trust and External-Liquidity Coverage

These items should be treated as one sprint because they share the same root
problem: the current model does not yet have enough directionally constrained
credit-risk logic or external-liquidity data to support rating-agency-style
interpretation. The expected sprint outcome is a defensible challenger model
and dataset, not just one-off feature additions.

| Rank | Open issue | Why it matters | Priority | Sprint deliverable |
|---:|---|---|---:|---|
| 1 | Mozambique/Kenya ranking exposed weak inference in the active artifact | Kenya is scored riskier than Mozambique because the current industry PCA and crisis blend dominate; the result is explainable but not yet analytically satisfying | Critical | Reproduce both country scores, publish driver table, and use the case as an acceptance test for challenger scoring |
| 2 | Crisis overlay can de-risk high pillar-risk countries | The 90/10 blend with a 1-10 crisis-probability component can lower a high base risk score when the crisis probability score is below the base pillar score | Critical | Redesign classifier overlay so crisis probability is additive, monotonic, or displayed separately |
| 3 | Unsupervised PCA can learn economically weak signs | Some banking features can affect the safety score in counterintuitive directions because PCA captures covariance, not causal credit risk | Critical | Replace or constrain PCA for credit-risk ratios using directional transforms/monotonic scoring |
| 4 | Imputation can make sparse countries look better than observed peers | Missing critical fields such as detailed banking concentration or real-estate exposure can be KNN-imputed favorably | High | Add critical-field missingness penalties and show raw/imputed driver flags |
| 5 | Existing source stack excludes BOP, IIP, IRFCL, PIP/CPIS, DIP/CDIS, Fiscal Monitor, QGFS/GFS, and market-price/rate feeds | These sources are needed for external-liquidity and market-access features | Critical | Add staged official-source adapters and normalized caches for the minimum viable external-liquidity block |
| 6 | Debt-service burden is not fully modeled | Fitch/Moody's/S&P-style debt affordability depends on interest and principal burden, not debt stock alone | Critical | Add debt-service ratios from IMF BOP, World Bank IDS/QEDS, Fiscal Monitor, and/or GFS where coverage permits |
| 7 | Gross external financing needs are missing | Core S&P-style external liquidity risk requires current account payments, short-term external debt, non-resident deposits, and maturing long-term external debt | Critical | Compute an approximation from BOP + IIP/external debt + reserves and flag assumption quality |
| 8 | Current account receipts/payments are missing | They are denominators for external liquidity and debt-service ratios | Critical | Add IMF BOP dataflow and compute CXR/CXP measures |
| 9 | International reserves adequacy is only proxied | Current `m2_to_reserves` and `sovereign_liability_to_reserves` do not fully capture usable reserves against imports/CXP/short-term debt | High | Add IMF IRFCL and reserves/imports, reserves/CXP, reserves/short-term external debt |
| 10 | Portfolio flows and portfolio liabilities are missing | Sudden-stop and market-access risk are central to external-finance assessment | High | Add IMF BOP portfolio flows and PIP/CPIS portfolio positions |
| 11 | Net IIP and external liabilities are missing | External solvency requires stock positions, not only current account flow | High | Add IMF IIP and compute net IIP/GDP, external liabilities/GDP, portfolio liabilities/GDP |
| 12 | Sovereign foreign-currency debt share is missing or weakly covered | FX debt structure is a core vulnerability metric | High | Test QEDS, World Bank IDS, GFS, and debt-management-source coverage |
| 13 | Gross financing needs and interest/revenue are not robustly modeled | Fiscal refinancing pressure and affordability are central to sovereign-credit analysis | High | Add IMF Fiscal Monitor, GFS/QGFS, and WEO fiscal indicators where available |
| 14 | Market access indicators are missing | Bond yields, spreads, CDS, issuance frequency, and failed auctions are high-value capital-market risk signals | High | Decide public proxy versus licensed market source before implementation |
| 15 | Current model is still a relative snapshot model | Scores are percentile-based and can move when source coverage or country universe changes | High | Add stable reference distributions by approved release and show rank-vs-score movement separately |
| 16 | GDP orientation and wealth anchoring remain a policy choice | GDP per capita affects pillar orientation but the active model is not a clean GDP-anchored sovereign-credit model | High | Decide whether GDP per capita is an anchor, peer stratifier, or ordinary feature |
| 17 | Honest grouped-CV AUC is modest | The retrained classifier is materially weaker than historical README claims and should not be oversold | High | Treat classifier as a weak signal until out-of-time and challenger comparisons support promotion |
| 18 | No external benchmark/challenger against rating-agency style outcomes | Current objective is banking-system risk, but users are also asking sovereign-credit/liquidity questions | High | Compare challenger score against ratings, spreads, or crisis outcomes |
| 19 | FDI flow stability is missing | Stable financing should be separated from hot-money financing | Medium | Add IMF BOP FDI flows and DIP/CDIS positions after the core BOP/IIP block |
| 20 | Commodity/export concentration and terms-of-trade exposure are missing | Narrow export bases can create external liquidity shocks | Medium | Add BOP/export composition where available; consider commodity-price exposure proxy |
| 21 | Real effective exchange-rate, equity-price, and property-price stress are missing | Fitch MPI-style macroprudential indicators use asset-price/REER pressure with credit growth | Medium | Explore IMF/BIS/OECD/national coverage after the core sprint |

#### 21.5.2 Sprint 2: App Reliability and User-Facing Trust

These app fixes should follow Sprint 1 because a reliable app serving weak
inference is not sufficient. They remain high priority because they determine
whether users can safely consume the improved model and dataset.

| Rank | Open issue | Why it matters | Priority | Sprint deliverable |
|---:|---|---|---:|---|
| 22 | Country-level score explanation remains limited | Suspicious outputs require manual artifact inspection | High | Add model-driver table with raw value, imputed value, source period, direction, and peer percentile |
| 23 | Last-known-good rollback process is not implemented end to end | A bad artifact or broken deployment can still take down the public app | Critical | Add app fallback to previous verified artifact and document rollback command/runbook |
| 24 | App health/freshness/degraded-mode status is incomplete | Users cannot tell whether data are stale, app is using fallback, or a refresh failed | Critical | Add visible health panel using manifest, source freshness, last refresh status, and fallback status |
| 25 | Public Streamlit live deployment is not re-verified after every push | GitHub pushes should auto-deploy, but live app state can lag or fail silently | High | Add post-push live health check and deployment-status evidence to each UI/data checkpoint |
| 26 | Candidate workflow and promotion workflow do not prove live publication | Automation builds candidates but does not by itself prove the public app updated | High | Add deployment verification and explicit release notes after promotion |
| 27 | Snapshot selection is not implemented in the app | Users cannot inspect YE2025 versus mid-2026 artifacts from the hosted UI | High | Add controlled snapshot selector backed by archived verified artifacts |
| 28 | Methodology tab is improved but needs user review | The new UI-native cards are clearer, but content should be reviewed after live deployment | Medium | Review the live Methodology tab and tune copy/layout based on product-owner feedback |
| 29 | Accessibility and responsive QA are not formalized | Light-mode fixes were made, but systematic UI QA is still informal | Medium | Add browser smoke coverage for light/dark mode, small screens, and key tabs |

#### 21.5.3 Sprint 3: Release Governance and Repository Hygiene

| Rank | Open issue | Why it matters | Priority | Sprint deliverable |
|---:|---|---|---:|---|
| 30 | Freshness, coverage, imputation, and score-change thresholds are not approved | The app can show data and scores, but users need reliable promotion gates behind them | Critical | Set thresholds for source staleness, minimum direct coverage, imputation share, and material score/rank/tier changes |
| 31 | Snapshot definitions and provisional/final rules are not approved | Users need to know whether a snapshot is preliminary, final, backfilled, or replaced by a later source vintage | Critical | Define snapshot lifecycle, naming, approval states, and replacement rules |
| 32 | Named model owner, data owner, and release approver are not encoded | Promotion remains dependent on informal review | High | Add CODEOWNERS/release checklist or equivalent governance control |
| 33 | Raw datasets remain in Git/LFS history | Repo/deploy size and LFS bandwidth remain higher than necessary | High | Move raw data to release/object-storage assets and plan any history rewrite as a separate controlled operation |
| 34 | Repository privacy/access-control decision is unresolved | Moving the repo private may affect Streamlit Cloud access and app visibility settings | Medium | Decide public/private repo and public/private app sharing model before adding licensed/private data |
| 35 | Manifest-builder paths can produce different metadata richness | Lightweight manifest generation omits official retrieval/validation metadata unless refresh flow adds it | Medium | Document script contract or add metadata-preserving update mode |
| 36 | Dependency/artifact portability still relies on pickle compatibility | Pickled artifacts can break across scikit-learn/XGBoost versions | Medium | Keep Python 3.11 constraints pinned; evaluate skops/ONNX/joblib policy for future releases |
| 37 | Existing diagnostic scripts remain partly untriaged | Maintenance burden and duplicate logic can confuse future work | Medium | Consolidate useful diagnostics into supported scripts and archive/delete stale scripts |
| 38 | Product positioning remains unresolved | Earlier "Copilot" wording suggested conversational or guided analytic capability, while current product is a dashboard/research tool | Medium | Decide whether to rename or implement governed conversational workflow |

#### 21.5.4 Sprint Execution Status (2026-07-10, Session 3)

Work executed against the backlog above. Statuses here supersede the table
rows; unlisted ranks are unchanged.

**Sprint 1 (model trust):**

- **Rank 1 — CLOSED (acceptance case met).** Driver-table tooling added
  (`python -m src.scripts.explain_country_scores`, published to
  `artifacts/score_drivers.json`) and the Kenya/Mozambique case is now an
  automated acceptance test
  (`tests/test_directional_scoring.py::test_challenger_acceptance_kenya_mozambique_and_no_derisking`).
  Challenger scores: MOZ 8.7 >= KEN 5.9 (production had KEN 8.7 > MOZ 7.5).
- **Rank 2 — CLOSED.** The classifier overlay is now upward-only:
  `uplift = max(0, 0.1 * ((1 + 9 * P) - pillar_score))`, monotone in crisis
  probability and unable to lower a high pillar score. Applied in both the
  training and fixed-pipeline inference paths; `crisis_uplift` is a published
  score column.
- **Rank 3 — CLOSED (design); challenger pending promotion review.** Every
  pillar feature now has a declared risk direction
  (`FEATURE_RISK_DIRECTIONS`); pillar components are constrained principal
  directions (negative loadings clipped, 50% shrinkage toward equal weights),
  so no feature can move the score in an economically counterintuitive
  direction. Monotonicity is unit-tested. Fitting fails loudly if a new
  feature lacks a declared direction.
- **Rank 4 — CLOSED.** Countries missing critical banking soundness fields
  receive a bounded, disclosed penalty (max +1.5) instead of benefiting from
  KNN imputation; `critical_missing_share` and `critical_penalty` are score
  columns and appear in the driver table with per-feature raw/imputed flags.
- **Rank 17 governance note applied.** The coverage-bias validation gate now
  measures the pre-policy score (penalties/uplift excluded) and the
  policy-inclusive correlation is reported separately.
- **Crisis-label provenance (July review item 20, re-triaged into this
  sprint):** reconciliation harness and checksum registry implemented; one
  manual download step remains (see section 21.2 item 20).
- **Ranks 5-16, 18-21 (external-liquidity data block, benchmarking, reference
  distributions, GDP-anchor policy) — OPEN.** These require new IMF
  dataflows (BOP, IIP, IRFCL, PIP/CPIS, Fiscal Monitor, GFS) and owner policy
  decisions; not addressable from code alone in this session.

**Challenger promotion decision (explicitly NOT taken unilaterally):** the
directional challenger re-ranks the universe far beyond every section 3
review threshold in `docs/GOVERNANCE.md` (mean |change| 2.27 points, 161/201
countries move >= 1 point, Spearman 0.35 vs production). The active serving
artifacts are therefore unchanged; the challenger is archived at
`artifacts/snapshots/2026-06-30-challenger-directional` with
`artifacts/challenger_comparison.json` documenting the largest movements and
three named concerns for review: (a) level-share features
(`real_estate_loans`, `fx_loan_exposure`) push mortgage-heavy advanced
economies (GBR/AUS/CAN) into high-risk tiers — replacing levels with
growth/price-based indicators (rank 21) is recommended first; (b) coverage
correlates ~-0.5 with the pure model score through observed governance
gradients — threshold decision is rank 30/owner; (c) post-crisis countries
(GHA, DOM) score low on clean recapitalized balance sheets — a crisis-recency
feature is recommended.

**Sprint 2 (app reliability):**

- **Rank 23 — CLOSED.** `load_model_artifact_with_fallback` serves the newest
  checksum-valid archived bundle when the active artifact is missing,
  corrupt, or fails validation; unit-tested including tamper detection. The
  runbook now has an executable manual-rollback procedure.
- **Rank 24 — CLOSED.** System Health panel in the app header: serving mode
  (active/fallback), snapshot ID/status/age, per-source freshness against the
  proposed SLAs, and explicit degraded-mode warnings (`src/health.py`,
  unit-tested).
- **Rank 22 — ADVANCED.** The driver table (rank 1 tooling) provides raw
  value, imputed value, direction, weight, contribution, and peer percentile
  per feature; surfacing it inside the Country Profile tab remains open.

**Sprint 3 (governance):**

- **Ranks 30-31 — DRAFTED, awaiting owner approval.** `docs/GOVERNANCE.md`
  proposes concrete freshness SLAs (enforced in the health panel), coverage
  and imputation gates (enforced where noted), score-change review thresholds
  (which this session's own challenger correctly trips), and the full
  snapshot lifecycle state machine. Approval is a one-line status change in a
  reviewed PR.
- **Rank 32 — CLOSED.** `.github/CODEOWNERS` (per-area ownership) and
  `docs/RELEASE_CHECKLIST.md` (promotion checklist referencing the governance
  thresholds).

#### 21.5.5 Session 3 Continuation — Owner Decisions Applied (2026-07-10)

The owner reviewed the session-3 output live and made four decisions, all
applied the same day:

1. **Governance approved (ranks 30-31 — CLOSED).** `docs/GOVERNANCE.md`
   status changed to APPROVED (@MMJGGR, 2026-07-10) as proposed: freshness
   SLAs, coverage/imputation gates, score-change review thresholds, snapshot
   lifecycle, and (added) the artifact-portability policy (rank 36 — CLOSED).
2. **Coverage-bias gate re-baselined (owner decision on the rank 30 open
   question).** `validate_model` now gates on the partial correlation of
   coverage with the pre-policy score, controlling for log GDP per capita and
   the six governance scores; raw correlations are reported as informational.
3. **Product name corrected (rank 38 / tracker item 17 — CLOSED).** The
   external app name is **BankEnv**; the prior verbose workbench name is
   rejected.
4. **Challenger iterated per owner instruction ("iterate first, then
   re-review"), still NOT promoted.** Challenger v2 removes the level-share
   features `real_estate_loans` and `fx_loan_exposure` from the pillar
   (growth-based measure retained; levels remain available to classifier and
   explorer) and adds a `years_since_banking_crisis` industry feature (capped
   at 25) so post-recapitalization balance sheets are not scored as pristine.
   Results: validation 3/3 PASSED (bias gate partial correlation -0.10);
   Kenya/Mozambique acceptance holds (MOZ 8.3 >= KEN 7.9); all three v1
   concerns resolved in direction (GBR 7.8→5.0 now below PAK 9.2; AUS
   7.6→2.1; GHA 2.1→9.5). Movement vs production is still far beyond
   promotion thresholds (mean |delta| 1.68, 135/201 countries >= 1 point,
   Spearman 0.62), so v2 replaces v1 in the archived bundle
   `artifacts/snapshots/2026-06-30-challenger-directional` with an updated
   `artifacts/challenger_comparison.json`; the active serving artifacts are
   unchanged pending the owner's promotion review.

Additional backlog closures in the continuation:

- **Rank 22 — CLOSED.** The Country Profile tab now has a "Score Drivers"
  expander computing per-feature attribution in-app (raw vs imputed value,
  critical flag, risk contribution, peer percentile, crisis uplift and
  missingness penalty metrics) for the selected country and snapshot.
- **Rank 27 — CLOSED.** Header snapshot selector loads any archived bundle
  read-only (checksum-verified); challenger bundles are labelled UNAPPROVED
  and a warning banner distinguishes viewing from serving.
- **Rank 35 — CLOSED.** `build_data_manifest.py` now documents its contract
  and preserves `retrieval`/`source_mode`/`validation` metadata from an
  existing manifest by default (`--fresh` to drop deliberately).
- **Rank 37 and review item 26 leftovers — CLOSED.** All 19 unreferenced
  one-off diagnostic scripts moved to `src/scripts/archive/` with a README
  naming the supported script set; `replication/outputs` was verified to be
  a single 104 KB tree (no duplicates remain).

Items that remain open and why they cannot be closed from this environment:

- **Ranks 5-14 (external-liquidity data block) — CLOSED for staged challenger
  feature coverage.** The broad full-flow downloader was replaced with a
  bounded, feature-oriented BOP/IIP retrieval path because IMF ignores
  `c[...]` query filters on these SDMX dataflow URLs. The new builder queries
  exact path keys by ISO3 country batches, writes normalized observations, and
  computes staged external-liquidity features: current-account receipts and
  payments, goods/services trade flows, reserve adequacy, net IIP, external
  liabilities, portfolio liabilities/flows, an investment-income-service proxy,
  and a gross-external-financing-need proxy. Local full-universe run:
  19,172 observations; 185/201 scored countries with at least one
  external-liquidity feature; BOP flow ratios cover 181/201 countries (90.0%);
  IIP position ratios cover 159-166/201 countries (79.1%-82.6%). CPIS, CDIS,
  and QEDS remain unavailable under the tested current IMF API IDs, but this is
  no longer blocking the MVP challenger block because BOP/IIP cover portfolio
  flows/positions and the exact contractual debt-service series is explicitly
  represented as a proxy pending a future QEDS/IDS enhancement. These features
  are not wired into production scoring until a separate challenger comparison
  is reviewed.
- **Ranks 15-16, 18 (reference-distribution policy, GDP-anchor policy,
  external benchmark):** owner methodology decisions plus (for 18) licensed
  or external outcome data.
- **Ranks 25-26 (live-deployment verification):** requires the deployed
  Streamlit Cloud URL and a push to the branch Streamlit deploys.
- **Rank 28 (Methodology copy review):** owner review of the live UI.
- **Rank 29 (formal accessibility/responsive QA):** needs a browser-matrix
  pass against the live deployment; local bare-mode and HTTP smoke checks
  are in place.
- **Rank 33 (LFS history rewrite):** destructive repository operation
  requiring owner coordination (all clones invalidated).
- **Rank 34 (repo privacy):** owner account decision.

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
- [x] Checkpoint 4: reproducible model-policy sensitivity audit covering GDP,
  confidence regression, risk floors, coverage correlation, and crisis-label
  composition.
  - Published commit: `8ad21c2`
  - Pull request: draft
    [#1](https://github.com/MMJGGR/banking-stability-copilot/pull/1)
  - Verification: 29 tests passed. Candidate refreshes now fail when model
    validation gates fail while still uploading diagnostic artifacts for
    review.
- [x] Checkpoint 5: cutoff-aware local cached-source model snapshots for
  YE2025 and mid-2026.
  - Pending commit: `build verified local snapshots`
  - Pull request: draft
    [#1](https://github.com/MMJGGR/banking-stability-copilot/pull/1)
  - Artifacts:
    - Active serving manifest: `artifacts/data_manifest.json` with
      `snapshot_id=2026-06-30`, `snapshot_status=verified`, and
      `model.cutoff_verified=true`.
    - Archived YE2025 bundle: `artifacts/snapshots/2025-12-31`.
    - Archived mid-2026 bundle: `artifacts/snapshots/2026-06-30`.
  - Verification: 31 tests passed; compile check passed; Streamlit browser
    smoke test rendered `Snapshot 2026-06-30 | verified`, 201 countries, and no
    browser console errors.
  - Caveat: official source endpoints are still not configured; both snapshots
    are built from approved local caches. The mid-2026 cutoff has no 2026 source
    observations in the current cache set.
- [x] Checkpoint 6: official API retrieval, cache normalization, and active
  mid-2026 serving artifact.
  - Pending commit: `wire official api refresh`
  - Pull request: draft
    [#1](https://github.com/MMJGGR/banking-stability-copilot/pull/1)
  - Artifacts:
    - Active serving manifest: `artifacts/data_manifest.json` with
      `snapshot_id=2026-06-30`, `snapshot_status=verified`,
      `source_mode=official_api_sdmx_worldbank`, and
      `model.cutoff_verified=true`.
    - Archived official API bundle:
      `artifacts/snapshots/2026-06-30-official-api`.
    - Raw official downloads retained locally under ignored
      `data/raw/official_refresh_20260710_001908`.
  - Verification: official source check passed; full official refresh passed;
    model validation 3 passed / 0 failed; 34 tests passed; compile check
    passed; local Streamlit returned HTTP 200 on port 8514.
  - Source coverage: WEO `IMF.RES:WEO(9.0.0)`, FSIC
    `IMF.STA:FSIC(13.0.1)` through 2026-04, MFS
    `IMF.STA:MFS_DC(8.0.0)` through 2026-05, FSIBSIS
    `IMF.STA:FSIBSIS(18.0.0)` through 2026-M04, WGI through 2024.
- [x] Checkpoint 7: Streamlit hosted startup memory fix.
  - Pending commit: `make historical data loads on demand`
  - Scope:
    - Removed full IMF cache loading from Streamlit startup.
    - Added country-sliced parquet reads for WEO, FSIC, and MFS history.
    - Added an explicit Data Explorer control so historical source data loads
      only when requested for the selected country.
  - Verification: `35 passed`; compile check passed; filtered USA reads
    returned WEO 1,990 rows, FSIC 11,927 rows, and MFS 18,432 rows; local
    Streamlit returned HTTP 200 on port 8520 with approximately 134 MB working
    set and 328 MB private memory after first response.
- [x] Checkpoint 8: Streamlit usability and comparison workflow pass.
  - Pending commit: `improve streamlit comparison ui`
  - Scope:
    - Replaced dark-only custom CSS with Streamlit theme-aware control,
      tab, card, and dropdown styling so light mode remains readable.
    - Replaced fixed peer display with an editable peer multiselect seeded
      from nearest-neighbor peers.
    - Added a cross-country indicator comparison panel for WEO, FSIC, MFS,
      and WGI, loading only the selected countries.
    - Removed hard-coded Plotly dark templates so charts follow the active
      Streamlit theme.
  - Verification: compile check passed; `40 passed`; local Streamlit rendered
    the Country Profile peer selector and Data Explorer comparison panel
    without a browser error.
- [x] Checkpoint 9: current Methodology tab and README status refresh.
  - Pending commit: `refresh methodology tab`
  - Scope:
    - Replaced README-rendered Methodology tab with a manifest-backed current
      methodology view tied to the active snapshot, model card, and data card.
    - Updated stale README release-status, source-count, and output-artifact
      claims to match the verified `2026-06-30` serving manifest.
  - Verification: compile check passed; unit suite passed; local Streamlit
    Methodology tab rendered current snapshot metadata without the old
    February 2026 legacy-artifact warning.
- [x] Checkpoint 10: Methodology active-source count correction.
  - Pending commit: `fix methodology source counts`
  - Scope:
    - Fixed FSIBSIS and WGI source summaries so active manifests report
      source-appropriate indicator counts instead of omitting the field.
    - FSIBSIS now reports unique `INDICATOR` labels with observations by the
      cutoff; WGI now reports populated governance score columns.
    - Updated the Methodology tab to label the field as
      `Indicators / Measures` and show the count basis rather than implying
      zero for source schemas without `indicator_code`.
    - Preserved official-refresh metadata in `artifacts/data_manifest.json`
      while adding the corrected counts.
  - Verification: active manifest now reports FSIBSIS 289 measures and WGI
    6 governance indicators; compile check passed; `42 passed`.
- [x] Checkpoint 11: Methodology model/data card display cleanup.
  - Pending commit: `improve methodology cards`
  - Scope:
    - Replaced raw `docs/MODEL_CARD.md` and `docs/DATA_CARD.md` dumps in the
      Streamlit Methodology tab with purpose-built UI cards.
    - Added a concise Model Card covering intended use, score construction,
      limitations, open review flags, and release governance.
    - Added a concise Data Card covering active source roles, source freshness,
      snapshot rules, direct-coverage watchlist, and priority missing data
      families such as debt-service burden, gross external financing needs,
      current-account receipts, reserves adequacy, and portfolio flows.
  - Verification: compile check passed; `42 passed`; local Streamlit returned
    HTTP 200 on port 8531 with no stderr.
- [x] Checkpoint 12: Data Explorer calculated-series builder.
  - Pending commit: `add data explorer calculated series`
  - Scope:
    - Added a bounded Data Explorer tool for raw multi-indicator panels,
      ratios, cross-sectional shares, period-over-period changes, base-period
      changes, and rebased indices.
    - Calculations align observations by country, date, and reporting
      frequency; no annual/quarterly/monthly series are silently mixed.
    - Ratio mode excludes zero denominators and displays the formula.
    - Cross-sectional share mode uses the selected-country group as the
      denominator for each aligned period.
    - Temporal modes calculate each country independently after the selected
      frequency and time window are applied.
  - Verification: compile check passed; `tests/test_calculated_series.py`
    passed; full suite passed (`48 passed`); local Streamlit returned HTTP
    200 on port 8532 with no stderr. Follow-up hotfix added explicit
    Streamlit element keys for calculated-series Plotly charts and latest-value
    tables after hosted Streamlit exposed duplicate auto-generated IDs in the
    raw multi-indicator loop; compile check, calculated-series tests, full
    suite (`48 passed`), and local Streamlit HTTP 200 on port 8533 passed.
- [x] Checkpoint 13: Data Explorer scope, FSIBSIS comparison access, and UX cleanup.
  - Pending commit: `improve data explorer scope and fsibsis tools`
  - Scope:
    - Moved the country selector out of the global header and into the Country
      Profile tab so Global Summary and Methodology remain explicitly global.
    - Added an independent Data Explorer focus-country selector used only for
      single-country source tabs and default comparison peer seeding.
    - Replaced the two large Data Explorer expanders with clearer analysis tabs
      for cross-country indicator comparison and calculated series.
    - Added FSIBSIS as a first-class source in both comparison and calculated
      indicator tools.
    - Added cached FSIBSIS wide-to-long conversion for annual, quarterly, and
      monthly balance-sheet observations; the single-country FSIBSIS tab now
      uses the same loader path as comparison/calculation tools.
    - Made comparison/calculated country selections reset per Explorer focus
      country to avoid stale peer selections after a country change.
  - Verification: compile check passed; targeted FSIBSIS cache check confirmed
    USA has 83 measures, 6,445 non-null observations, native annual/quarterly/
    monthly periods through 2026-04-30; full suite passed (`48 passed`); local
    Streamlit returned HTTP 200 on port 8507.
- [x] Checkpoint 14: External-liquidity SDMX discovery resolver fix.
  - Pending commit: `fix external liquidity sdmx discovery`
  - Scope:
    - Diagnosed the first `external-data.yml` run: the job succeeded but
      downloaded no large IMF dataflows because all candidate sources were
      marked unresolved.
    - Fixed the root cause: IMF SDMX 3.0 dataflow responses often contain only
      a DSD URN, not embedded dimensions. Discovery now follows the dataflow's
      `structure` reference to `/structure/datastructure/{agency}/{DSD}/+` and
      extracts ordered dimensions from the DSD.
    - Added discovered key structures under
      `config/external_sources_discovery.json` for BOP, IIP, IRFCL, Fiscal
      Monitor, and QGFS.
    - Made `fetch_external_sources.py` fail loudly if a workflow run fetches
      zero sources, avoiding another false-green Actions run.
  - Verification: `tests/test_external_sources.py` passed; local full
    discovery resolved BOP, IIP, IRFCL, FM, and GFS/QGFS; live endpoint smoke
    confirmed all five resolved data URLs return SDMX CSV; a local Fiscal
    Monitor fetch normalized 1,024 rows covering 18 countries.
- [x] Checkpoint 15: Bounded external-liquidity feature builder and workflow.
  - Pending commit: `add bounded external liquidity features`
  - Scope:
    - Added `src/external_liquidity.py` and
      `src/scripts/build_external_liquidity_features.py`.
    - Replaced broad full-dataflow workflow fetching with exact BOP/IIP
      path-key queries for model-country ISO3 batches.
    - Added staged derived features for current-account receipts/payments,
      goods/services flows, reserves adequacy, net IIP, external liabilities,
      portfolio liabilities/flows, an investment-income-service proxy, and a
      gross-external-financing-need proxy.
    - Reclassified CPIS/CDIS/QEDS as non-blocking future enhancements for the
      MVP challenger block: BOP/IIP provide portfolio flows/positions; exact
      contractual debt service remains proxied until QEDS/IDS is added.
    - Updated the external-data workflow and operations runbook to build and
      upload feature caches/reports instead of hanging on full-flow downloads.
  - Verification: local full-universe run fetched 19,172 observations and
    covered 185/201 scored countries with at least one external-liquidity
    feature. BOP current-account/trade ratios cover 181/201 countries (90.0%);
    IIP position/reserve ratios cover 159-166/201 countries (79.1%-82.6%).
    Full test suite passed (`78 passed`, `1 skipped`). These features remain
    staged challenger inputs and do not change production predictions until a
    challenger comparison is reviewed.
- [x] Checkpoint 16: Non-duplicative debt-service and financing-pressure
  source extension.
  - Pending commit: `add debt service financing pressure features`
  - Scope:
    - Add World Bank WDI/IDS only where it fills gaps not already covered by
      WEO/MFS/BOP/IIP: total external debt service, public-and-publicly-
      guaranteed external debt service, government interest payments as a
      share of revenue, and government revenue/GDP as a denominator.
    - Do not add WB current-account, reserves, or broad external-liquidity
      fallback series because those are already represented by the existing
      WEO/MFS/BOP/IIP pipeline and staged external-liquidity block.
    - Derive fiscal-first financing-pressure candidates:
      `wb_total_external_debt_service_gdp`,
      `wb_ppg_external_debt_service_gdp`,
      `wb_total_external_debt_service_revenue_proxy`,
      `wb_ppg_external_debt_service_revenue_proxy`, and
      `wb_public_financing_need_ext_debt_service_proxy_gdp`.
    - Keep the features staged as challenger inputs; do not change production
      scores or country rankings until a feature-impact review is completed.
  - Verification so far: targeted unit tests passed
    (`tests/test_external_liquidity_features.py`: `3 passed`) after removing
    duplicate-prone WB current-account and reserve series. Added `--skip-imf`
    so WB debt-service coverage can be tested without re-fetching the existing
    IMF external-liquidity block. Later packaging checks found the committed
    staged WB debt-service artifact has weak country coverage and no Kenya or
    Mozambique WB debt-service values; the IMF BOP/IIP staged features do
    cover both countries.
    WB-only 201-country run completed locally using the non-duplicative source
    set: government interest/revenue covers 142/201 countries (70.6%),
    revenue/GDP covers 143/201 (71.1%), total and PPG external debt-service/GDP
    cover 120/201 (59.7%), direct debt-service/export and debt-service/GNI
    measures cover 111-119/201 (55.2%-59.2%), and the debt-service/revenue
    proxy covers 85/201 (42.3%). Full combined IMF+WB local fetch exceeded the
    local command timeout and should run in GitHub Actions or be split by
    source family before using it as a release gate. Full test suite passed
    (`79 passed`, `1 skipped`).
- [x] Checkpoint 17: Live Streamlit startup import hotfix.
  - Pending commit: `fix streamlit startup import compatibility`
  - Scope:
    - Replaced the fragile multi-symbol `src.model_store` import in `app.py`
      with a module import plus backward-compatible fallbacks for archive-aware
      snapshot helpers. This prevents a live startup failure when Streamlit
      serves a stale/partial module cache that lacks newer model-store helper
      symbols.
    - Removed `key=` arguments from `st.dataframe` calls because explicit keys
      are needed for Plotly duplicate-element fixes, but dataframe keys are not
      supported consistently across the Streamlit versions covered by the
      serving dependency range.
  - Verification: `python -m py_compile app.py src/model_store.py` passed; full
    test suite passed (`79 passed`, `1 skipped`); local Streamlit startup check
    returned HTTP 200 on port 8562.
- [x] Checkpoint 18: External-facing UX cleanup.
  - Pending commit: `clean streamlit external ux`
  - Scope:
    - Removed the top headline, global implementation-note caption, and
      horizontal divider so the app opens directly on the primary navigation.
    - Hid snapshot selection and system-health diagnostics from the default
      frontend. They remain available only when `SHOW_ADMIN_DIAGNOSTICS=true`
      is set.
    - Removed decorative icons from status labels and data captions.
    - Reduced top padding and shortened Data Explorer helper text for a more
      utilitarian layout.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8563.
- [x] Checkpoint 19: BankEnv naming and favicon.
  - Pending commit: `rename app bankenv`
  - Scope:
    - Renamed the external app to **BankEnv** in Streamlit metadata, README,
      source docstrings, operations runbook, and the working plan.
    - Added `assets/bankenv-favicon.svg`: a minimal dark favicon with `BE`
      lettering and a baseline motif.
    - Removed vendor-adjacent styling language from source comments.
  - Verification: `python -m py_compile app.py src/config.py
    src/dashboard/components.py src/dashboard/styles.py src/crisis_classifier.py
    src/crisis_labels.py src/feature_engineering.py` passed; full test suite
    passed (`79 passed`, `1 skipped`); local Streamlit startup check returned
    HTTP 200 on port 8564.
- [x] Checkpoint 20: Menu spacing and regional chart cleanup.
  - Pending commit: `fix menu spacing and regional chart legend`
  - Scope:
    - Restored sufficient top padding so Streamlit's toolbar no longer clips
      the primary tab/menu row after the headline removal.
    - Replaced the regional risk bar chart's continuous colorbar with fixed
      risk-band bar colors and no legend.
  - Verification: `python -m py_compile src/dashboard/global_view.py
    src/dashboard/styles.py app.py` passed; full test suite passed
    (`79 passed`, `1 skipped`); local Streamlit startup check returned HTTP
    200 on port 8565.
- [x] Checkpoint 21: Visible BankEnv app brand.
  - Pending commit: `show bankenv brand in app`
  - Scope:
    - Added a compact BankEnv brand row above the primary tabs using the same
      `BE` mark as the favicon.
    - Kept the layout utilitarian: no marketing headline and no extra
      explanatory copy.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8566.
- [x] Checkpoint 22: BankEnv logo refinement.
  - Pending commit: `refine bankenv logo`
  - Scope:
    - Replaced the plain `BE` tile with a compact bank/operating-environment
      signal mark.
    - Applied the same mark to the favicon and in-app brand row for visual
      consistency.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8567.
- [x] Checkpoint 23: Theme-aware BankEnv logo.
  - Pending commit: `adapt bankenv logo for dark mode`
  - Scope:
    - Converted the in-app BankEnv mark from fixed SVG colors to CSS variables.
    - Added dark-mode CSS overrides so the in-app tile inverts to a light mark
      on dark backgrounds while retaining the same motif.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8568.
- [x] Checkpoint 24: BankEnv logo readability pass.
  - Pending commit: `simplify bankenv logo mark`
  - Scope:
    - Replaced the house/temple-like mark with a larger operating-environment
      analytics mark: axes, bars, and trend line.
    - Increased in-app logo size to improve readability and reduce smudging.
    - Preserved dark-mode adaptation through CSS variables.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8569.
- [x] Checkpoint 25: App-wide staged external-liquidity visibility.
  - Pending commit: `surface staged external liquidity appwide`
  - Scope:
    - Packaged compact derived external-liquidity reference files under
      `data/reference/` so the hosted Streamlit app can display the new data
      without downloading large IMF/WB flows at startup.
    - Added the staged external-liquidity source to Data Explorer comparison
      and calculated-series tools, plus a dedicated External Liquidity panel
      with latest country values, coverage, cross-country comparison, and
      explicit score-role labelling.
    - Added Methodology/Data Card coverage for the staged dataset and changed
      gap statuses from generic missing items to staged/proxy/not-yet-scored
      where applicable.
    - Added a Country Profile hook that will display external-liquidity fields
      only if a future promoted production model includes those fields in its
      approved feature artifact.
    - Confirmed active artifacts still do not use the staged external-liquidity
      fields in production scoring. The app therefore exposes them for insight,
      not as hidden score drivers.
    - Documented artifact mismatch risk for Kenya/Mozambique interpretation:
      `cache/risk_model.pkl` currently scores Kenya higher risk than
      Mozambique, while the dated 2026-06-30 and directional challenger
      artifacts score Mozambique higher.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py
    src/external_liquidity.py` passed; full test suite passed (`79 passed`,
    `1 skipped`) with repo root on `PYTHONPATH`; local Streamlit startup check
    returned HTTP 200 on port 8573.
- [x] Checkpoint 26: Production liquidity wiring and 2026-06-30 artifact
  promotion.
  - Commits/PRs:
    - `5cf7905` merged `claude/remediation-implementation-plan-y7d3xc` into
      `master`, adding production training-path wiring for liquidity and
      affordability features plus government-liquidity reference data.
    - Candidate build run `29156989689` completed successfully for
      `as_of_date=2026-06-30`: CI tests passed, official sources refreshed,
      candidate snapshot built, smoke tests passed, and candidate artifacts
      uploaded.
    - Promotion run `29157371667` downloaded the candidate bundle, re-ran
      serving smoke tests, and pushed `promote/2026-06-30`.
    - Promotion PR #9 merged to `master` as `0d3d3ad`, committing refreshed
      serving artifacts via LFS.
  - Current served artifact:
    - `artifacts/data_manifest.json` reports snapshot `2026-06-30`, status
      `verified`, source mode `official_api_sdmx_worldbank`.
    - Production feature matrix now includes
      `net_iip_gdp`, `external_liabilities_gdp`,
      `reserves_to_goods_services_imports`, and
      `gross_external_financing_need_proxy_gdp`.
    - Current production scores after promotion: USA `6.9`, Kenya `8.0`,
      Mozambique `8.5`.
  - Artifact hygiene:
    - Removed stale local-only smoke/download artifacts from the working tree.
    - Added `.gitignore` rules so future local smoke artifacts and downloaded
      workflow bundles do not appear beside tracked production artifacts.
  - Follow-up:
    - Peer selection remains weak because `find_peers()` still uses only
      economic and industry pillar Euclidean distance. USA defaults are no
      longer Cyprus/Italy/Dominica/Fiji after the promoted model, but the
      resulting peers are still not methodologically defensible. Replace with
      a peer engine that combines region/income/development scale, banking
      structure, macro/liquidity features, and data-quality filters.
- [x] Checkpoint 27: Government-liquidity app-wide surfacing cleanup.
  - Pending commit: `surface government liquidity across app`
  - Scope:
    - Confirmed the promoted production artifact uses
      `govt_interest_to_revenue` and `govt_debt_to_revenue`.
    - Added Government liquidity to the generic Data Explorer source dropdowns
      for cross-country indicator comparison and calculated-series tools.
    - Added Country Profile display of government-liquidity model inputs when
      they are present in the active production feature matrix.
    - Cleaned frontend wording so the app does not describe currently
      production-scored liquidity fields as purely staged challenger inputs.
    - Updated the government-liquidity report/template notes and Data Card
      documentation to distinguish active model inputs from insight-only
      packaged fields.
  - Verification: `python -m py_compile app.py src/government_liquidity.py`
    passed; full test suite passed (`90 passed`, `1 skipped`); local
    Streamlit startup check returned HTTP 200 on port 8575.
- [x] Checkpoint 28: Peer-engine replacement and liquidity integration audit.
  - Pending commit: `replace weak peer selector`
  - Liquidity integration confirmed:
    - Production-scored government-liquidity inputs:
      `govt_interest_to_revenue`, `govt_debt_to_revenue`.
    - Production-scored external-liquidity inputs:
      `net_iip_gdp`, `external_liabilities_gdp`,
      `reserves_to_goods_services_imports`,
      `gross_external_financing_need_proxy_gdp`,
      `investment_income_debits_to_cxr`.
    - These columns are in the promoted `cache/risk_model.pkl` feature matrix,
      included in `src/pillar_pipeline.py` `ECONOMIC_FEATURES`, have declared
      entries in `FEATURE_RISK_DIRECTIONS`, and are assembled into training by
      `src/liquidity_features.py`.
  - Peer fix:
    - Replaced the old `find_peers()` distance, which used only economic and
      industry pillar scores, with a multi-factor peer engine using model score
      proximity, economic scale, GDP per capita, banking ratios, government
      liquidity, external liquidity, and data-coverage filtering.
    - Updated Country Profile and Data Explorer to pass the active model
      feature matrix to the peer engine.
    - Added a regression test proving that a small economy with matching pillar
      scores no longer outranks structurally comparable large/high-income peers
      when model features are available.
    - Current promoted-artifact USA defaults changed to United Kingdom,
      Germany, France, Italy, Canada, and Japan.
  - Verification: `python -m py_compile app.py src/utils.py` passed; full test
    suite passed (`91 passed`, `1 skipped`); local Streamlit startup check
    returned HTTP 200 on port 8576.
- [x] Checkpoint 29: Score-driver summary metric fallback.
  - Pending commit: `fix score driver summary metrics`
  - Scope:
    - Score Drivers table was rendering, but the three summary metrics could
      display as large blank dashes on the hosted app when the driver payload
      omitted `crisis_uplift`, `critical_missing_share`, or
      `critical_penalty`.
    - Added a fallback from the driver payload to the selected country score
      row, so the metrics display the active artifact values already shown in
      `country_scores`.
    - Changed the true-missing display from an em dash to `n/a` so legacy
      artifacts do not look like blank metrics on mobile/dark mode.
  - Verification: `python -m py_compile app.py` passed; full test suite
    passed (`91 passed`, `1 skipped`); local Streamlit startup check returned
    HTTP 200 on port 8577.
- [x] Checkpoint 30: Legacy-artifact score-driver fallback.
  - Pending commit: `derive legacy score driver metrics`
  - Scope:
    - Live mobile screenshot showed the Score Drivers table populated but the
      three summary metrics displayed `n/a`, indicating the hosted app was
      still serving or falling back to an artifact whose `country_scores` did
      not carry the newer summary columns.
    - Added tested fallback logic that derives critical-field missing share and
      missingness penalty directly from the driver rows and fitted pipeline
      when the artifact lacks explicit fields.
    - Crisis uplift now defaults to `+0.00` for legacy artifacts without an
      explicit additive-uplift field.
  - Verification: `python -m py_compile app.py src/utils.py` passed; full test
    suite passed (`93 passed`, `1 skipped`); local Streamlit startup check
    returned HTTP 200 on port 8578.
- [x] Checkpoint 31: Peer-engine mixed-deploy compatibility hotfix.
  - Pending commit: `guard peer selector stale imports`
  - Scope:
    - Live Streamlit error showed new `app.py` calling
      `find_peers(..., feature_values=model_features)` while the imported
      `src.utils.find_peers` still had the old three-argument signature.
    - Added `safe_find_peers()` in `app.py` to use the new multi-factor peer
      engine when available and fall back to the old signature instead of
      crashing during mixed/stale Streamlit redeploys.
    - Routed Country Profile and Data Explorer peer defaults through the safe
      wrapper.
  - Verification: `python -m py_compile app.py` passed; full test suite
    passed (`93 passed`, `1 skipped`); local Streamlit startup check returned
    HTTP 200 on port 8579.
- [x] Checkpoint 26: Staged general-government (sovereign fiscal) liquidity
  block.
  - Pending commit: `add staged government liquidity features`
  - Owner steer: "It should be full government liquidity not only external."
    The prior staged block covered only *external* liquidity (BOP/IIP + WB
    external debt service); the government's own fiscal liquidity was not a
    first-class family. This addresses part of backlog ranks 6/12/13
    (debt-service burden, fiscal affordability, gross financing need) at the
    staged-challenger level.
  - Scope:
    - Added `src/government_liquidity.py` and
      `src/scripts/build_government_liquidity_features.py`. The block is derived
      from the already-cached IMF WEO general-government series
      (`GGXWDG_NGDP`, `GGXCNL_NGDP`, `GGXONLB_NGDP`, `GGR_NGDP`, `GGX_NGDP`,
      `GGSB_NPGDP`), so it needs no external API and is buildable/testable
      locally. Cutoff and observation-status selection match the production WEO
      feature path (actuals + estimates only; projections excluded).
    - Derived features: gross debt/GDP, fiscal/primary/structural balance,
      implied interest burden (primary minus overall balance), the
      rating-agency affordability ratios interest-to-revenue and
      debt-to-revenue, and overall/primary deficit financing-flow signals.
    - Surfaced app-wide by generalizing the staged-insight render helpers: a new
      "Government liquidity" Data Explorer tab, a Methodology "Staged
      Government-Liquidity Dataset" summary, and an updated missing-data-family
      row. Features are staged challenger inputs and do not change production
      scoring.
    - Packaged compact outputs under `data/reference/` for the hosted app;
      documented the block in `docs/DATA_CARD.md` and `data/reference/README.md`.
      Follow-up (blocked by GitHub App `workflows` permission this session):
      add a build step to `.github/workflows/external-data.yml` so the staged
      cache is rebuilt in CI; for now it is rebuilt locally with
      `python -m src.scripts.build_government_liquidity_features --reference-dir data/reference`.
  - Remaining fiscal gap (registered, not closed): a full gross financing need
    needs debt amortization/rollover from IMF Fiscal Monitor / GFS, which WEO
    does not carry.
  - Verification: `python -m py_compile app.py src/government_liquidity.py
    src/scripts/build_government_liquidity_features.py` passed; full test suite
    passed (`84 passed`, `1 skipped`); real build against the resolved WEO cache
    covered 94-100% of model countries for the core affordability ratios
    (structural balance ~42%) with economically sensible values (e.g. Kenya
    interest/revenue ~30%, Egypt/Pakistan/Sri Lanka highest interest burden,
    Japan debt/revenue ~577%); local Streamlit startup returned HTTP 200 on
    port 8599 with no stderr.
- [x] Checkpoint 27: Market and external stress inputs (backlog ranks 19-21).
  - Pending commit: `add market and external stress features`
  - Owner steer: "also need market and external items."
  - Scope (extends the staged external-liquidity block, reusing its SDMX/World
    Bank fetch path, so no new workflow is required — `external-data.yml`
    already runs `build_external_liquidity_features --fetch`):
    - Rank 19 (FDI flow stability): IMF BOP direct-investment liability and net
      flows (functional category `D_F`), plus `stable_financing_share` (FDI over
      gross inward FDI+portfolio flows).
    - Rank 20 (export concentration / terms of trade): World Bank net-barter
      terms-of-trade index and a `commodity_export_share_pct` proxy summing
      fuel, ores/metals, agricultural-raw, and food merchandise-export shares.
    - Rank 21 (REER valuation stress): World Bank real-effective-exchange-rate
      index plus `reer_appreciation_5y_pct` (latest REER vs trailing five-year
      mean). Equity-price and property-price stress remain uncovered (no
      reliable public API series wired yet) and are registered in the report
      notes as a follow-up.
    - Added human-readable labels and unit tests; the features flow through the
      existing staged External Liquidity panel automatically.
  - Model status: these remain **staged challenger inputs**, not production
    scoring, exactly like the rest of the external and government blocks.
    Promotion requires a challenger-vs-production comparison and owner approval
    under `docs/GOVERNANCE.md`.
  - Data status: the committed `data/reference/external_liquidity_features.parquet`
    gains the new columns (currently null) on the next CI `--fetch` run, because
    api.imf.org / api.worldbank.org are blocked in the dev environment.
  - Verification: `python -m py_compile src/external_liquidity.py app.py`
    passed; full test suite passed (`87 passed`, `1 skipped`); build runs
    gracefully on the existing committed observations (new columns present,
    null until fetched).
- [x] Checkpoint 28: Liquidity-feature challenger (features fed into the model).
  - Pending commit: `add liquidity feature challenger`
  - Owner steer: "They should be fed into the model... tastefully." The staged
    external + government liquidity features are now wired into the real pillar
    pipeline and evaluated as a governed challenger; the active serving
    artifacts are unchanged pending owner promotion review.
  - Curation (tasteful, non-duplicative): only genuinely new, data-backed
    signals were added to the economic pillar, with a declared
    `FEATURE_RISK_DIRECTIONS` entry each. Government block contributed the two
    affordability ratios `govt_interest_to_revenue` and `govt_debt_to_revenue`
    (the debt/balance levels duplicate existing `govt_debt_gdp` /
    `fiscal_balance_gdp` and were skipped). External block contributed
    `net_iip_gdp`, `external_liabilities_gdp`,
    `reserves_to_goods_services_imports`,
    `gross_external_financing_need_proxy_gdp`, and
    `investment_income_debits_to_cxr`. Market/FDI/REER were omitted because
    they are null until the next CI fetch.
  - Wiring: `train()` gained an optional `extra_features` merge (left-merge,
    country universe unchanged; production passes nothing and is unaffected).
    `src/scripts/build_liquidity_challenger.py` runs a control train (no
    extras) and a challenger train (with extras), both reusing the cached
    classifier so the crisis overlay is held fixed, and writes
    `artifacts/liquidity_challenger_comparison.json` plus an archived
    `artifacts/snapshots/2026-06-30-challenger-liquidity/challenger_scores.parquet`.
    It never calls `model.save()`.
  - Effect on model output (isolated challenger-vs-control): mean |score change|
    0.19, 5 countries move >= 1 point, 21 risk-tier changes, Spearman 0.989 —
    a sharpening, not a re-ranking. Economically sensible: Singapore -2.1
    (large net creditor + reserve cover), Macao/Saudi/Aruba safer; Micronesia
    +2.0, Mongolia +1.1, Cyprus +1.0, Sri Lanka +0.8 riskier on weak
    external/fiscal liquidity.
  - Incidental finding: the control train does NOT reproduce the active
    `cache/risk_model.pkl` (mean |delta| 1.68, Spearman 0.62), i.e. the active
    serving pickle is stale relative to the current pipeline (consistent with
    the previously noted Kenya/Mozambique artifact mismatch). The governance
    gate is therefore evaluated on the isolated feature effect, not the
    confounded headline-vs-production delta; rebuilding the active artifact is a
    separate action.
  - Promotion: NOT taken. Only the tier-change threshold (21 > 15) trips on the
    isolated effect (mean 0.19 < 0.5, Spearman 0.989 > 0.90), so this is a much
    milder, more promotable change than the directional challenger, but it still
    requires owner review under docs/GOVERNANCE.md.
  - Verification: `python -m py_compile` on the changed modules passed; full
    test suite passed (`90 passed`, `1 skipped`) including new challenger-logic
    unit tests.
- [x] Checkpoint 29: Liquidity challenger PROMOTED to the active model + durable
  wiring (owner: "proceed with both").
  - Pending commit: `promote liquidity model and refresh active artifacts`
  - Scope:
    - Added `src/liquidity_features.py` as the single source of truth for the
      promoted liquidity features and wired both production training entry
      points (`refresh_data.py`, `build_local_snapshot.py`) to assemble and
      pass them, so every future refresh keeps the features (they are no longer
      challenger-only).
    - Rebuilt and VERIFIED the serving artifacts locally at cutoff 2026-06-30
      with the liquidity features live (`cache/risk_model.pkl`,
      `cache/inference_pipeline.pkl`, `cache/crisis_features.parquet`,
      `cache/imputed_features.parquet`, refreshed manifest and policy audit).
      Publication of these LFS artifacts must run through CI: this dev
      environment's proxy returns Forbidden on `lfs.github.com` uploads, so the
      code wiring is pushed and the artifacts are regenerated + committed via
      Git LFS by `refresh-data.yml` / `promote-snapshot.yml` in GitHub Actions.
    - Local validation: 3 passed / 0 failed; snapshot status verified; partial
      coverage-bias correlation -0.38 (within the approved gate). The seven
      liquidity features are live in the feature matrix (coverage 162-193/201).
    - Kenya/Mozambique in the rebuilt model: MOZ 8.5 (rank 35) > KEN 8.0
      (rank 49) — the rank-1 acceptance case is resolved once the CI-built
      artifact is published.
  - Wiring scope (important): the features are inputs to the PILLAR (structural
    risk score). They are NOT inputs to the supervised crisis classifier, which
    retains its literature-based feature set. External BOP/IIP features cannot
    join the classifier because it trains on a year-matched historical panel and
    those series exist only for the latest cross-section; the government
    affordability ratios (interest/revenue, debt/revenue) are year-matchable
    from WEO and remain a candidate classifier extension.
  - Crisis classifier confusion matrix (grouped out-of-fold, 2,178
    country-epochs, 8.5% base rate, 158 countries): AUC 0.564 — a deliberately
    weak early-warning signal, consistent with the documented 0.57-0.59. At the
    default 0.5 threshold the calibrated model fires on nobody; at the
    Youden-optimal threshold (~0.07) recall 0.87 / precision 0.10. It enters the
    hybrid score only as a 10% upward-only overlay.
  - Data vintage of the active snapshot: WEO IMF.RES:WEO(9.0.0), actuals through
    2025-12-31 (Oct-2025 vintage); FSIC 13.0.1 through 2026-04; MFS_DC 8.0.0
    through 2026-05; FSIBSIS 18.0.0 through 2026-M04; World Bank WGI through
    2024. Retrieved 2026-07-09/10.
  - Workflow status: source-check.yml (weekly) detects new vintages and opens an
    issue; refresh-data.yml (monthly + manual dispatch) builds a candidate
    snapshot that now includes the liquidity features; publication remains a
    reviewed step by design. Auto-update is intentionally gated, not silent.
- [x] Checkpoint 30: Sovereign affordability ratios added to the crisis
  classifier (owner: "add to classifier").
  - Pending commit: `add affordability ratios to crisis classifier`
  - The two government affordability ratios are year-matchable from historical
    WEO, so unlike the latest-only external block they are valid inputs to the
    classifier's temporal epoch panel. `_extract_weo_at_year` now also pulls
    GGR_NGDP (revenue) and derives `govt_interest_to_revenue` and
    `govt_debt_to_revenue` per epoch (interest = primary - overall balance;
    revenue kept only as a denominator, not a feature). Both are declared +1 in
    `MONOTONE_DIRECTION` and added to `FEATURE_PRIORITY`.
  - Effect: grouped out-of-fold AUC improved from 0.564 to 0.616 and false
    positives fell (TN 491 -> 660 at the Youden threshold); recall 0.83,
    precision 0.10. Still a weak-but-improved early-warning signal feeding the
    10% upward-only overlay. Full suite: 90 passed, 1 skipped.
- [x] Checkpoint 31: Live-app import crash guarded for mixed Streamlit deploys.
  - Commit: `guard app utility imports for live deploys`
  - Issue: live `bankenv.streamlit.app` showed an `ImportError` during
    `app.py` startup at `from src.utils import driver_metric...`. Local and
    GitHub `master` already contained `driver_metric_value`, so the failure was
    consistent with Streamlit serving a refreshed `app.py` against a stale or
    cached helper module during redeploy.
  - Fix: `app.py` now imports `src.utils` as a module, binds `find_peers` from
    that module, and uses a local compatibility fallback for
    `driver_metric_value` if the helper function is unavailable. This prevents
    the whole app from failing at import time while preserving the shared helper
    when the deployment is coherent.
  - Verification: direct `import app` succeeds locally; targeted tests passed:
    `pytest tests/test_peer_selection.py tests/test_model_store.py -q`
    (7 passed).
- [x] Checkpoint 32: Peer selector fallback hardened for stale Streamlit helper
  modules.
  - Commit: `harden app peer fallback for stale deploys`
  - Issue: the live app still showed USA default peers as Cyprus, Italy,
    Dominica, and Fiji. That is the old two-pillar fallback behavior: when a
    stale `src.utils.find_peers` rejects the `feature_values=` argument, the
    app previously called the old selector without model features.
  - Fix: `app.py` now contains an app-local robust peer selector equivalent to
    the shared utility implementation. If Streamlit serves a mixed deployment
    where the helper module is stale, `safe_find_peers` falls back to the
    robust selector rather than the two-pillar legacy selector.
  - Verification: local normal mode and simulated stale-helper mode both return
    USA peers as United Kingdom, Germany, France, Italy, Canada, and Japan;
    Cyprus, Dominica, and Fiji are excluded. `python -m py_compile app.py
    src/utils.py` passed; targeted tests passed:
    `pytest tests/test_peer_selection.py tests/test_model_store.py -q`
    (7 passed).
