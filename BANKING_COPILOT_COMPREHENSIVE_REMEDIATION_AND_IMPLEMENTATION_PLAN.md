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
- Keep the product positioned as a transparent analytical workbench, not an interactive AI product.
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

Section 21.5 is the authoritative **Current Closure Register**. It records what
was completed, what was removed from the active remediation scope, and the
specific evidence or re-entry gate for each retired item. Earlier issue tables
are historical discovery records; they do not represent current open work.

## 2. Historical Issue Register (Cleared, Superseded, or Carried Forward)

This section is not a current backlog. It records the original findings that
drove the remediation work. Current dispositions are consolidated in section
21.5; no earlier row should be read as a standalone open issue.

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
this plan against the code and identified the following additional issues. This
table is the discovery record; final dispositions are in section 21.5.

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

## 21. Delivery Action Tracker and Closure Register

This section records work execution history and final disposition. Section
21.5 is the authoritative closure register. A checked item has been implemented
and verified or has been deliberately retired with a recorded reason; it does
not imply that a failed research candidate was approved for production.

1. [x] Fix the critical classifier and application defects.
2. [x] Establish a supported Python 3.11 dependency baseline.
3. [x] Add the minimum automated test suite and pull-request CI.
4. [x] Snapshot definitions and provisional/final rules approved in
   `docs/GOVERNANCE.md`.
5. [x] Implement period parsing and actual/estimate/projection controls.
6. [x] Rebuild `2025-YE` as the verified cached-source baseline. Archived
   bundle: `artifacts/snapshots/2025-12-31`; later official-API refreshes
   reproduced the end-to-end source path.
7. [x] Redesign validation around country-grouped and out-of-time tests.
8. [x] Crisis labels, GDP input treatment, confidence floors, and risk-floor
   sensitivity audited. The exact IMF WP/26/94 episode artifact and checksum are
   pinned; the current pillar policy and limitations are recorded in
   `artifacts/model_policy_audit.json`, `MODEL_CARD.md`, and governance docs.
9. [x] Persist the complete fitted inference pipeline. New candidate artifacts
   preserve classifier fill values and calibration plus pillar imputation,
   scaling, PCA orientation, and fixed reference distributions. The active
   `2026-06-30` artifact now includes the sidecar inference pipeline and a
   verified manifest.
10. [x] Implement WEO, FSIBSIS, FSIC, MFS, and WGI source adapters.
11. [x] Freshness, coverage, imputation, and score-change thresholds defined
    and approved in `docs/GOVERNANCE.md`.
12. [x] Produce a provisional `2026-H1`/mid-2026 local cached-source snapshot.
    Archived bundle: `artifacts/snapshots/2026-06-30`; active Streamlit
    manifest now displays `Snapshot 2026-06-30 | verified`. `2026-Q1` remains
    optional if a separate quarter-end checkpoint is required.
13. [x] Remove legacy raw source exports from the current branch and keep raw
    retrievals outside routine version control. Candidate workflow artifacts,
    compact serving caches, and `data/raw/` exclusion are implemented. A
    destructive history rewrite was explicitly retired because it would break
    existing clones without improving the deployed checkout.
14. [x] Promotion automation added 2026-07-10: `promote-snapshot.yml`
    downloads a candidate bundle, re-runs the serving-artifact smoke tests
    (`src/scripts/smoke_test_artifacts.py`, also gating `refresh-data.yml`),
    commits artifacts via Git LFS, and opens a reviewed promotion pull
    request. Final merge remains an intentional human governance gate; release
    tags are optional because the promoted snapshot and merge commit provide
    the release identifier.
15. [x] Complete the Streamlit serving-only refactor. Lightweight model loading,
    derived caches, manifest display, on-demand country-sliced WEO/FSIC/MFS
    history, on-demand FSIBSIS loading, theme-safe controls, customizable
    peer sets, and cross-country indicator comparison are complete; active
    verified snapshot display is confirmed in browser; last-known-good
    fallback completed 2026-07-10 (session 3); health/freshness diagnostics
    and snapshot selection are hidden from the default frontend and gated
    behind `SHOW_ADMIN_DIAGNOSTICS=true`.
16. [x] Reconcile current documentation warnings and add model/data cards and an
    operations runbook. Active release evidence is artifact-backed; invalidated
    candidate metrics are suppressed until a future candidate passes governance.
17. [x] Updated 2026-07-11 (owner): external app name standardized as
    **BankEnv** (page title, favicon, README, and working plan). Rejected
    names and vendor-adjacent naming patterns should not be reintroduced.
18. [x] Monitoring, rollback, ownership, and release procedures complete:
    automatic last-known-good fallback, admin diagnostics, executable rollback
    runbook, CODEOWNERS, release checklist, approved governance thresholds, and
    live desktop/mobile smoke verification. Automated endpoint verification is
    installed with the final closure checkpoint.

### 21.1a Refresh and Publication Workflow — Current State (2026-07-14)

The 2026-07-09 probe below was superseded by the installed production
workflows and is no longer an open blocker:

- `source-check.yml`, `refresh-data.yml`, `quality-check.yml`,
  `external-data.yml`, and `promote-snapshot.yml` are on `master`; refresh,
  external-data, promotion, and quality runs have completed successfully.
- Official IMF SDMX and World Bank endpoints are the primary source path. The
  scheduled workflows do not require the obsolete ten custom URL secrets and
  intentionally check out without historical Git LFS payloads.
- Candidate refreshes smoke-test serving artifacts before upload. Promotion
  downloads the exact candidate, repeats the smoke test, and opens a reviewed
  pull request; human merge is the publication control.
- The legacy root-level raw IMF exports were removed from the current branch.
  Compact caches and manifests remain versioned for deterministic serving;
  destructive Git-history rewriting was rejected as operationally unnecessary.
- `https://bankenv.streamlit.app` was re-verified on 2026-07-14: all four
  primary tabs rendered the 201-country 2026-06-30 snapshot without a Streamlit
  exception on desktop and mobile viewports. A mobile header-clipping defect
  found in this check is fixed in the closure checkpoint.
- Weekly source availability plus the monthly candidate refresh provide the
  dynamic update loop. A separate version-diff issue bot was removed from scope
  because it duplicates the scheduled refresh and GitHub already alerts on
  failed workflows.


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
20. [x] Automated crisis-label reconciliation, exact IMF WP/26/94 PDF
    extraction, checksum pinning, and dictionary verification implemented and
    tested (`src/scripts/verify_crisis_labels.py`, `data/reference/README.md`,
    `tests/test_verify_crisis_labels.py`).
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
    explicitly disclaim the BIS HP-filter gap. An optional BIS research adapter
    was later added; production integration was retired under section 21.5.4.
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
26. [x] Dead modules removed, 19 diagnostic scripts archived, and replication
    outputs reviewed and deduplicated where safe.
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

Historical governance note: the retrained 12-epoch classifier replaced the
leakage-inflated 0.84 claim with materially weaker evidence. Checkpoints 45-47
then invalidated threshold-dependent claims, built the clean validation
foundation, and stopped promotion after the forward-time gate failed.

### 21.5 Current Closure Register (Updated 2026-07-14)

This register replaces the earlier ranked backlog. Each original item was
rechecked against the current code, active artifacts, tests, workflows, and live
application. There are no unchecked production-remediation items in this plan.

"Closed" means implemented and verified. "Retired" means deliberately removed
from the active remediation scope because it is a duplicate, a product-scope
mismatch, a source/licensing constraint, or unsafe to claim without new
evidence. Retired research may re-enter only through the stated gate.

#### 21.5.1 Production model and data — closed

| Original rank/finding | Final disposition | Evidence |
|---|---|---|
| 1 Kenya/Mozambique ordering | Closed. The active 201-country artifact now scores Mozambique 8.6 and Kenya 8.0; the directional acceptance test prevents the original inversion from silently returning. | `cache/risk_model.pkl`; `tests/test_directional_scoring.py` |
| 2 crisis uplift lowered risk | Closed. The classifier contribution is upward-only and cannot reduce the structural score. | `src/train_model.py`; directional tests |
| 3 economically reversed feature signs | Closed for the pillar model. Every active pillar input has an explicit risk direction and constrained non-negative loading. | `src/pillar_pipeline.py`; monotonicity tests |
| 4 missing critical fields could improve scores | Closed. Critical-field missingness has a bounded, disclosed risk penalty. | `src/pillar_pipeline.py`; Score Drivers |
| 8, 10, 11 external-flow and balance-sheet coverage | Closed. Current reference data cover current-account receipts/payments, portfolio positions/flows, net IIP, and external liabilities, with source and period metadata. | `artifacts/external_liquidity_features_report.json` |
| 15 unstable sample-relative orientation | Closed. Production inference uses frozen training distributions and declared directions. | `src/pillar_pipeline.py`; inference sidecar |
| 16 GDP orientation | Closed by design. GDP is an ordinary direction-declared feature; it no longer decides the sign of a component. | `src/pillar_pipeline.py`; `MODEL_CARD.md` |
| 19 FDI/stable-financing coverage | Closed. FDI and stable-financing features now cover approximately 88-90% of model countries. | external-liquidity report |
| 20 commodity/terms-of-trade coverage | Closed. Commodity exposure and terms of trade cover 172/201 and 190/201 countries respectively. | external-liquidity report |
| Government-liquidity Explorer cache | Closed in this checkpoint. Every candidate refresh rebuilds compact government observations/features from the freshly retrieved WEO cache, packages them, and promotion installs them with the serving snapshot. | refresh and promotion workflows |
| Current-vs-candidate liquidity roles | Closed in this checkpoint. Methodology derives active roles from the promoted model instead of presenting active government inputs as challengers. | `app.py`; current artifact |
| Crisis-label source and candidate evaluation foundation (C1-C5) | Closed as engineering foundations, not promoted performance. Exact IMF WP/26/94 labels, an annual/event panel, purged expanding validation, forward calibration, monotonic candidates, confidence intervals, and ledgers are retained. The challenger failed its forward-time gate and therefore did not enter production. | Checkpoints 46-47; PR #14 |
| Legacy classifier diagnostics | Closed by containment. The app no longer presents threshold-selected legacy precision/recall/confusion metrics as clean validation. The existing bounded uplift is identified as provisional; any replacement must pass the forward-time gate. | Methodology/model-card guard in this checkpoint |

The model remains a country-level banking operating-environment screener. The
structural pillar score is the primary output. The bounded crisis uplift is not
a stand-alone crisis prediction, a rating, or an automatic decision rule.

#### 21.5.2 Application usability and reliability — closed

| Original rank/finding | Final disposition | Evidence |
|---|---|---|
| 22 explainability | Closed. Country Score Drivers show feature, pillar, raw value, contribution, imputation status, and dominant driver. | `app.py`; `src/scripts/explain_country_scores.py` |
| 23 corrupt/stale artifacts | Closed. Checksum validation and last-known-good fallback protect startup. | `src/model_store.py`; fallback tests |
| 24 health visibility | Closed with a utilitarian UX rule: users see a concise message only when serving is degraded; detailed health remains admin-only. | `app.py`; operations docs |
| 25-26 deploy verification | Closed. The live app was manually smoke-tested across Global, Country, Explorer, and Methodology and an automated endpoint check is installed. Duplicate rank 26 was removed. | live check on 2026-07-14; workflow |
| 27 public snapshot selector | Retired from the public UI. Snapshot selection is an operator diagnostic because only reviewed artifacts may be served; exposing arbitrary local snapshots would undermine the publication gate. | `SHOW_ADMIN_DIAGNOSTICS` |
| 28 stale Methodology | Closed. Model/data cards and source/model roles are rendered from current artifacts with concise limitations. | `app.py`; model/data cards |
| 29 responsive/accessibility QA | Closed for the supported baseline. Desktop and 390x844 mobile smoke checks passed on all primary tabs; the discovered clipped mobile brand row was fixed. Full WCAG certification was removed because it is a separate assurance engagement, not a release blocker for this research app. | browser check; CSS regression guard |
| 30-32 governance/ownership/rollback | Closed. Thresholds are approved; CODEOWNERS, release checklist, and rollback runbook are present. | `docs/GOVERNANCE.md`; `.github/CODEOWNERS`; `docs/RELEASE_CHECKLIST.md` |
| 35 observation-status preservation | Closed. Manifest/caches preserve reported, estimate, projection, carried-forward, and imputed status where supplied. | manifest builder and tests |
| 36 pickle portability/security | Closed under the approved policy: hashes, controlled repository provenance, and pinned training dependencies are required. Format migration remains optional technical debt, not an active incident. | governance; model-store checks |
| 37 dead/diagnostic code | Closed. Dead modules were removed and diagnostic scripts were archived. | `src/scripts/archive` |
| 38 product identity | Closed. BankEnv naming, page title, logo/favicon, and analyst-workbench positioning are aligned. | `README.md`; `app.py` |

Live verification on 2026-07-14 found the public
`https://bankenv.streamlit.app` serving the verified 2026-06-30 snapshot for
201 countries. No Streamlit exception appeared in any primary tab. A Country
warning and a Methodology monitoring notice were expected domain warnings, not
runtime exceptions.

#### 21.5.3 Automation and repository hygiene — closed

- Official IMF SDMX and World Bank retrievals run outside Streamlit.
- Weekly availability checks, monthly candidate builds, external-liquidity
  retrieval, smoke tests, and reviewed promotion are installed on `master`.
- Government-liquidity reference files now travel with the same candidate and
  promotion lifecycle as the other serving artifacts.
- Legacy root-level raw exports (approximately 442 MB) were removed from current
  `HEAD`. Compact caches remain for deterministic serving.
- Git-history rewriting was retired: it is destructive, invalidates existing
  clones, and provides no material benefit to the current Streamlit checkout.
- A separate source-version issue bot was retired as duplicative. Scheduled
  refreshes already retrieve the current official version and GitHub reports
  failed availability/refresh workflows.

#### 21.5.4 Removed from the active backlog, with reasons

| Removed item | Why it is not a current remediation issue | Re-entry gate |
|---|---|---|
| Rank 5 broad "more external sources" umbrella | Duplicated the specific BOP/IIP/WB/BIS items and encouraged source count over feature quality. | Add a source only for a named mechanism with provenance, coverage, and ablation evidence. |
| Exact general-government debt service and full GFN (ranks 6/13) | WEO supports interest/revenue and debt/revenue but does not contain principal amortization/rollover. Inventing "total GFN" from WEO would be misleading. | Verified IMF Fiscal Monitor/GFS or equivalent annual principal schedule with useful model-country/time coverage. |
| True external GEFN and usable-reserve/ST-debt adequacy (ranks 7/9) | Current fields are explicitly labelled proxies. Exact measures require short-term external debt, nonresident deposits, maturity schedules, and IRFCL usable reserves that are not consistently available in the current public pipeline. | Primary-source coverage sufficient for time-aware validation; keep proxy names until then. |
| Sovereign foreign-currency debt share (rank 12) | No consistent primary public series with adequate country/time coverage was found. | Primary IMF/WB field with provenance and minimum coverage gate. |
| Market-access spreads/yields/CDS/issuance (rank 14) | Useful but substantially licensed, market-frequency, and sovereign-credit focused; mixing sparse licensed data into the public banking-system model would reduce reproducibility. | Separately governed market overlay with redistribution rights and coverage tests. |
| External rating-agency replication benchmark (rank 18) | BankEnv is a banking-system operating-environment screener, not a sovereign/bank rating clone. Rating-category agreement would test a different target. | A predeclared benchmark against public crisis/stress outcomes or an approved ratings-mapping research question. |
| REER/equity/property stress (rank 21) | Current REER pull has zero usable coverage, equity is absent, and the BIS property adapter is opt-in research only. Empty/sparse fields must not be promoted merely to make the feature list longer. | Populate a primary-source cache, meet coverage/vintage gates, then pass incremental temporal validation. |
| Direct pre-FSI banking history (C3) | WB proxies and optional BIS series improve mechanism coverage but cannot create historical supervisory FSI observations that were never published. This is a source limitation, not unfinished app wiring. | Demonstrable new archival source with country/year coverage and no leakage. |
| Archived real-time vintages (C6) | The current backtest uses revised-vintage macro data and says so. Historical point-in-time archives are not available uniformly across sources. | Re-run only when archived vintages exist; future challengers must disclose revised-vintage bias. |
| Hierarchical hazard/regime/commodity challenger | It failed the 2014-2018 forward holdout. Tuning the holdout or exposing its probabilities would overfit and mislead users. | Independent forward regime with acceptable ROC-AUC, calibration, precision/recall, alert burden, and confidence intervals under frozen rules. |
| Public arbitrary challenger toggles | Unapproved scenario outputs could be mistaken for live model scores and did not cascade consistently. | Only saved, reviewed analytical overlays with clear "does not change live score" labeling. |
| Repository privacy (rank 34) | No licensed/private data is committed and Streamlit can serve the current public repository. Privacy is an account/product decision, not a code defect. | Revisit before introducing licensed data, secrets, or restricted model IP. |
| Destructive LFS history rewrite (rank 33 remainder) | Current-head cleanup captures the deployment benefit; rewriting history creates clone/commit risk. | Owner-approved repository migration with backups and coordinated clone replacement. |
| Conversational "copilot" layer | It is not needed for the current transparent analyst workbench and would add cost, privacy, and reliability surface without improving the validated score. | Separate product brief, security review, and evidence-grounded retrieval design. |

#### 21.5.5 Future research is governed, not silently open

The retained hazard/CV/BIS/evidence modules are reusable research foundations.
They do not create a live hazard score, confusion matrix, probability, or
commodity/liquidity cascade. A future candidate becomes an implementation issue
only after a dated research proposal identifies:

1. the target and horizon;
2. the new primary data and archived-vintage limitations;
3. nested country/time validation with frozen thresholds;
4. ROC-AUC, average precision, Brier/calibration, precision, recall, specificity,
   alert burden, event lead time, confidence intervals, and baseline ablations;
5. a forward regime not used for model or threshold selection; and
6. the app surface, rollback artifact, and named approval.

Failure of a candidate is a valid stop outcome. It must remain in the research
record and must not be converted into a production feature simply to close a
checklist.


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
  - Commit: `ba80402`.
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
  - Commit: `ba80402`.
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
  - Commit: `534efeb`.
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
  - Commit: `1d6835f`.
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
  - Commit: `b21e835`.
  - Scope:
    - Replaced README-rendered Methodology tab with a manifest-backed current
      methodology view tied to the active snapshot, model card, and data card.
    - Updated stale README release-status, source-count, and output-artifact
      claims to match the verified `2026-06-30` serving manifest.
  - Verification: compile check passed; unit suite passed; local Streamlit
    Methodology tab rendered current snapshot metadata without the old
    February 2026 legacy-artifact warning.
- [x] Checkpoint 10: Methodology active-source count correction.
  - Commit: `1d0aa3b`.
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
  - Commit: `3cae256`.
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
  - Commits: `217d263`, duplicate-ID hotfix `1d6eeae`.
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
  - Commit: `f27335f`.
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
  - Commit: `d89eea0`.
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
  - Commit: `d5d5492`.
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
  - Commit: `1c6e3da`.
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
  - Commit: `ed84d8d`.
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
  - Commit: `094f9b3`.
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
  - Commit: `e5b9803`.
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
  - Commit: `c229273`.
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
  - Commit: `3eb3d14`.
  - Scope:
    - Added a compact BankEnv brand row above the primary tabs using the same
      `BE` mark as the favicon.
    - Kept the layout utilitarian: no marketing headline and no extra
      explanatory copy.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8566.
- [x] Checkpoint 22: BankEnv logo refinement.
  - Commit: `cec6a66`.
  - Scope:
    - Replaced the plain `BE` tile with a compact bank/operating-environment
      signal mark.
    - Applied the same mark to the favicon and in-app brand row for visual
      consistency.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8567.
- [x] Checkpoint 23: Theme-aware BankEnv logo.
  - Commit: `084088a`.
  - Scope:
    - Converted the in-app BankEnv mark from fixed SVG colors to CSS variables.
    - Added dark-mode CSS overrides so the in-app tile inverts to a light mark
      on dark backgrounds while retaining the same motif.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8568.
- [x] Checkpoint 24: BankEnv logo readability pass.
  - Commit: `58c474d`.
  - Scope:
    - Replaced the house/temple-like mark with a larger operating-environment
      analytics mark: axes, bars, and trend line.
    - Increased in-app logo size to improve readability and reduce smudging.
    - Preserved dark-mode adaptation through CSS variables.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py`
    passed; full test suite passed (`79 passed`, `1 skipped`); local Streamlit
    startup check returned HTTP 200 on port 8569.
- [x] Checkpoint 25: App-wide staged external-liquidity visibility.
  - Commit: `dc4c7a0`.
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
    - At this checkpoint the active artifacts did not yet use the staged fields.
      Checkpoint 29 later promoted the curated liquidity subset and preserved the
      remaining fields as explicitly insight-only.
    - The Kenya/Mozambique mismatch observed here was resolved by the promoted
      artifact: the current active score is Mozambique 8.6 versus Kenya 8.0.
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
  - Superseded finding: the weak peer selector described at this checkpoint was
    replaced in Checkpoint 28 by the multidimensional peer engine; editable
    peer sets remain available for analyst judgment.
- [x] Checkpoint 27: Government-liquidity app-wide surfacing cleanup.
  - Commit: `28d82d3`.
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
  - Commit: `4443354`.
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
  - Commit: `fed03b8`.
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
  - Commit: `f64e786`.
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
  - Commit: `2f74854`.
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
  - Commit: `b2b40f1`.
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
      The final closure checkpoint wires this compact reference output into the
      candidate-refresh and promotion lifecycle.
  - Source limitation: full gross financing need needs debt amortization and
    rollover data that WEO does not carry. It is retired from active remediation
    under section 21.5.4 rather than approximated under a misleading name.
  - Verification: `python -m py_compile app.py src/government_liquidity.py
    src/scripts/build_government_liquidity_features.py` passed; full test suite
    passed (`84 passed`, `1 skipped`); real build against the resolved WEO cache
    covered 94-100% of model countries for the core affordability ratios
    (structural balance ~42%) with economically sensible values (e.g. Kenya
    interest/revenue ~30%, Egypt/Pakistan/Sri Lanka highest interest burden,
    Japan debt/revenue ~577%); local Streamlit startup returned HTTP 200 on
    port 8599 with no stderr.
- [x] Checkpoint 27: Market and external stress inputs (backlog ranks 19-21).
  - Commit: `b8c6ea3`.
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
      mean). Equity and property-price production integration was later retired
      under section 21.5.4 because the public cache did not meet coverage and
      vintage gates; the BIS property adapter remains opt-in research code.
    - Added human-readable labels and unit tests; the features flow through the
      existing staged External Liquidity panel automatically.
  - Historical model status: these began as staged challenger inputs. The
    curated liquidity subset was later promoted through Checkpoint 29; FDI,
    commodity exposure, and terms of trade are populated as insight features,
    while REER remains zero-coverage and was not promoted.
  - Current data status is recorded in
    `artifacts/external_liquidity_features_report.json`; this supersedes the
    pre-fetch null-column note from this checkpoint.
  - Verification: `python -m py_compile src/external_liquidity.py app.py`
    passed; full test suite passed (`87 passed`, `1 skipped`); build runs
    gracefully on the existing committed observations (new columns present,
    null until fetched).
- [x] Checkpoint 28: Liquidity-feature challenger (features fed into the model).
  - Commit: `c5137b1`.
  - Owner steer: "They should be fed into the model... tastefully." The staged
    external + government liquidity features were wired into the real pillar
    pipeline and evaluated as a governed challenger. Checkpoint 29 records the
    later owner-approved promotion and artifact publication.
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
  - Historical result at this checkpoint: promotion was not yet taken. Only the tier-change threshold (21 > 15) tripped on the
    isolated effect (mean 0.19 < 0.5, Spearman 0.989 > 0.90), so this is a much
    milder, more promotable change than the directional challenger, but it still
    required owner review under `docs/GOVERNANCE.md`; that review and the final
    promotion are recorded in Checkpoint 29.
  - Verification: `python -m py_compile` on the changed modules passed; full
    test suite passed (`90 passed`, `1 skipped`) including new challenger-logic
    unit tests.
- [x] Checkpoint 29: Liquidity challenger PROMOTED to the active model + durable
  wiring (owner: "proceed with both").
  - Commits: `2bf1762`, `876e4d7`; promotion PR #9 merged as `0d3d3ad`.
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
      Publication was completed through `refresh-data.yml` and
      `promote-snapshot.yml`; promotion PR #9 merged the generated LFS artifacts.
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
  - Commit: `5cf7905`.
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
- [x] Checkpoint 31: Fix stale serving after model republish (live-app bug).
  - Symptom: after promoting the liquidity model, the live app showed wrong
    Country Profile data (e.g. USA Score Drivers all zero, though the live model
    has USA critical-missing 12.5% / penalty +0.38) and stale peers.
  - Root cause: the cached loaders (`load_all_data`, `load_inference_pipeline`,
    `compute_country_drivers`) were keyed only on static values ("Active",
    country_code), with the model/pipeline excluded from the key (underscore
    args). Republishing the artifact did not bust them, so the app served the
    pre-update computation (the pre-existing `# timestamp: force_reload` comment
    was a manual workaround for exactly this). A stale on-disk LFS artifact
    could additionally fail checksum and degrade to an older archived bundle.
  - Fix: added `_serving_version()` (the manifest's `cache/risk_model.pkl`
    sha256, read fresh each run) and threaded it as a cache key into the three
    loaders, so any republish auto-busts them. Added a `force` re-download to
    `ensure_lfs_file`, and `load_model_artifact` now force-refreshes a
    checksum-mismatched artifact before degrading to a fallback bundle, so a
    stale resolved LFS file self-heals instead of serving an old archive.
  - Verified: build_driver_table on the live model gives USA 12.5% / +0.38, MOZ
    25% / +0.38, KEN 0% (full coverage) — the panel logic was already correct;
    the bug was purely stale caching/serving. Full suite 94 passed, 1 skipped;
    Streamlit boot HTTP 200.
- [x] Checkpoint 32: Country Profile fixes — peer state leak, reserves units,
  unified liquidity section.
  - Peer cross-contamination: the Country Profile peer multiselect used a static
    widget key (`custom_peer_codes`), so Streamlit persisted one country's
    selection across country changes (viewing the US then Kenya showed the US's
    peers — Germany/UK/France/Italy — instead of Kenya's nearest neighbours).
    Keyed the widget per country (`custom_peer_codes_{code}`) so it re-seeds.
  - Reserves units: `reserves_to_goods_services_imports` is a percent of annual
    imports; it now displays as months of imports (value x 12 / 100), e.g.
    Kenya 36.18% -> 4.3 months. The model still uses the underlying ratio.
  - Sectioning: replaced the two stray "External/Government Liquidity Inputs"
    tables with a single "Liquidity Inputs Used In Score" table carrying a
    Category column (External / Government).
  - Verified: compile OK, full suite 94 passed / 1 skipped, Streamlit boot 200.
- [x] Checkpoint 33: App organization and visual-density cleanup.
  - Scope: UX-only; no scoring, data retrieval, model artifact, or peer-selection
    logic was changed.
  - Changes:
    - Shortened top-level tab labels to reduce mobile header truncation
      (`Global`, `Country`, `Explorer`, `Methodology`).
    - Reworked Country Profile into clearer sections: country score summary,
      score components, country evidence, and peers.
    - Renamed user-facing pillar labels to clearer business language:
      `Operating Environment` and `Banking System`.
    - Collapsed Score Drivers by default so technical attribution is available
      without dominating the country page.
    - Grouped model inputs, governance indicators, and liquidity inputs under a
      single Country Evidence block with sub-tabs.
    - Cleaned Peer Countries table labels and caption text without changing the
      peer algorithm.
    - Added an Explorer Workspace header with concise task framing and shortened
      Explorer tool tabs (`Compare`, `Calculate`, `External liquidity`,
      `Government liquidity`).
    - Tightened spacing and tab padding in `src/dashboard/styles.py`, including
      mobile-specific tab and metric sizing.
  - Verification: `python -m py_compile app.py src/dashboard/styles.py` passed;
    `pytest tests/test_peer_selection.py tests/test_calculated_series.py
    tests/test_model_store.py -q` passed (13 passed); direct `import app`
    succeeded.
- [x] Checkpoint 34: Normalize liquidity as ordinary model/source features.
  - Scope: UX-only; no scoring, data retrieval, promoted artifacts, or peer
    selection logic was changed.
  - Issue: liquidity appeared in several places as if it were a separate app
    module: Country Evidence had both Model Inputs and a separate Liquidity tab,
    and Explorer had standalone External/Government liquidity tabs even though
    the same series were already available through the normal source selectors.
  - Changes:
    - Removed the separate Country Evidence Liquidity tab.
    - Folded active government and external liquidity fields into the normal
      Country Evidence -> Model inputs table.
    - Kept reserves coverage displayed as months of imports while preserving
      the underlying model value.
    - Removed standalone Explorer External liquidity and Government liquidity
      tabs; liquidity remains accessible through Compare and Calculate source
      choices.
    - Consolidated Methodology/Data Card liquidity coverage under one
      `Liquidity Feature Coverage` section with External/Government subtabs.
    - Aligned model-input terminology with the app's public labels:
      Operating Environment and Banking System.
  - Verification: `python -m py_compile app.py src/dashboard/components.py
    src/dashboard/styles.py` passed; `pytest tests/test_peer_selection.py
    tests/test_calculated_series.py tests/test_model_store.py -q` passed
    (13 passed); direct `import app` succeeded.
- [x] Checkpoint 35: Remove redundant model/source explanation surfaces.
  - Scope: UX-only; no scoring, data retrieval, model artifact, or peer logic
    changed.
  - Issue: Country Evidence showed normal model inputs plus a separate
    `Model Features & Weights` expander with overlapping feature lists and
    methodology text. Score Drivers separately displayed `Critical Fields
    Missing` and `Missingness Penalty` even though the penalty is derived from
    the imputed critical-field share. Explorer also showed raw single-country
    source tabs as a full section beside cross-country Compare/Calculate tools,
    and Methodology repeated sources/features before rendering Model/Data Cards.
  - Changes:
    - Removed the duplicate `Model Features & Weights` expander from Country
      Evidence. Country now shows the model inputs once; methodology/model-card
      content remains in the Methodology tab.
    - Combined critical missingness and penalty into one Score Drivers metric:
      `Critical Field Imputation`, with the imputed share as the value and the
      related score penalty as the delta.
    - Moved raw single-country source histories into a collapsed `Source
      history` expander and only renders the raw source tabs after the user
      enables loading.
    - Simplified Methodology to a concise overview plus `Model Card` and
      `Data Card`; removed duplicated top-level Active Sources / Feature Set /
      Validation sections that were repeated in the cards.
    - Renamed `Active Sources` to `Snapshot Sources` in the Data Card because
      that count reflects manifest-backed snapshot sources, not every packaged
      derived dataset.
  - Verification: `python -m py_compile app.py src/dashboard/components.py
    src/dashboard/styles.py` passed; `pytest tests/test_peer_selection.py
    tests/test_calculated_series.py tests/test_model_store.py -q` passed
    (13 passed); direct `import app` succeeded.
- [x] Checkpoint 36: Expose derived government-liquidity features in Explorer.
  - Scope: data presentation only; no scoring, model artifact, peer-selection,
    or upstream retrieval logic changed.
  - Issue: Data Explorer's government source used only raw WEO-style
    observations, so derived fiscal-liquidity features already packaged in
    `government_liquidity_features.parquet` did not appear in Compare/Calculate.
    This specifically hid `govt_interest_to_revenue` from Explorer even though
    it was present in model inputs and methodology coverage.
  - Change: `load_government_insight_data()` now appends derived snapshot
    observations from the packaged government feature table into the same
    normalized government source feed used by Explorer. This exposes
    `govt_interest_to_revenue`, `govt_debt_to_revenue`,
    `govt_overall_deficit_gdp`, and `govt_primary_deficit_gdp` through the
    normal source selector without adding a standalone liquidity UI section.
  - Verification: local source-feed check confirmed `govt_interest_to_revenue`
    appears for KEN, MOZ, and USA in the GOVT comparison source; `python -m
    py_compile app.py src/dashboard/components.py src/dashboard/styles.py`
    passed; `PYTHONPATH=. pytest tests/test_peer_selection.py
    tests/test_calculated_series.py tests/test_model_store.py -q` passed
    (13 passed).
- [x] Checkpoint 37: Convert Explorer government ratios from snapshot dots to
  historical calculated series.
  - Scope: Data Explorer presentation/calculation only; no scoring, model
    artifact, peer-selection, or upstream retrieval logic changed.
  - Issue: Checkpoint 36 exposed derived government-liquidity fields in the
    Explorer source selector by appending latest feature-snapshot rows. That
    made computed fields such as `govt_interest_to_revenue` visible, but their
    charts rendered as a single point with an odd x-axis instead of a proper
    time series.
  - Change: Explorer now derives government ratios from the raw WEO fiscal
    observation history for every aligned country/period. The derived series
    include `govt_interest_gdp`, `govt_interest_to_revenue`,
    `govt_debt_to_revenue`, `govt_overall_deficit_gdp`, and
    `govt_primary_deficit_gdp`.
  - Verification: local source-feed check confirmed `govt_interest_to_revenue`
    now has 6,194 annual observations across 190 countries from 1980-2025,
    including 44 annual KEN observations and 25 USA observations in the selected
    comparison source; `python -m py_compile app.py src/dashboard/components.py
    src/dashboard/styles.py` passed; `PYTHONPATH=. pytest
    tests/test_peer_selection.py tests/test_calculated_series.py
    tests/test_model_store.py -q` passed (13 passed).
- [x] Checkpoint 38: Wire candidate liquidity upgrades cross-app and add model
  monitoring.
  - Scope: data/model governance and app surfacing. Active serving scores are
    not silently changed by this checkpoint.
  - Issue: new official-source candidate fields needed to be available
    app-wide, evaluated through the model pipeline, and disclosed in
    Methodology without creating separate liquidity silos or promoting weak
    fields prematurely.
  - Changes:
    - Refreshed packaged external-liquidity reference artifacts from the
      current WB/IMF observation cache and corrected coverage denominators to
      the 201 scored countries.
    - Added monitored candidate model fields with explicit risk directions:
      `reserves_to_current_account_payments`, `portfolio_liabilities_gdp`,
      `commodity_export_share_pct`, `wb_total_external_debt_service_gni_pct`,
      `wb_ppg_external_debt_service_gdp`, and
      `wb_public_financing_need_ext_debt_service_proxy_gdp`.
    - Kept the active production feature list separate from candidate fields in
      `src/liquidity_features.py`; the candidate comparison path can include
      them without changing live scores.
    - Updated the liquidity challenger to compare active-liquidity retrain vs
      active-plus-candidate retrain, so score movement isolates the incremental
      candidate effect.
    - Added derived annual external series for Explorer where source history
      supports them, including commodity-export concentration and REER gap
      calculations.
    - Added Methodology / Model Card monitoring output for candidate
      score-movement results and packaged the crisis classifier validation
      summary plus confusion-matrix image.
  - Data status:
    - At least one external-liquidity feature covers 198/201 scored countries
      (98.5%).
    - Candidate coverage: `commodity_export_share_pct` 172/201 (85.6%),
      `reserves_to_current_account_payments` 166/201 (82.6%),
      `portfolio_liabilities_gdp` 159/201 (79.1%), and WB debt-service/public
      financing proxies about 119-120/201 (59.2-59.7%).
    - IMF FDI/REER fields remain zero coverage in the current packaged pull, so
      they remain insight/backlog fields until the external-source resolver
      returns usable observations.
  - Model-monitoring result:
    - Candidate effect vs active-liquidity retrain: mean absolute score movement
      0.716, 60 countries move by at least 1 point, 67 risk-tier changes,
      Spearman rank correlation 0.912.
    - This trips the configured owner-review gate; candidate fields should be
      reviewed before promotion into the active serving artifact.
  - Verification: `python -m py_compile app.py src/liquidity_features.py
    src/pillar_pipeline.py src/external_liquidity.py src/government_liquidity.py
    src/scripts/build_liquidity_challenger.py src/utils.py` passed;
    `PYTHONPATH=. pytest tests/test_external_liquidity_features.py
    tests/test_directional_scoring.py tests/test_peer_selection.py
    tests/test_calculated_series.py tests/test_liquidity_challenger.py -q`
    passed (25 passed).
- [x] Checkpoint 39: Balance liquidity challenger with government dynamics and
  improve Methodology surfacing.
  - Scope: challenger/data-card/country-evidence surfacing. Active production
    scores remain unchanged.
  - Issue: the first candidate challenger leaned too heavily toward external
    liquidity and commodity-export exposure. Government liquidity was already
    partly live (`govt_debt_gdp`, `fiscal_balance_gdp`,
    `govt_interest_to_revenue`, `govt_debt_to_revenue`), so the fix needed
    genuinely incremental fiscal-liquidity dynamics rather than duplicate
    levels.
  - Changes:
    - Added WEO-derived government candidate fields:
      `govt_revenue_gdp`, `govt_primary_deficit_gdp`,
      `govt_interest_to_revenue_change_3y`,
      `govt_debt_to_revenue_change_3y`,
      `govt_primary_deficit_gdp_change_3y`, and
      `govt_revenue_gdp_change_3y`.
    - Added government fiscal trend construction from historical WEO
      observations and regenerated the packaged government-liquidity reference
      files.
    - Wired the government candidates into the challenger-only feature assembly
      with explicit risk directions; default production assembly still includes
      only the approved active liquidity features.
    - Updated the challenger report to group candidates into government and
      external liquidity, so Methodology can show balance rather than a flat
      list.
    - Added a collapsed Country Profile `Additional candidate evidence`
      section. Score Drivers remain live-score-only.
    - Expanded Methodology / Model Card monitoring to show active vs candidate
      liquidity features, coverage, score movement, promotion status, and the
      crisis classifier validation/confusion-matrix section.
  - Data status:
    - New government candidate coverage: `govt_primary_deficit_gdp` 201/201
      (100.0%), `govt_revenue_gdp` 197/201 (98.0%),
      `govt_debt_to_revenue_change_3y` 193/201 (96.0%), and the other
      government change fields 190/201 (94.5%).
  - Model-monitoring result:
    - Balanced candidate effect vs active-liquidity retrain: mean absolute
      score movement 0.683, 63 countries move by at least 1 point, 59
      risk-tier changes, Spearman rank correlation 0.923.
    - Result: still monitoring-only. The balanced challenger is more complete
      and better surfaced, but movement remains too large for silent production
      promotion.
  - Verification: `python -m py_compile app.py src/government_liquidity.py
    src/liquidity_features.py src/pillar_pipeline.py
    src/scripts/build_liquidity_challenger.py src/utils.py` passed;
    `PYTHONPATH=. pytest tests/test_government_liquidity_features.py
    tests/test_external_liquidity_features.py tests/test_directional_scoring.py
    tests/test_peer_selection.py tests/test_calculated_series.py
    tests/test_liquidity_challenger.py -q` passed (31 passed); direct
    `import app` succeeded in bare mode.
- [x] Checkpoint 40: Add challenger overlay and dominant live driver context.
  - Scope: app surfacing only. Active production scores and model artifacts are
    unchanged.
  - Issue: users need a way to inspect challenger effects without confusing
    them with the live score, and peer comparison needed a compact explanation
    of what primarily drives each displayed country.
  - Changes:
    - Added an optional `Show liquidity challenger overlay` toggle in the
      Country Profile peer table. When enabled, it displays the saved
      monitoring-only challenger score and delta beside the live score.
    - Added `Dominant Driver` to the peer table using live score attribution
      only, so selected country and peers show the largest live feature
      contribution without mixing candidate fields into Score Drivers.
    - Added Methodology text clarifying that the challenger overlay is
      analytical only and does not change live scores, rankings, or score
      drivers.
  - Verification: `python -m py_compile app.py` passed; `PYTHONPATH=. pytest
    tests/test_peer_selection.py tests/test_calculated_series.py
    tests/test_government_liquidity_features.py tests/test_liquidity_challenger.py
    -q` passed (18 passed); direct `import app` succeeded in bare mode.
- [x] Checkpoint 41: Move liquidity challenger overlay to country-level state.
  - Scope: app UX/state only. Active production scores and model artifacts are
    unchanged.
  - Issue: the challenger overlay was initially toggled only inside the peer
    table, even though the overlay is a country-level analytical lens that
    should affect the selected country header, candidate evidence, and peer
    comparison consistently.
  - Changes:
    - Moved `Show liquidity challenger overlay` to the top of the Country tab.
    - When enabled, the selected country header shows the challenger score and
      delta versus the live score.
    - The same country-level toggle controls peer-table challenger columns and
      expands the `Additional candidate evidence` section.
    - Removed the duplicate peer-level toggle and renamed the peer delta column
      to plain `Delta Challenger` for cleaner rendering.
  - Verification: `python -m py_compile app.py` passed; `PYTHONPATH=. pytest
    tests/test_peer_selection.py tests/test_calculated_series.py -q` passed
    (9 passed); direct `import app` succeeded in bare mode.
- [x] Checkpoint 42: Split liquidity and commodity overlays into independent
  analytical scenarios.
  - Scope: app surfacing and challenger-monitoring artifacts only. Active
    production scores, live rankings, and live score drivers remain unchanged.
  - Issue: commodity-export concentration was bundled into the liquidity
    challenger even though it is an external vulnerability factor, not a
    liquidity metric. This made the overlay concept too broad and could confuse
    users reviewing government/external liquidity evidence.
  - Changes:
    - Split candidate groups into `government_liquidity`,
      `external_liquidity`, and `external_vulnerability`.
    - Rebuilt three saved candidate scenarios for the same 2026-06-30 cutoff:
      liquidity-only, commodity-only, and combined.
    - Replaced the single Country Profile overlay with independent `Liquidity
      overlay` and `Commodity overlay` toggles. Turning on both shows the saved
      combined scenario; turning on one shows only that scenario.
    - Filtered country-level candidate evidence to match the selected overlay
      groups, with commodity exposure labelled as external vulnerability rather
      than liquidity.
    - Updated Methodology language so overlays are described as independent,
      monitoring-only analytical lenses.
  - Model-monitoring result:
    - Liquidity-only effect vs active retrain: mean absolute score movement
      0.170, 3 countries move by at least 1 point, 19 risk-tier changes,
      Spearman rank correlation 0.992.
    - Commodity-only effect vs active retrain: mean absolute score movement
      0.717, 63 countries move by at least 1 point, 60 risk-tier changes,
      Spearman rank correlation 0.917.
    - Combined effect vs active retrain: mean absolute score movement 0.683,
      63 countries move by at least 1 point, 59 risk-tier changes, Spearman
      rank correlation 0.923.
    - Result: all scenarios remain monitoring-only. Commodity exposure is the
      large-moving factor and should not be silently promoted into the live
      production model.
  - Verification: `python -m py_compile app.py
    src/scripts/build_liquidity_challenger.py` passed; `PYTHONPATH=. pytest
    tests/test_peer_selection.py tests/test_calculated_series.py
    tests/test_liquidity_challenger.py -q` passed (12 passed); Streamlit
    AppTest passed with zero exceptions for base, liquidity-only,
    commodity-only, and combined overlay states.
- [x] Checkpoint 43 (historical result, invalidated by Checkpoint 45): Improve crisis-classifier ROC-AUC/precision and package
  honest validation.
  - Scope: crisis-classifier training path, validation reporting, and refreshed
    2026-06-30 model artifacts. The pillar framework and candidate overlay
    governance are unchanged.
  - Issue: packaged crisis validation showed ROC-AUC 0.616 and precision 0.10.
    The classifier was too recall-oriented to be interpreted as a usable model
    signal. Local training also revealed that SMOTE was unavailable because the
    environment had an incompatible `imbalanced-learn` version.
  - Changes:
    - Upgraded the local training environment to the repo-pinned
      `imbalanced-learn==0.14.0`; the code now reports the actual SMOTE import
      error if dependency resolution breaks again.
    - Added year-matched crisis-recency memory
      (`years_since_banking_crisis`) to the crisis panel.
    - Added year-matched fiscal/external deterioration features:
      `govt_revenue_gdp`, `govt_revenue_gdp_change_3y`,
      `fiscal_balance_change_3y`, `primary_balance_change_3y`,
      `govt_debt_to_revenue_change_3y`,
      `govt_interest_to_revenue_change_3y`, `ca_deficit_severity`, and
      `ca_deficit_widening_3y`.
    - Switched the crisis classifier to a regularized, class-weighted logistic
      rare-event model after controlled grouped validation showed the simpler
      model gave better ROC-AUC/precision than the XGBoost path on this sparse
      crisis panel.
    - Replaced the fixed 0.5 evaluation threshold with a validation-derived
      operating threshold and threshold diagnostics, so precision/recall are
      measured at a realistic rare-event threshold.
    - Refreshed `artifacts/crisis_validation_summary.json` and the packaged
      confusion-matrix image for the Methodology tab.
  - Validation result:
    - Previous packaged validation: ROC-AUC 0.616, precision 0.10, recall 0.83.
    - New deployment grouped CV ROC-AUC: 0.655.
    - New unseen-country holdout: ROC-AUC 0.683, precision 0.235, recall 0.343,
      F1 0.279 at threshold 0.130.
    - New out-of-time 2018 holdout: ROC-AUC 0.646, precision 0.081, recall
      0.625, F1 0.143 at threshold 0.077.
    - Score movement versus the previous committed active artifact is controlled:
      mean absolute score delta 0.175, Spearman rank correlation 0.991, and
      6/201 countries move by at least 1 point.
  - Status: improved, but not solved. The classifier is now more usable as a
    weak-to-moderate early-warning overlay, not as a stand-alone crisis
    decision model. Further gains likely require better banking-system stress,
    deposit/funding, market-access, and liquidity time series rather than only
    estimator tuning.
  - Verification: `python -m py_compile src/crisis_classifier.py` passed;
    `PYTHONPATH=. pytest tests/test_crisis_classifier.py
    tests/test_crisis_labels.py -q` passed (6 passed); local 2026-06-30
    retrain with `--retrain-classifier` passed snapshot validation (3 passed,
    0 failed).
- [x] Checkpoint 44 (historical result, invalidated by Checkpoint 45): Recover crisis-classifier recall with a recall-constrained
  operating threshold.
  - Scope: threshold policy and validation reporting only. The trained
    classifier scores/probabilities are unchanged from Checkpoint 43.
  - Issue: the max-F1 operating threshold improved precision but lost too much
    recall for an early-warning overlay.
  - Change: added explicit threshold policies:
    - `balanced`: maximize F1.
    - `review`: maximize precision subject to recall >= 0.60.
    - `high_recall`: maximize precision subject to recall >= 0.70.
    The default evaluation policy is now `review`.
  - Validation result:
    - Unseen-country ROC-AUC remains 0.683.
    - Balanced policy: precision 0.235, recall 0.343, F1 0.279, 51 flagged.
    - Review policy: precision 0.145, recall 0.657, F1 0.237, 159 flagged.
    - High-recall policy: precision 0.127, recall 0.743, F1 0.217, 205
      flagged.
    - Review-policy confusion matrix: TN 280, FP 136, FN 12, TP 23.
  - Superseded interpretation: these threshold metrics were selected and
    reported on the same holdout. Checkpoint 45 invalidated them as clean
    external-test evidence.
  - Verification: local retrain confirmed no score/probability/uplift movement
    versus Checkpoint 43; `python -m py_compile src/crisis_classifier.py`
    passed; `PYTHONPATH=. pytest tests/test_crisis_classifier.py -q` passed
    (5 passed).
- [x] Checkpoint 45: Complete the deep crisis-classifier reliability diagnosis.
  - Scope: evidence audit only; no model probability, country score, threshold,
    or live-app behavior changed.
  - Result: promotion/expansion was blocked by six P0 findings: incomplete
    official labels, holdout threshold leakage,
    sparse historical banking inputs, unconstrained logistic coefficient signs,
    overlapping/underpowered panel design, and revised-vintage temporal bias.
  - Evidence: repository labels encode 103 systemic episodes plus 3 optional
    borderline episodes versus 161 systemic plus 3 borderline episodes in IMF
    WP/26/94; FSIC coverage is zero for all six classifier banking variables in
    the first six feature epochs and only 2-6 countries in 2004; direct model
    inspection confirmed material coefficient-direction reversals; code review
    confirmed threshold selection and reporting occur on the same holdout.
  - Interpretation correction: ROC-AUC 0.655 grouped CV / 0.683 single grouped
    holdout / 0.646 revised-data temporal holdout remain provisional ranking
    evidence. Published precision, recall, F1, alert count, and confusion matrix
    are not clean external-test results until the threshold is frozen upstream.
  - Subsequent disposition: Checkpoint 46 implemented the safe evaluation
    foundation and Checkpoint 47 retained it after the challenger failed the
    forward-time gate.
- [x] Checkpoint 46: Build the leakage-free crisis-model evaluation foundation
  and stop at the temporal reliability gate.
  - Added the exact official IMF WP/26/94 episode artifact and reproducible PDF
    extraction path, an auditable annual/event panel, long-history World Bank
    financial and vulnerability inputs, governed feature families, monotonic
    estimator candidates, nested grouped/forward validation, cross-fitted
    calibration, frozen inner-fold thresholds, bootstrap confidence intervals,
    and out-of-fold ledgers.
  - Grouped challengers reached approximately 0.72 ROC-AUC with 0.63 recall,
    but failed the pre-declared 2014-2018 forward holdout (0.37-0.56 ROC-AUC
    across tested models and unusable alert burden). The failure is recorded as
    evidence, not tuned away.
  - Production model artifacts, live scores, and overlays remained unchanged.
    Promotion stopped; section 21.5.5 records the evidence required before any
    future research candidate may re-enter implementation.
  - Verification: focused crisis-model/source suite passed (26 tests after the
    derived-source metadata correction); repository-wide suite passed (120
    tests, 1 skipped). GitHub CI status is recorded with the checkpoint push.
- [x] Checkpoint 47: Retain the safe early-warning foundation without deploying
  the failed hierarchical challenger.
  - Decision: PR #13 was preserved as the complete audit record while the
    mergeable infrastructure was rebuilt on `agent/hazard-foundation` from
    clean `master`. The mixed implementation commit was not cherry-picked.
  - Included: corrected regularization scaling for future retraining;
    horizon-embargoed expanding CV with event purge and forward-only
    calibration; opt-in BIS credit, gap, debt-service and property adapters;
    full-taxonomy descriptive mechanism coverage; and a defensive optional
    artifact parser.
  - Excluded: generated hazard/validation JSON, the snapshot builder,
    provisional alert thresholds, dashboard components, and all Streamlit
    wiring. Current production artifacts, scores, rankings, tabs, and live
    behavior remain unchanged.
  - [x] Verification/publish: 49 focused
    tests and 163 repository-wide tests passed, one repository test was skipped,
    every serving-artifact smoke check passed for the unchanged 201-country
    snapshot, and an explicit diff guard confirmed no app, dashboard, serving
    artifact, generated hazard artifact, or snapshot-builder change.
  - Foundation review: an independent final audit found and fixed two
    defensive-parser edge cases: duplicate country rows now invalidate the
    optional frame, and records with missing country codes can no longer become
    the literal code `NAN`. The exact post-fix branch passed GitHub `Quality
    checks / test` and PR #14 merged to `master` as `a1fb94d`.
  - Handover: PR #13 was closed as superseded after the foundation merge. Its
    remote `codex/hierarchical-risk-architecture` branch remains available as
    the rejected-model evidence and research record; no failed probabilities or
    alert surfaces reached production.
- [x] Checkpoint 48: Reconcile and close the comprehensive remediation plan.
  - Implementation commit: `8cde277` (`agent/close-remediation-plan`).
  - Re-audited every former open/partial marker against code, artifacts, tests,
    workflows, GitHub state, and the live application. Section 21.5 now records
    one current disposition for every issue; stale duplicate backlog tables and
    all 31 false `Pending commit` labels were reconciled.
  - Fixed the remaining relevant trust issues: invalid legacy crisis precision,
    recall, and confusion metrics now fail closed; Methodology derives liquidity
    roles from active loadings; current cards distinguish the served legacy
    overlay from the exact WP/26/94 research label foundation.
  - Fixed the remaining relevant data/release issues: candidate refresh and
    promotion now rebuild, checksum, package, and install the compact government-
    liquidity Explorer references from the same WEO snapshot; the live endpoint
    has a bounded scheduled reachability check.
  - Fixed the mobile brand clipping found during the final 390x844 browser pass.
    Global, Country, Explorer, and Methodology rendered without a Streamlit
    exception; the 201-country snapshot remained `2026-06-30 | verified`.
  - Removed approximately 442 MB of unreferenced root raw exports from current
    `HEAD` and ignored their download naming pattern. Historical rewriting was
    deliberately rejected; compact serving caches remain intact.
  - No risk-model, classifier, or inference-pipeline binary was rebuilt. Active
    scores remain USA 7.0, Kenya 8.0, and Mozambique 8.6.
  - CI portability follow-up: checksummed JSON artifacts are pinned to LF so
    manifest byte counts and hashes are identical on Windows and Linux.
  - Verification: 183 tests passed and one was skipped; serving-artifact smoke
    tests passed all checks; focused live-probe, workflow-contract, manifest-
    checksum, AppTest, compile, and diff checks passed.
