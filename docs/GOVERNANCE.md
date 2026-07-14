# Data and Model Governance Policy

Status: **APPROVED (@MMJGGR, 2026-07-10)** — thresholds and snapshot
lifecycle approved as proposed (backlog items 30 and 31 closed). The numeric
thresholds below are enforced in code where noted; revisions require a
reviewed pull request updating this document and the companion constants.

## 1. Source freshness SLAs (enforced: `src/health.py`, System Health panel)

Maximum acceptable age of each source's latest observation at serving time.
Derived from the section 9.4 cadence table of the remediation plan plus
typical publication lag.

| Source | Cadence expectation | SLA (days) |
|---|---|---:|
| WEO | Two vintages per year (Apr/Oct), annual data | 420 |
| FSIC | Rolling monthly/quarterly with country lag | 240 |
| MFS | Rolling monthly with country lag | 240 |
| FSIBSIS | Quarterly/annual balance-sheet reporting | 365 |
| WGI | Annual release, ~18-month reference lag | 900 |
| Snapshot manifest itself | Quarterly refresh expectation | 190 |

Breach behavior: the app shows a stale badge and per-source warning; candidate
refreshes must record the breach in the update report. A breach never causes
silent replacement of the last validated snapshot.

## 2. Coverage and imputation gates (candidate publication)

| Gate | Threshold | Enforced |
|---|---:|---|
| Countries scored | >= 150 | `smoke_test_artifacts.py` |
| Countries with display names | >= 90% | `smoke_test_artifacts.py` |
| Direct (non-imputed) share of critical banking fields, per country | < 50% missing triggers critical-field penalty; penalty max +1.5 | `PillarInferencePipeline` (per-country, disclosed in score) |
| Coverage bias: correlation of data coverage with the pre-policy model score | abs(corr) < 0.4 flags for review (does not block candidate upload) | `validate_model` in `train_model.py` |
| Row/country/indicator counts vs previous snapshot | decline > 10% blocks publication | manual review via candidate report (automation pending) |

Approved treatment of the coverage-bias gate: retain 0.4 as a standing review
flag, not an automatic publication block. Measure it on the pre-policy score
(before the critical-field penalty and crisis uplift, which correlate with
coverage by design) and report both the raw correlation and the development-
level-controlled diagnostic when available. Changing the threshold or making
it blocking requires a reviewed policy update.

## 3. Score-change review thresholds (promotion)

A candidate that differs from the serving artifact beyond any of these bounds
requires explicit owner review of the comparison report before promotion:

| Measure | Review threshold |
|---|---:|
| Mean absolute score change | > 0.5 |
| Countries moving >= 1.0 points | > 20 |
| Spearman rank correlation vs production | < 0.90 |
| Any country moving >= 3.0 points | any |
| Risk-tier changes (5-tier scale) | > 15 countries |

The comparison report format is `artifacts/challenger_comparison.json`
(production vs challenger scores, largest movements, acceptance cases, and
open concerns). Archived 2026-07-10 directional/liquidity challenger reports
are historical experiments and do not define the role of fields in the newer
served artifact. Current feature roles must be read from the serving
artifact's persisted loading maps; a new candidate requires a comparison
against that exact serving baseline.

## 4. Snapshot lifecycle and naming (backlog item 31)

Snapshot identifiers are cutoff dates (`YYYY-MM-DD`), optionally suffixed with
a variant tag (`-official-api`, `-challenger-<name>`).

| State | Meaning | Transition rule |
|---|---|---|
| `candidate` | Built by refresh workflow or locally; uploaded for review; never served | Passes smoke tests to become `verified` |
| `verified` | Manifest checksums match artifacts; validation recorded | Owner review of the update/comparison report to become `approved` |
| `approved` | Owner-reviewed; serves as the active snapshot | Superseded by the next `approved`; retained in `artifacts/snapshots/` |
| `provisional` | Mid-quarter snapshot on incomplete source reporting | Replaced in place by `final` when the reporting window closes; both vintages retained |
| `final` | Post reporting-lag rebuild of a provisional cutoff | Immutable; restatement requires a new suffixed snapshot |
| `restated` | Re-scored history under a newer model version | Must carry the model version in the variant tag; never silently replaces the original series |
| `fallback` | Automatic serving state, not a stored status: the app is serving the newest archived bundle because the active artifact failed | Cleared by restoring a valid active artifact |

Rules:

- Cutoffs follow the section 7.1 cadence: YE (Dec 31), Q1 (Mar 31), H1 (Jun
  30), Q3 (Sep 30) plus optional month-end monitors, all provisional at first
  publication except YE after the agreed reporting lag.
- Every archived bundle under `artifacts/snapshots/` must contain the
  complete serveable set (model, pipeline, classifier, features, imputed
  sidecar, manifest) so any bundle can serve as a rollback target.
- A merged PR replacing the active artifacts must reference the snapshot ID,
  its validation results, and (for model changes) the comparison report.

Current-state disclosure: the serving 2026-06-30 manifest is `verified`, but
the repository contains no recorded transition of that artifact to
`approved`. It is a controlled legacy serving exception, not evidence that
owner approval or external validation occurred. The next artifact-changing
promotion must close that lifecycle gap rather than carrying it forward.

## 5. Artifact portability policy (backlog item 36)

- Serving and training artifacts are pickles pinned to Python 3.11 and the
  exact library versions in `constraints-dev.txt`; CI installs under those
  constraints and `src/config.py` refuses interpreters older than 3.10.
- Any dependency upgrade that changes scikit-learn, XGBoost, or pandas major
  or minor versions requires rebuilding and re-verifying all serving
  artifacts in the same PR (checksums in the manifest make a stale artifact
  fail loudly rather than load quietly).
- Safer serialization (skops for the sklearn pipeline, native XGBoost JSON
  for the booster) is the target for a future release; until then artifacts
  must only be loaded from checksum-verified, repository-controlled bundles,
  which both the active loader and the fallback chain enforce.

## 6. Model-change classification

| Change | Examples | Requirement |
|---|---|---|
| Data refresh, same pipeline | New source vintage scored with fixed transforms | Verified snapshot + update report |
| Policy parameter change | Penalty size, floor levels, SLA values | This document updated + policy audit rerun |
| Scoring redesign | Direction constraints, overlay formula, feature add/drop | Full challenger comparison + owner approval + model card update |
| Classifier retrain | New epochs or labels | Exact WP/26/94 reconciliation + leakage-safe grouped/forward validation + owner approval |

## 7. Early-warning research foundation

The repository contains the exact IMF WP/26/94 Appendix I, Table A1 artifact
(161 systemic episodes and three explicitly borderline episodes), dormant
development utilities for annual discrete-time crisis hazards, descriptive
mechanism evidence, optional BIS history, and expanding temporal cross-
validation. They do not alter serving scores. The served classifier predates
the exact label artifact, and its legacy validation summary is marked
`invalid_superseded` because the earlier label implementation was incomplete
and operating-threshold selection contaminated the reported holdout metrics.

Replacement research challengers failed the pre-declared 2014-2018 forward
holdout and were not promoted. Their grouped performance cannot override that
failure. Any future model built from this foundation remains a research
challenger until all of the following are satisfied:

- Candidate models and hyperparameters are preregistered and compared with
  expanding outcome-year folds. A horizon can enter training only when its
  label becomes observable before the next validation block.
- Event identifiers are purged across train/validation boundaries, and any
  probability calibration applied to an out-of-fold prediction uses earlier
  folds only.
- A final confirmation period is untouched by feature selection, model
  selection, calibration, and threshold setting. The already-inspected
  2014-2022 results cannot be relabelled as a pristine future test.
- ROC-AUC, average precision and lift over prevalence, Brier score,
  calibration, precision, recall, specificity, false alerts per 100
  country-years, unique-event recall, and alert burden are reported.
- Alert thresholds and analyst-capacity limits are frozen before confirmation.
  A tier that flags nearly every country is disabled even if recall is high.
- Historical-core and modern-full experts are evaluated separately and either
  cross-calibrated or replaced by a demonstrably better shared model before a
  common operational threshold is used.
- Mechanism evidence coverage uses the complete governed signal taxonomy as
  its denominator. Observed-among-supported coverage can be diagnostic but
  cannot be presented as overall evidence completeness.
- No generated probability artifact, alert policy, or research dashboard is
  merged into serving code until the statistical gate passes and the owner
  approves the model change.
- The Methodology tab must fail closed: classifier metrics and a confusion
  matrix may be displayed as current validation evidence only when the report
  explicitly carries `validation_status: validated_clean`,
  `clean_validation: true`, and `display_metrics: true`. Invalid, superseded,
  research-only, or unlabelled reports remain available for audit but are not
  presented as valid model performance.
- Image evidence is independently bound to its report. A confusion matrix is
  displayed only when a clean validation-report `schema_version: 1` includes a
  `confusion_matrix_artifact` object with `schema_version: 1`, a repository-
  relative `path`, and the exact `sha256`; the resolved path must remain inside
  the repository and its bytes must match. Failure of this image gate does not
  suppress otherwise clean metrics, but no static or stale image is used.
