# Data and Model Governance Policy

Status: **PROPOSED — pending owner approval** (backlog items 30 and 31 in the
remediation plan). The numeric thresholds below are enforced in code where
noted so they are exercised from day one; the owner can revise any value and
the enforcement points pick the change up from this single source of truth's
companion constants.

Owner sign-off is recorded by changing this Status line to `APPROVED
(name, date)` in a reviewed pull request.

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

Note on the coverage-bias gate: under the directionally constrained pillar
model this correlation is ~0.5 because sparse-data countries also have
observably weaker governance and macro fundamentals. The gate is measured on
the pre-policy score (before the critical-field penalty and crisis uplift,
which correlate with coverage by design). Owner decision required: keep 0.4
and treat as a standing review flag, or re-baseline to a partial correlation
controlling for development level.

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
(production vs challenger scores, largest movements, acceptance cases, open
concerns). The 2026-07-10 directional-constraint challenger exceeds all of
these bounds and is therefore archived as
`artifacts/snapshots/2026-06-30-challenger-directional` awaiting review, not
promoted.

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

## 5. Model-change classification

| Change | Examples | Requirement |
|---|---|---|
| Data refresh, same pipeline | New source vintage scored with fixed transforms | Verified snapshot + update report |
| Policy parameter change | Penalty size, floor levels, SLA values | This document updated + policy audit rerun |
| Scoring redesign | Direction constraints, overlay formula, feature add/drop | Full challenger comparison + owner approval + model card update |
| Classifier retrain | New epochs or labels | Grouped + out-of-time validation report + owner approval |
