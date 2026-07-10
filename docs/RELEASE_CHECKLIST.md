# Release / Promotion Checklist

Copy this checklist into the promotion pull request description and check
every item. A promotion PR that changes serving artifacts must not merge with
unchecked items (encoded via CODEOWNERS review; see docs/GOVERNANCE.md for
thresholds).

## Candidate quality

- [ ] `python -m pytest -q` passes in a clean environment
- [ ] `python -m src.scripts.smoke_test_artifacts` passes on the candidate bundle
- [ ] Snapshot manifest present: snapshot ID, cutoff, source versions, checksums
- [ ] Model validation results recorded (`validation` block in the manifest)
- [ ] No post-cutoff observation in the snapshot (cutoff_verified true)

## Change review

- [ ] Update report reviewed: added/removed countries, coverage changes, source revisions
- [ ] Score changes within GOVERNANCE.md section 3 thresholds, OR
- [ ] Comparison report (`artifacts/challenger_comparison.json` format) reviewed and this box initialed by the release approver
- [ ] Largest country movements individually explained (driver table: `python -m src.scripts.explain_country_scores <CODES>`)
- [ ] Model card / data card updated if the scoring method or source set changed

## Deployment

- [ ] Previous approved bundle archived under `artifacts/snapshots/` (rollback target)
- [ ] After merge: live app shows the new snapshot ID and a healthy System Health badge
- [ ] Rollback procedure confirmed executable (docs/OPERATIONS_RUNBOOK.md section "Rollback")

## Sign-off

- [ ] Release approver: ______ (GitHub handle, date)
