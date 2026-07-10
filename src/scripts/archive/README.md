# Archived diagnostic scripts

One-off investigation and debugging scripts retained for reference
(remediation plan backlog item 37, triaged 2026-07-10). None of these are
referenced by the application, tests, workflows, or documentation, and they
are not maintained: they may assume old column names, old cache layouts, or
the legacy pillar implementation.

Supported operational scripts live one level up in `src/scripts/`:
`refresh_data`, `check_sources`, `build_local_snapshot`,
`build_data_manifest`, `audit_model_policy`, `smoke_test_artifacts`,
`explain_country_scores`, and `verify_crisis_labels`.
