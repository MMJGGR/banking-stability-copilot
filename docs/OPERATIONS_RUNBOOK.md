# Banking Stability Copilot Operations Runbook

## Local Verification

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py train_model.py src tests
python -m pytest -q
python -m src.scripts.audit_model_policy
streamlit run app.py
```

## Check Official Source Availability

```bash
python -m src.scripts.check_sources
```

This checks the official IMF SDMX dataflows and the World Bank WGI API by
default. Use `--mode legacy --require-configured` only when testing the older
configured-URL/local-fallback adapter path.

## Build a Candidate Snapshot

The preferred route is the GitHub Actions workflow:

```text
Actions → Build candidate data snapshot → Run workflow
```

Supply the explicit cutoff, for example `2025-12-31`.

For a local build:

```bash
python -m src.scripts.refresh_data --as-of 2026-06-30
```

This command fetches official IMF SDMX and World Bank source data, normalizes
the app caches, scores through the model pipeline to the cutoff, runs model
validation, saves artifacts, and creates the serving manifest. It reuses the
existing validated crisis-classifier artifact by default; pass
`--retrain-classifier` only when a full supervised classifier refresh is
intended.

If a download succeeds but a downstream normalization/model step needs to be
rerun, reuse the retained raw files:

```bash
python -m src.scripts.refresh_data --as-of 2026-06-30 \
  --download-dir data/raw/<official_refresh_dir> \
  --reuse-downloads
```

When the current Parquet caches are the approved source input and no official
refresh is intended, build and archive a local cached-source snapshot instead:

```bash
python -m src.scripts.build_local_snapshot --as-of 2025-12-31
python -m src.scripts.build_local_snapshot --as-of 2026-06-30
```

By default this reuses the existing validated crisis-classifier artifact and
rebuilds cutoff-aware feature, pillar, manifest, and serving artifacts. Pass
`--retrain-classifier` only when a full classifier refresh is intended and the
longer runtime is acceptable.

## Candidate Review

Before promotion, verify:

- Workflow tests passed.
- Every source has an expected version and cutoff.
- Manifest status is `verified`.
- No country appears across grouped train/holdout sets.
- Coverage and imputation did not deteriorate materially.
- The model-policy audit was reviewed and material sensitivity changes were
  accepted or remediated.
- Large score and tier changes have explanations.
- Model and data cards are updated.
- Streamlit loads the candidate artifacts successfully.

## Promotion

Promotion is workflow-assisted; the final merge stays with a human reviewer.

1. Run `Actions -> Promote candidate snapshot`, supplying the candidate
   build's run ID and snapshot date. The workflow downloads the bundle,
   re-runs `python -m src.scripts.smoke_test_artifacts` against it (country
   names present, scores in range, raw/imputed sidecar coherence, verified
   manifest), commits the artifacts via Git LFS to `promote/<date>`, and
   opens a pull request.
2. Review the model-policy audit, feature-heuristic flags, and large score
   or tier changes on the pull request.
3. Merge the pull request; Streamlit Cloud redeploys from master.
4. Verify Streamlit health and the visible snapshot status.
5. Record the approver and release identifier.

Manual fallback: download the bundle, run the smoke test locally, and
replace the serving artifacts through a reviewed pull request (requires
git-lfs).

## Rollback

1. Restore the previous approved artifact bundle or release pointer.
2. Restore its matching `data_manifest.json`.
3. Redeploy or reboot Streamlit.
4. Verify the displayed snapshot ID and checksum.
5. Record the incident and affected release.

## Degraded Operation

When a source is unavailable:

- Do not replace the last validated source snapshot.
- Mark freshness SLA breaches in the candidate report.
- Keep the live application on the previous approved snapshot.
- Escalate only if the source remains unavailable beyond its expected cadence.

## Secrets

Required secret names are documented in `.env.example`.

- Never commit credentials or signed private URLs.
- Prefer read-only source credentials.
- Rotate credentials after exposure or personnel changes.
- Limit GitHub Actions permissions to read access unless a later approved
  promotion workflow needs more.

## Incident Categories

| Severity | Example | Response |
|---|---|---|
| Critical | Wrong cutoff, corrupted artifact, application unavailable | Roll back immediately |
| High | Material score anomaly, source schema break | Block promotion and investigate |
| Medium | One source stale with valid fallback | Continue degraded and monitor |
| Low | Non-blocking chart or documentation issue | Schedule normal correction |
