# Banking Stability Copilot Operations Runbook

## Local Verification

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py train_model.py src tests
python -m pytest -q
streamlit run app.py
```

## Check Source Configuration

Copy `.env.example` to a local untracked `.env` or configure the equivalent
GitHub Actions secrets.

```bash
python -m src.scripts.check_sources
```

Use `--require-configured` in controlled environments where every official
source must be configured.

## Build a Candidate Snapshot

The preferred route is the GitHub Actions workflow:

```text
Actions → Build candidate data snapshot → Run workflow
```

Supply the explicit cutoff, for example `2025-12-31`.

For a local build:

```bash
python -m src.scripts.refresh_data --as-of 2025-12-31
```

This command fetches sources, validates and normalizes them, trains to the
cutoff, runs model validation, saves artifacts, and creates the serving
manifest.

## Candidate Review

Before promotion, verify:

- Workflow tests passed.
- Every source has an expected version and cutoff.
- Manifest status is `verified`.
- No country appears across grouped train/holdout sets.
- Coverage and imputation did not deteriorate materially.
- Large score and tier changes have explanations.
- Model and data cards are updated.
- Streamlit loads the candidate artifacts successfully.

## Promotion

Promotion is intentionally manual until governance controls are proven.

1. Download the candidate artifact bundle.
2. Verify manifest checksums.
3. Review data and model change reports.
4. Replace serving artifacts through a reviewed pull request or approved
   release pointer.
5. Verify Streamlit health and visible snapshot status.
6. Record the approver and release identifier.

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
