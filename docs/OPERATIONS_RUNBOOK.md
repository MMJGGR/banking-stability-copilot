# BankEnv Operations Runbook

## Local Verification

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py train_model.py src tests
python -m pytest -q
python -m src.scripts.audit_model_policy
streamlit run app.py
```

## Build External-Liquidity Feature Inputs

Preferred route (api.imf.org must be reachable, which it is from GitHub
Actions):

```text
Actions → Fetch external-liquidity source data → Run workflow
```

The workflow discovers each family's dataflow ID and key structure, then uses
the bounded external-liquidity feature builder. It queries exact BOP/IIP SDMX
path keys for the model-country ISO3 universe instead of downloading complete
large IMF dataflows. It uploads `external_sources_discovery.json`,
`external_liquidity_features_report.json`, and `cache/external/*.parquet` as
run artifacts for review. Nothing is committed automatically.

Current feature coverage from the July 2026 local full-universe run is 185 of
201 scored countries with at least one staged external-liquidity feature; core
BOP flow ratios cover 181 countries and IIP position ratios cover 159-166
countries depending on feature.

Local equivalent:

```bash
python -m src.scripts.discover_external_sources
python -m src.scripts.build_external_liquidity_features --fetch --start-period 2018
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

### Automatic (last-known-good fallback)

The application degrades automatically: when the active
`cache/risk_model.pkl` is missing, fails its manifest checksum, or fails
contract validation, `load_model_artifact_with_fallback` serves the newest
checksum-valid bundle from `artifacts/snapshots/` instead. The header badge
switches to "Fallback Mode" and the System Health panel names the bundle in
use and the load error. No operator action is needed to keep the app up, but
fallback mode is an incident: diagnose and restore the active artifact.

### Manual rollback to a specific snapshot

```bash
# 1. Pick the bundle to restore (each is a complete serveable set):
ls artifacts/snapshots/

# 2. Copy its artifacts over the active set:
cp artifacts/snapshots/<SNAPSHOT>/risk_model.pkl            cache/risk_model.pkl
cp artifacts/snapshots/<SNAPSHOT>/inference_pipeline.pkl    cache/inference_pipeline.pkl
cp artifacts/snapshots/<SNAPSHOT>/crisis_classifier.pkl     cache/crisis_classifier.pkl
cp artifacts/snapshots/<SNAPSHOT>/crisis_features.parquet   cache/crisis_features.parquet
cp artifacts/snapshots/<SNAPSHOT>/imputed_features.parquet  cache/imputed_features.parquet
cp artifacts/snapshots/<SNAPSHOT>/data_manifest.json        artifacts/data_manifest.json

# 3. Verify locally, then commit and push through a reviewed PR (git-lfs
#    required for cache/*):
python -m pytest -q tests/test_model_store.py tests/test_model_fallback.py
streamlit run app.py   # header must show the restored snapshot ID, healthy badge
```

4. Verify the displayed snapshot ID and checksum after deployment.
5. Record the incident and affected release.

## Degraded Operation

When a source is unavailable:

- Do not replace the last validated source snapshot.
- Mark freshness SLA breaches in the candidate report.
- Keep the live application on the previous approved snapshot.
- Escalate only if the source remains unavailable beyond its expected cadence.

The System Health panel in the app header surfaces degraded states to users:
serving mode (active/fallback), snapshot age, and per-source freshness against
the SLAs proposed in `docs/GOVERNANCE.md`.

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
