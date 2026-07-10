# Proposed workflow changes (pending `workflows` permission)

The automation session that produced this branch could not push changes
under `.github/workflows/` because the GitHub App token lacks the
`workflows` permission. Apply these by moving the three YAML files into
`.github/workflows/` (overwriting the two existing ones) in a commit made
with your own credentials:

- `promote-snapshot.yml` (new): downloads a candidate bundle from a
  "Build candidate data snapshot" run, re-runs the serving-artifact smoke
  tests against it, commits the artifacts via Git LFS, and opens a
  promotion pull request for review.
- `refresh-data.yml` (modified): adds the smoke-test gate after the
  candidate build and ships `artifacts/feature_heuristics.json` with the
  candidate bundle.
- `quality-check.yml` (modified): installs with `-c constraints-dev.txt`
  so CI builds are reproducible.

```bash
git mv .github/workflows-proposed/promote-snapshot.yml .github/workflows/
git mv .github/workflows-proposed/refresh-data.yml .github/workflows/
git mv .github/workflows-proposed/quality-check.yml .github/workflows/
git rm -r .github/workflows-proposed
```
