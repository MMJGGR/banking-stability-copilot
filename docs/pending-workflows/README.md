# Pending workflow files

Files here are ready-to-install GitHub Actions workflows that automation
cannot push: GitHub rejects any App credential without the `workflows`
permission for changes under `.github/workflows/`, and the integration used
by Claude sessions does not carry that scope.

To install one: on GitHub, Add file → Create new file →
`.github/workflows/<name>.yml` → paste the file's content → commit. Delete
the copy here in the same PR/commit once installed.

- `external-data.yml` — external-liquidity source retrieval (backlog ranks
  5-14); companion scripts `discover_external_sources.py` and
  `fetch_external_sources.py` are already on master.
