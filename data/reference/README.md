# Reference datasets

This directory holds small, manually retrieved reference files whose
provenance must be pinned by checksum.

## Staged challenger feature datasets

Compact derived feature files the hosted app reads directly, so it never loads
the large upstream IMF/World Bank flows at startup. They are **staged
challenger inputs** and do not affect production scoring.

- `external_liquidity_features.parquet` / `external_feature_observations.parquet`
  / `external_liquidity_features_report.json` — IMF BOP/IIP + World Bank
  external-liquidity block, plus market/external stress inputs: FDI flow
  stability (IMF BOP direct investment), export-concentration / terms-of-trade
  and commodity dependence (World Bank), and real-effective-exchange-rate
  valuation stress (World Bank). Rebuilt by
  `src/scripts/build_external_liquidity_features.py` (needs IMF/WB API access,
  so it runs in GitHub Actions; the market/FDI/REER columns populate on the
  next fetch).
- `government_liquidity_features.parquet` /
  `government_liquidity_observations.parquet` /
  `government_liquidity_features_report.json` — IMF WEO general-government
  (sovereign fiscal) liquidity block: gross debt, primary/structural balance,
  implied interest burden, and interest-to-revenue / debt-to-revenue
  affordability ratios. Rebuilt by
  `src/scripts/build_government_liquidity_features.py`; needs no external API
  because the WEO series come from the cache, so it can be rebuilt locally with
  `--reference-dir data/reference`.

## Laeven-Valencia crisis episode dataset (required for label provenance)

The training labels in `src/crisis_labels.py` are transcribed from:

> Laeven, L., & Valencia, F. (2026). Systemic Banking Crises Database:
> 1970-2025. IMF Working Paper WP/26/94.

IMF web hosts return HTTP 403 to non-browser clients, so the published
workbook cannot be fetched by automation. To verify the transcription:

1. Download the episode dataset (Excel or CSV) from the working-paper page in
   a browser and place it in this directory.
2. Run:

   ```bash
   python -m src.scripts.verify_crisis_labels --dataset data/reference/<file>
   ```

   Pass `--country-col/--start-col/--end-col/--sheet` if the layout is not
   auto-detected.
3. The script pins the file's SHA-256 in `crisis_label_source.json`, writes a
   reconciliation report to `artifacts/crisis_label_reconciliation.json`, and
   fails when the in-code dictionary disagrees with the dataset.

Once the registry file exists, `tests/test_verify_crisis_labels.py` re-runs
the reconciliation on every test run, so any future label edit that diverges
from the pinned dataset fails CI.
