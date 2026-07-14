# Reference datasets

This directory holds small, manually retrieved reference files whose
provenance must be pinned by checksum.

## Packaged liquidity feature datasets

Compact derived feature files the hosted app reads directly, so it never loads
the large upstream IMF/World Bank flows at startup. Some columns are active
production model inputs; other columns are surfaced for analysis only. The app
labels score role from the active model artifact.

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

The governed episode artifact is extracted from Appendix I, Table A1 of:

> Laeven, L., & Valencia, F. (2026). Systemic Banking Crises Database:
> 1970-2025. IMF Working Paper WP/26/94.

`systemic_banking_crises_1970_2025.csv` contains all 164 published rows: 161
systemic episodes and three explicitly borderline episodes. Each row carries
the source table, URL, and pinned source-PDF SHA-256. `src/crisis_labels.py`
loads and validates this artifact; it is not a separate hand-maintained event
dictionary.

IMF web hosts may return HTTP 403 to non-browser clients, so source retrieval
is an explicit offline step. To reproduce the pinned artifact:

1. Download the official working-paper PDF in a browser.
2. Run:

   ```bash
   python -m src.scripts.extract_crisis_labels_from_imf_pdf --pdf <WP2694.pdf>
   ```

3. The extractor enforces the pinned PDF checksum, exact classification counts,
   and source-row provenance before writing the CSV.

`tests/test_crisis_labels.py` verifies the exact counts, checksum format,
published dates, and borderline policy on every test run. The older
`verify_crisis_labels` utility remains available for an optional independent
cross-check against a separately downloaded workbook/CSV.
