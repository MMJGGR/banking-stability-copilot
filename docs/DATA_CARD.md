# Banking Stability Data Card

## Current Active Sources

| Source | Role | Current effective position |
|---|---|---|
| IMF WEO | Macroeconomic and fiscal indicators | Official SDMX dataflow `IMF.RES:WEO(9.0.0)`; rows through 2031 horizon; snapshot logic separates actuals, estimates, and projections |
| IMF FSIC | Core banking soundness indicators | Official SDMX dataflow `IMF.STA:FSIC(13.0.1)`; normalized cache reaches 2026-04 |
| IMF MFS | Monetary and banking balance sheets | Official SDMX dataflow `IMF.STA:MFS_DC(8.0.0)`; normalized cache reaches 2026-05 |
| IMF FSIBSIS | Detailed banking balance-sheet and income-statement data | Official SDMX dataflow `IMF.STA:FSIBSIS(18.0.0)`; normalized cache reaches 2026-M04 |
| World Bank WGI | Structural governance indicators | World Bank API; current data through 2024 |
| IMF systemic banking crises | Supervised target labels | May 2026 Laeven-Valencia release covers 1970-2025; borderline events are excluded from training by default |

The generated `artifacts/data_manifest.json` is the authoritative machine-
readable record of the current serving inputs and checksums.

Current active serving snapshot:

- Snapshot cutoff: `2026-06-30`
- Manifest status: `verified`
- Source mode: `official_api_sdmx_worldbank`
- Archived official API checkpoint bundle: `artifacts/snapshots/2026-06-30-official-api`
- Previous local cached-source checkpoint bundle: `artifacts/snapshots/2026-06-30`
- YE2025 checkpoint bundle: `artifacts/snapshots/2025-12-31`

Freshness note: the `2026-06-30` model is a mid-2026 cutoff snapshot. The
official IMF banking and monetary sources now include 2026 observations where
available. WEO remains the October 2025 vintage upstream (`9.0.0`) with a
forecast horizon through 2031; the model cutoff excludes periods after
2026-06-30 and does not include WEO projections in scoring.

The crisis-label source is IMF Working Paper 26/94. The project distinguishes
systemic episodes from the source's explicitly borderline Nicaragua 2018,
Vietnam 2022, and Sri Lanka 2023 cases. Borderline cases are queryable for
sensitivity analysis but are not positive training targets by default.

## Observation Status

New normalized WEO caches preserve:

- `actual`
- `estimate`
- `projection`
- `unknown`

Feature and snapshot workflows will additionally support:

- `carried_forward`
- `imputed`
- `missing`

No estimate or projection may be silently represented as actual.

## Material Coverage Gaps

Known weak features include:

- `tot_deterioration_3yr`: no current direct coverage.
- `external_debt_gdp`: no current coverage in the local WEO cache under the
  configured indicator code; cached-classifier scoring uses an explicit
  all-null fallback until the source mapping is resolved.
- `loan_concentration`: approximately one-third coverage.
- `large_exposure_ratio`: approximately one-third coverage.
- `sovereign_exposure_fsibsis`: approximately one-third coverage.
- Several FSIBSIS-derived funding and income features: below 50% coverage.

Country-level outputs must disclose coverage, freshness, carry-forward, and
imputation.

## Refresh and Fallback Policy

Retrieval order:

1. Official IMF SDMX / World Bank API retrieval.
2. Reused raw official downloads in the selected download directory when
   `--reuse-downloads` is explicitly passed.
3. Legacy configured export URLs or local fallback files when
   `--retrieval-mode legacy` is explicitly selected.
4. Fail without replacing production.

Candidate snapshots are produced through the manual
`Build candidate data snapshot` GitHub workflow. They are uploaded for review
and are not automatically promoted.

## Quality Gates

A source update is rejected when:

- Required fields disappear.
- Observation periods move backwards.
- Duplicate country-indicator-period records appear.
- Country or indicator coverage falls beyond tolerance.
- Invalid values increase materially.
- Projections enter an actual-only snapshot.
- The manifest or checksums are incomplete.

## Retention

- Raw source snapshots should be stored as release or object-storage assets.
- Git should contain code, schemas, manifests, and compact serving artifacts.
- Every published snapshot must remain reproducible from its raw-source IDs,
  transformation version, model version, and manifest.
