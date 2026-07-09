# Banking Stability Data Card

## Current Local Sources

| Source | Role | Current effective local position |
|---|---|---|
| IMF WEO | Macroeconomic and fiscal indicators | Local vintage contains values through 2030; snapshot logic must separate actuals, estimates, and projections |
| IMF FSIC | Core banking soundness indicators | Existing normalized cache reaches 2025 Q3 |
| IMF MFS | Monetary and banking balance sheets | Existing normalized cache reaches 2025 Q3 |
| IMF FSIBSIS | Detailed banking balance-sheet and income-statement data | Raw local source includes observations through November 2025; derived historical features previously used annual data only |
| World Bank WGI | Structural governance indicators | Current release covers through 2024 |

The generated `artifacts/data_manifest.json` is the authoritative machine-
readable record of the current serving inputs and checksums.

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
- `loan_concentration`: approximately one-third coverage.
- `large_exposure_ratio`: approximately one-third coverage.
- `sovereign_exposure_fsibsis`: approximately one-third coverage.
- Several FSIBSIS-derived funding and income features: below 50% coverage.

Country-level outputs must disclose coverage, freshness, carry-forward, and
imputation.

## Refresh and Fallback Policy

Retrieval order:

1. Configured official API export.
2. Configured official bulk download.
3. Last validated local snapshot.
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
