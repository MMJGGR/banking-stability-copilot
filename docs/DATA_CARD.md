# Banking Stability Data Card

## Current Active Sources

| Source | Role | Current serving-manifest position |
|---|---|---|
| IMF WEO | Macroeconomic and fiscal indicators | Local normalized cache; 208 countries, 145 indicators, latest included observation 2025-12-31 |
| IMF FSIC | Core banking soundness indicators | Local normalized cache; 155 countries, 398 indicators, latest included observation 2026-05-31 |
| IMF MFS | Monetary and banking balance sheets | Local normalized cache; 180 countries, 91 indicators, latest included observation 2026-05-31 |
| IMF FSIBSIS | Detailed banking balance-sheet and income-statement data | Local normalized cache; 141 countries, 289 observed indicator labels, latest included observation 2026-05-31 |
| World Bank WGI | Structural governance indicators | Local normalized cache; 207 countries, six measures, latest included observation 2024-12-31 |
| IMF WP/26/94 crisis episodes | Governed label reference for research/retraining | Exact Appendix I, Table A1 artifact: 161 systemic and three explicitly borderline episodes; not the label build used by the served legacy classifier |

The generated `artifacts/data_manifest.json` is the authoritative machine-
readable record of the current serving inputs and checksums.

Current active serving snapshot:

- Snapshot cutoff: `2026-06-30`
- Manifest status: `verified` (actively served; no recorded `approved` lifecycle transition)
- Source mode: `local_cached_sources`
- Generated: `2026-07-12T14:39:35.619879+00:00`
- Archived official API checkpoint bundle: `artifacts/snapshots/2026-06-30-official-api`
- Previous local cached-source checkpoint bundle: `artifacts/snapshots/2026-06-30`
- YE2025 checkpoint bundle: `artifacts/snapshots/2025-12-31`

Freshness note: the `2026-06-30` model is a mid-2026 cutoff snapshot. Its
locally cached IMF banking and monetary inputs include May 2026 observations
where available. The active manifest does not record source dataflow versions,
so those versions must not be inferred from the archived official-API bundle.
WEO projections are excluded from scoring.

The governed crisis-label source is Laeven and Valencia (2026), *Systemic
Banking Crises Database: 1970-2025*, IMF Working Paper WP/26/94, Appendix I,
Table A1. The pinned CSV preserves all 164 published rows and the source PDF
checksum. Nicaragua 2018, Vietnam 2022, and Sri Lanka 2023 are explicitly
borderline and are not positive targets by default. This exact artifact was
added after the served classifier was trained; it governs research and future
retraining, not the provenance of the current legacy classifier probabilities.

## Research-only BIS Historical Supplement

The repository includes an offline adapter for four official BIS bulk data
sets: total credit (`WS_TC`), the published credit-to-GDP gap
(`WS_CREDIT_GAP`), debt-service ratios (`WS_DSR`), and selected residential
property prices (`WS_SPP`). These series are optional model-development inputs.
They are not in `DEFAULT_FEATURE_SPECS` or any active/default serving feature
set, are not used by the committed serving artifacts, and are never retrieved
by Streamlit at runtime.

The normalized history is built with
`python -m src.scripts.fetch_bis_financial_history`. Local cache and manifest
outputs are ignored by Git and must be promoted through the normal candidate
data workflow if a future validated model uses them. BIS coverage is narrower
than the World Bank fallback universe, so country, period, and data-regime
coverage must be reported before model comparison.

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
- BIS debt-service, credit-gap and property series are higher-quality direct
  candidates for covered countries but have a materially narrower universe;
  they remain opt-in research inputs.

Country-level outputs must disclose coverage, freshness, carry-forward, and
imputation.

## Packaged Liquidity Datasets

These blocks are packaged as compact reference files under `data/reference/`
and surfaced in the Country Profile, Data Explorer, and Methodology tabs. The
active model artifact determines score role: fields present in the promoted
feature matrix are production-scored; remaining packaged fields are insight
only until a future promoted model includes them.

| Dataset | Source | What it adds | Coverage (model countries) |
|---|---|---|---|
| External liquidity | IMF BOP/IIP + World Bank WDI/IDS | Current-account receipts/payments, reserve adequacy, net IIP, external and portfolio liabilities, external debt service, a gross-external-financing-need proxy, FDI flow stability, export-concentration / terms-of-trade, and REER valuation stress | ~79-90% (market/FDI/REER series populate on the next CI fetch) |
| Government liquidity | IMF WEO general government (`GGXWDG_NGDP`, `GGXCNL_NGDP`, `GGXONLB_NGDP`, `GGR_NGDP`, `GGX_NGDP`, `GGSB_NPGDP`) | Gross public debt, primary/structural balance, implied interest burden, and the rating-agency affordability ratios interest-to-revenue and debt-to-revenue | ~94-100% for core ratios; structural balance ~42% |

In the current serving 2026-06-30 artifact, the persisted economic loading map
shows six packaged government-liquidity fields as active score inputs:
`govt_interest_to_revenue`, `govt_debt_to_revenue`, `govt_revenue_gdp`,
`govt_interest_to_revenue_change_3y`,
`govt_debt_to_revenue_change_3y`, and
`govt_revenue_gdp_change_3y`. The Methodology tab derives this role from the
served loading map instead of trusting an older candidate-report label.

Government-liquidity build notes:

- The cutoff and observation-status selection match the production WEO feature
  path (actuals and estimates only; projections excluded).
- `govt_interest_gdp` is derived as primary balance minus overall balance
  because the overall balance already nets out interest; it is an implied
  interest bill, not a reported interest-expense series.
- A full gross financing need additionally requires debt amortization/rollover,
  which WEO does not carry; that remains an IMF Fiscal Monitor / GFS source gap.
- Built by `python -m src.scripts.build_government_liquidity_features`; no
  external API calls are required because the WEO series come from the cache.

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
