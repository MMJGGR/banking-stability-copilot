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

### Research-only BIS historical supplement

The local build workflow can optionally add official BIS bulk histories for
private/bank credit, the published credit-to-GDP gap, private-sector
debt-service ratios, and selected residential property prices. These series are
not part of the active serving snapshot and are never retrieved by Streamlit.
They are intended for historical-core versus modern-full challenger tests.

| BIS data set | Research role | Local adapter coverage check |
|---|---|---|
| `WS_TC` | Private- and bank-credit depth | 43 countries; history begins in 1947 for some series |
| `WS_CREDIT_GAP` | Official private credit-to-GDP gap | 43 countries; history begins in 1957 |
| `WS_DSR` | Total, household and corporate debt-service ratios | 32 countries; history begins in 1999 |
| `WS_SPP` | Selected residential property-price growth | 57 countries; some nominal histories begin in 1927 |

The combined local smoke test normalized 66,854 quarterly observations across
59 countries and 10 candidate features. Coverage is materially narrower than
the World Bank fallback, so BIS fields are opt-in challenger inputs rather than
silent replacements. Every normalized observation preserves its source URL,
series key, status, vintage and data-set version.

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
- BIS debt-service, credit-gap and property candidates have stronger concept
  quality but materially narrower country coverage; expert routing and
  forward-time validation are required before any scoring use.

Country-level outputs must disclose coverage, freshness, carry-forward, and
imputation.

## Packaged Liquidity Datasets

These blocks are packaged as compact reference files under `data/reference/`
and surfaced in the Country Profile, Data Explorer, and Methodology tabs. The
active structural model artifact determines its own score inputs. The separate
hierarchical hazard is trained on the annual historical panel: overlapping
canonical fields can enter that panel, while most newly staged external- and
government-liquidity fields currently enrich the latest mechanism evidence
only. A packaged or visible field must not be described as predictive merely
because it is present in the current cross-section.

| Dataset | Source | What it adds | Coverage (model countries) |
|---|---|---|---|
| External liquidity | IMF BOP/IIP + World Bank WDI/IDS | Current-account receipts/payments, reserve adequacy, net IIP, external and portfolio liabilities, external debt service, a gross-external-financing-need proxy, FDI flow stability, export-concentration / terms-of-trade, and REER valuation stress | ~79-90% (market/FDI/REER series populate on the next CI fetch) |
| Government liquidity | IMF WEO general government (`GGXWDG_NGDP`, `GGXCNL_NGDP`, `GGXONLB_NGDP`, `GGR_NGDP`, `GGX_NGDP`, `GGSB_NPGDP`) | Gross public debt, primary/structural balance, implied interest burden, and the rating-agency affordability ratios interest-to-revenue and debt-to-revenue | ~94-100% for core ratios; structural balance ~42% |

In the current promoted 2026-06-30 artifact, `govt_interest_to_revenue` and
`govt_debt_to_revenue` are production-scored government-liquidity inputs.

Mechanism signal strengths are current-cohort percentiles. They support
cross-sectional diagnosis but are not yet directly comparable across vintages;
a frozen reference distribution is required before interpreting a change as a
country-level time-series movement.

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
