# PRD amendment v0.2: source-grounded forecasting data contracts

Date: 2026-09-16. Applies together with `broad-feature-forecasting.md` v0.1, retaining F01-F18 and M0-M6. This amendment takes precedence for source semantics and M2 execution. Author: assistant under the owner's instruction to inspect live data, tune the PRD and start execution when ready. Production promotion remains unauthorized.

## Readiness decision

**Ready to execute M2 canonical-data and panel development on the isolated branch, after committing this amendment. Not ready for production forecasts or claims of predictive validation.** The inspection identified solvable information loss in the serving-cache representation. Rebuild a research-only canonical store from official source responses rather than guess the missing dimensions or silently exclude the affected features. Missing historical publication/revision dates remain a limitation; the first panel is explicitly retrospective.

## Evidence inspected

- Entire five-source serving caches from the September 15 snapshot, checked against the committed manifest.
- Prior full inventory: 2,368 cache-series identities, 70,771 conflicting country/feature/period cells. These are cache identities/conflicts, not independent predictors or necessarily erroneous source observations.
- Fresh September 16 official SDMX structure responses (all references) and bounded Kenya observation queries for FSIC, FSIBSIS, MFS_DC and WEO, plus the World Bank WGI API. All nine requests returned HTTP 200. Run `35130443893`, artifact `10461475785`, archive SHA256 `b95635e2566884ab029608c0a6854dc20fb6aec8a2fb12e90d9d89e03354e79e`.
- Source documentation: https://data.imf.org/en/Resource-Pages/IMF-API ; https://data.imf.org/en/datasets/IMF.STA%3AFSIC ; https://data.imf.org/en/datasets/IMF.STA%3AFSIBSIS ; https://data.imf.org/en/datasets/IMF.STA%3AMFS_DC ; https://www.worldbank.org/en/publication/worldwide-governance-indicators/documentation .

This is source-data/schema inspection, not authenticated dashboard inspection. Bounded live samples establish the response format, not a fresh all-country data refresh. Full research retrieval must carry its own timestamps/hashes and must not be mixed silently with the serving vintage.

## Findings and binding implementation requirements

### D01 — Preserve all published entity codes

The serving WEO cache has 3,024 conflicting cells, all under truncated `G11` or `G20`. Official codelists distinguish such entities as `G110` (Advanced Economies), `G119` (G7), `G200` (Emerging Market and Developing Economies), `G201`, `G202` and `G205`. Never truncate source identifiers to three characters. Keep original entity labels and source code, with a separate country/aggregate/global/unclassified role. Aggregates are contextual features, not additional countries or independent training outcomes. Record, rather than discard, unmapped entities. Full retrieval must test whether preserving codes resolves the measured collisions.

### D02 — Retain sector and transformation dimensions

FSIC/FSIBSIS keys include COUNTRY, SECTOR, INDICATOR and FREQUENCY. MFS keys include COUNTRY, INDICATOR, TYPE_OF_TRANSFORMATION and FREQUENCY. Live MFS Kenya broad money has both `XDC` and `SA_XDC` at the same date: these are distinct unadjusted and seasonally adjusted measurements, not contradictory revisions. Additional transformations include percentage changes and monetary-base/broad-money ratios. Preserve and label every transformation from the official codelist; never average them, overwrite by row order, or conflate growth with levels. Keep financial subsectors beyond deposit takers for potential spillover features; exact target definitions must specify their intended sector.

### D03 — Distinguish attribute-only rows from missing observations

Official CSV responses include rows carrying indicator-level metadata but no COUNTRY/TIME_PERIOD/OBS_VALUE. The bounded WEO response has 106 such rows; FSIC has 311; FSIBSIS has 470. Recover attributes according to their DSD attachment keys (for example UNIT by INDICATOR), retaining conflicting metadata as an error/quarantine. A genuine dated observation with missing numeric value is a different record and stays missing. Metadata rows must not become countries, targets or artificial observations.

### D04 — Units, scale and semantics are explicit

Recover FSIBSIS source codes and UNIT attributes from raw responses. MFS may encode the unit in its verified transformation codelist while the UNIT column is empty. Preserve the resolution method. WGI `GOV_WGI_*.SC` denotes the absolute 0-100 score, not the older approximately -2.5 to 2.5 estimate or a percentile rank.

Preserve raw OBS_VALUE and SCALE separately. The inspected MFS/FSI samples already contain large nominal amounts alongside SCALE=6; do not multiply values again merely because SCALE exists. Cross-source arithmetic requires an independently documented scale/unit conversion rule. A same-code scale or definition change must be visible, not silently bridged. Raw domestic-currency magnitudes are not automatically comparable across countries; model-ready representations must use defensible normalization or country-relative transformations, evaluated separately from raw-level benchmarks. No arbitrary feature-count cap is introduced.

### D05 — Status, estimates, forecasts and source errors

The MFS live sample carries status `B`; decode it using the official status codelist rather than treating it as a crisis or ordinary observation. FSIBSIS includes two dated records carrying `#VALUE!` and missing values. Preserve these records and reasons. Do not reinterpret blanks as zeros or discard all negative values; negative net positions, profit and fiscal balances can be valid. Record observed zeros and implausibility flags separately.

WEO's COUNTRY_UPDATE_DATE is not proof of when every historical value first became public. Historical period <= cutoff is not proof of actual realization. Keep projection/estimate/unknown status distinct. When the raw source does not expose an actual-status boundary, exclude years at/after the declared current-vintage boundary from realized targets and mark older values `historical_status_unverified`; such values may support only an explicitly retrospective experiment. A documented source-specific target-status policy is mandatory.

### D06 — Annual endpoints are not invented annual aggregates

For the first panel, preserve source frequency and select exact December/quarter-four/year-end snapshots as separate features. Never sum ratios, average stocks into year-end values, sum cumulative year-to-date income flows, or duplicate annual observations to pretend they are quarterly information. Later flow/stock/ratio aggregation rules require semantic registration. Feature lag/difference calculations must use exact calendar-year matches; a missing year cannot shorten a nominal one-year lag.

### D07 — Cross-source duplication and shared context

FSIC contains both ratios and component amounts; FSIBSIS contains overlapping balance-sheet/income-statement components. Retain both in the library, but record overlaps and test family/group ablations instead of counting them as independent confirmations. WEO includes global commodity prices and group aggregates; features recorded for one global entity must not be automatically discarded by country-coverage filters. Context may be broadcast to countries only with an explicit context entity and as-of rule. Do not treat a similar country as an observed contagion edge.

### D08 — Definition breaks and irregular history

Retain consolidation basis, accounting standard, derivation/source flags and breaks whenever populated. Do not invent a stable historical definition from a single current metadata release. Emit counts of sparse/short histories, missing consecutive years, frequency disagreements and scales. Keep age/missingness as interpretable feature-quality information, separate from provenance dates and raw country identifiers. Do not call coverage statistical confidence.

### D09 — Retrospective versus genuine historical availability

The first canonical store records retrieval timestamps and DSD hashes but leaves unknown historical public-release and vintage timestamps unknown. WGI's revised historical scores are one harmonized current vintage, not a record of what was published each historical year. Freeze explicit retrospective lag assumptions before target-dependent experiments. Strict point-in-time mode must still fail when evidence is absent. Source metadata inspection can examine the complete schema without constituting predictive confirmation.

### D10 — Development registration, not a false untouched holdout

M2 execution will build the broad library and exact annual endpoints, with initial one-/two-year targets for deposit-taker NPL, regulatory capital and liquidity ratios. This small target roster does not limit predictors. First-stage acceptance is correct identities, units/flags, calendar alignment, outcome separation and reproducibility. No feature selection may use future outcomes.

Before the first real-data fitting run, commit a separate experiment plan defining training/development origin ranges, lag assumptions, matched cohorts, metrics, fixed baseline parameters and permitted tuning. The already inspected historical periods and prior failed crisis holdouts cannot be called fresh confirmation. A final confirmation period is not designated or consumed by M2. The first runs, if executed, must be labelled diagnostic/development-only, not grounds for selecting a production winner.

### D11 — Source snapshot separation

Live research retrievals go to a new immutable research directory with response URLs, retrieval times, byte sizes, HTTP states, source versions and SHA256 hashes. Download failures, partial sources, incompatible schemas or unresolved metadata cannot be silently replaced by a different vintage. Source caches, serving manifests, classifiers, scores and app routes remain unchanged. A source-level failure must be explicit even if other sources can be inspected.

### D12 — Supplemental source scope remains open

The five core sources are the initial connected input set, not the maximum library. Inventory already available supplementary BIS/World Bank/external-liquidity source configurations separately. Unused or not-yet-fetched sources remain in a dependency ledger. No paid purchase or new authenticated service is authorized.

## M2 acceptance additions

1. Preserve `G110` and `G119` as separate entities; both survive permutation/duplicate tests.
2. Preserve `XDC`, `SA_XDC` and percentage-change transformations as distinct MFS features.
3. Parse attribute-only rows using their attachment keys; contradictory UNIT declarations fail closed.
4. Preserve missing dated values, nonstandard status tokens, scale and flags; do not multiply SCALE without a registered conversion.
5. Map official codes to labels via the matching DSD/concept/codelist, not a name-only heuristic.
6. Use exact calendar endpoints for annual inputs and exact target years; no backfilled or interpolated labels.
7. Source-level ledgers reconcile raw observations, metadata rows, rejected rows, duplicate/conflict cells and canonical cells.
8. Every retained source identity is in the library; staged/canonical/panel/model-admitted states are separate.
9. M2 source and panel execution leaves all production bytes and master unchanged.
10. Passing these gates establishes execution readiness and engineering correctness only, not forecasting skill.

## Authorized next execution

Implement a DSD-aware research normalizer and annual endpoint/panel builder; test against the inspected live samples; run a full official research retrieval and audit when successful; export metadata/coverage and target-availability ledgers. Any unresolved semantic issue is a recorded data dependency, not a license to guess. Keep PR #28 draft. M3-M5 statistical comparison/confirmation and M6 live integration are not declared complete by this amendment.
