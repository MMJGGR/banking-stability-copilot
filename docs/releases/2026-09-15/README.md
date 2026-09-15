# September 15, 2026 corrected banking-data release

User authorized correction and promotion after independent validation. This workflow prepares a branch; the final PR review and merge are separate.

## Corrections
- FSIC Tier 1 uses canonical FSI626_CFSI_PT, never Common Equity Tier 1 FSI15_CFSI_PT. Conflicting observations and economic dimensions fail closed.
- The pillar KNN metric now subtracts observed coordinates before taking their norm. This implements the same missing-aware Euclidean metric without cancellation between large nominal values. Neighbor count, distance weighting, feature units, pillar policy and existing classifier weights are unchanged.
- Historical classifiers/pipelines retain their existing metric. Only the rebuilt pillar pipeline adopts the numeric fix; no classifier retraining occurs.

## Validation
- All five source caches match the verified original September 15 artifact 10402061259 byte-for-byte.
- 303 tests passed; 1 skipped; zero failures.
- 201 scored countries retained; 15 manifest artifacts verified.
- Complete FSIC extraction agrees exactly for original, reversed and three shuffled source orders.
- New scores reproduce exactly from full raw model inputs at one, two and four computational threads. Historical exact replay uses saved imputed economic values; legacy raw-KNN discrepancies remain explicitly disclosed.
- Vietnam Tier 1: 10.58593648681355; capital quality: 87.77238459235349.
- Maximum score movement from both repairs/rebuild: 1.5; versus production: 4.8.
- Source commit: 6c832c59213e10a2da4abb577a0a7bc61ba71937
- Evidence: https://github.com/MMJGGR/banking-stability-copilot/actions/runs/34990214433

## Rollback
Revert the complete atomic release through a reviewed PR and rerun serving checks. Baseline: 860cfd64d795458a871f1a654e7497bf3a360012, preserved at rollback/pre-september-2026-09-15.

## Scope limits
This does not recertify the predictive classifier. PCA/imputation refits remain part of refresh methodology. Historical crisis-reference corrections are not new economic shocks. Selected-input validation does not certify every unused raw series.
