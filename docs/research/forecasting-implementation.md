# Broad-feature forecasting: implementation handoff

Research branch: `research/broad-feature-forecasting-2026-09-16`
Baseline: `5957ca779dafa21f2e098c819bfb060f43243206`.
PRD: [`../prd/broad-feature-forecasting.md`](../prd/broad-feature-forecasting.md).
PRD-first commit: `e3829e86f3d44c0fdeea01ef8f4071909bdb5e0b` (documentation only).

## Current scope

This is the first executable foundation, not the complete forecasting product. Production is deliberately unchanged. There is no live model, new crisis probability, predictive-validation claim, final holdout result, or permission to promote.

| Milestone | State in this handoff |
|---|---|
| M0: requirements and isolated branch | Complete; PRD committed before any implementation |
| M1: broad source inventory | Implementation and synthetic checks complete; repository workflow generates full five-source evidence |
| M2: temporal panel and baseline harness | Partial: point-in-time selector, purged splits and matched baseline harness implemented; real panel assembly, target registry and metadata recovery remain |
| M3: broad/structured model experiments | Not run; no winner selected |
| M4: dynamic states, analogues and hazards | Existing hazard foundation remains available; new integration not implemented |
| M5: confirmation and analyst report | Not performed |
| M6: live integration | Not authorized |

## Implemented files and requirement coverage

- `src/forecasting/inventory.py` (F01-F04): adapters for FSIC, FSIBSIS, MFS, WEO and WGI; stable identity hashes over source/measure/frequency/unit and all available economic dimensions; explicit label-only/unknown-unit status; conflicting cell ledger; per-feature country/year coverage. No economic predictor allowlist or count cap. Unobserved wide-source labels are reported separately. Unknown source schemas fail rather than return an empty success.
- `src/forecasting/temporal.py` (parts of F03, F05, F13): verified-vintage versus explicit retrospective modes; later-publication/revision gates; conflicting latest observations rejected; latest missing values not replaced by older values; train labels purged when unresolved at the validation origin. Supplied timestamp provenance remains the caller's responsibility.
- `src/forecasting/baseline.py` (parts of F05, F08-F09, F15): dense ridge with training-only median imputation/scaling; all training-observed predictors eligible; all-missing training predictors recorded as unlearnable rather than erased from the library. Matched persistence/compact/broad comparisons report development-only metrics. Alpha defaults are prototype settings, not selected optimal parameters.
- `src/scripts/audit_forecasting_sources.py` (F04, F17): all-five-source checksum verification, read-only inventory and fresh output directories; protects serving/source paths; failed inventories retain a failure marker, not success. No model-pickle loading or source network retrieval.
- `tests/test_forecasting_foundation.py`: 58 local tests passed on Python 3.13 / scikit-learn 1.8 before publication. The research workflow rechecks them on Python 3.11 / scikit-learn 1.5.2; existing repository quality CI remains independent.

## Reproduction

Use the repository's development environment, with the existing committed LFS source caches hydrated. Do not run the production refresh command for this research work.

```bash
python -m pytest tests/test_forecasting_foundation.py -q
python -m src.scripts.audit_forecasting_sources \
  --snapshot . --output research-output/forecasting-inventory-NEW-RUN
```

The command refuses to overwrite an existing output directory. Registry CSVs enumerate candidate source identities, not automatically admitted predictors. Separate frequency/unit variants are not independent evidence. `summary.json`, conflict ledgers, country coverage and year coverage constitute the source audit. Test and scope evidence are attached to the research workflow run.

## Important data findings to carry into M2

The initial schema inspection (run `35128233460`) verified all five committed source-cache hashes and found different storage forms: FSIC/MFS/WEO are long-form; FSIBSIS is a sector/indicator-labelled wide table; WGI is a country-year matrix. Some unit fields are empty, FSIBSIS does not expose a separate canonical source-code column, and these caches do not carry historical public release/vintage timestamps. Preserve those series in the registry and recover upstream metadata rather than inventing their semantics or claiming point-in-time availability.

No final confirmation period has been evaluated by this foundation. Before any target-dependent experiment: version the target registry and lag policy, identify previously inspected years, establish a genuinely unexamined confirmation policy, and freeze the matched evaluation plan. A current-vintage retrospective panel must remain labelled retrospective.

## Remaining acceptance work

The source inventory does not yet provide an economic feature-family ontology, full historical availability, raw-dimension recovery or resolved source conflicts. The model library does not yet implement elastic net, dual-path factors, boosted trees, state-space trajectories, analogue forecasts, uncertainty calibration or joint peer simulations. The PRD covers these; this handoff does not claim them.

The next implementation milestone is canonical metadata recovery and a ragged broad-feature country-origin panel using the existing crisis-research contracts where applicable. Then register development folds and compare models on the same eligible rows, with broader-population coverage reported separately. Any final model still requires predictive validation and a separate owner-approved release.
