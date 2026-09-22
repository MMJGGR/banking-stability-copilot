# Broad-feature forecasting: execution status

Updated 2026-09-22. Branch `research/broad-feature-forecasting-2026-09-16`; draft PR #28. Production baseline `5957ca779dafa21f2e098c819bfb060f43243206` is unchanged. Requirements are the original PRD, the v0.2 source contracts and the v0.3 discovery/experiment registration. The original PRD-first commit is `e3829e86f3d44c0fdeea01ef8f4071909bdb5e0b`; v0.3 was committed before its implementation and real-data fitting.

## Breadth is a requirement, not a model result

`panel.py` TARGETS registers NPL, capital and liquidity OUTCOMES. It does not restrict predictors. The source library retains 5,318 identities; the historical country panel contains 15,295 level/lag/change representations. No top-k predictor limit, global complete-case requirement, serving-pillar allowlist or positive-correlation screen is used.

The new `discovery.py` fits target-independent empirical-rank components without the old two-pillar directions, development anchor or positive equal-weight shrinkage. Component count follows a declared variance fraction, not a fixed two-dimensional space. The broad regression and common-plus-residual paths retain all numerically identifiable directions; low-variance information is not deleted from the direct path.

## Completed execution

M0 and M1: requirements-first branch and complete core-source inventory.

M2 data slice: full official research retrieval and canonical panel executed in run 35134477499/artifact 10463106988 (September 16 vintage), with dimension recovery, distinct entities, explicit status/units and exact annual endpoints. This is retrospective data: historical release/vintage dates remain unknown. The 214 supplied feature-store entities are not the 201 published-score population.

M3 first registered slice, implemented and locally executed: target-independent discovery; causal within-entity normalization of currency amounts; training-only rank transforms; complete variance spectrum and all feature/component loadings; observed residual contributions and coverage; exact-profile duplication sensitivity; nested forward development comparisons of persistence, target-own-history compact ridge, broad ridge, and broad common-plus-residual ridge. Source/model admission is recorded separately from the uncapped library.

The local run passed 116 research tests, including 19 new discovery tests. It completed all 12 target/horizon/window cases (48 model/comparator evaluations), with no skipped cases. Local calculations used Python 3.13; GitHub Actions independently repeats the registered run on pinned Python 3.11 dependencies. Latest workflow evidence, not this prose, determines CI completion.

## Local results (not a production approval)

- Input archive SHA256: `f670f2e7caff532201c1a198130d0f8b34cb04784cb643dae2c5466af9359c9b`; 957 internal input-file digests checked before and after execution.
- Discovery origin 2023, underlying observations through 2022 with the one-year lag: 214 entities, 9,644 reference-varying representations. Of the full 15,295-column contract, 3,929 were constant in the reference and 1,722 wholly missing after admission/normalization. These remain in the ledger; no feature-count cap.
- Components required for 80%/90%/95% transformed variance: 73/103/128. The first component explains 13.435%. These are descriptive compression diagnostics, not a certified number of economic forces.
- 9,644 reference-varying columns form 4,292 exact transformed profiles when the observed mask is included. Collapsing those exact profiles changes the 90% count from 103 to 102. This does not establish independence of the remaining profiles or license deletion from the global library.
- Currency representations use causal within-entity prior scale; 21 unresolved-unit representations stay quarantined, with identities retained. The full inventory is not equivalent to all columns being admitted in every fold.
- Forecast development uses 2016-2018 and 2019-2021 windows, inner-only penalty selection and strict training-label availability boundaries. 4,958 matched target/horizon/origin examples, 19,832 prediction ledger rows. Broad models learned from 7,788-10,346 varying columns depending on the training fold; compact comparators used 9 own-measure frequency/lag/change representations.
- Neither initial broad specification beat persistence on aggregate MAE for any of the six target/horizon pairs. No production winner, predictive superiority or crisis-probability improvement is claimed. Results must be retained, not retuned on the same outer development windows and renamed confirmation.

## Implemented files and requirement coverage

- `inventory.py` and `audit_forecasting_sources.py`: uncapped source/identity/conflict/coverage inventory and immutable input verification (F01-F04, F17).
- `canonical.py`, `wgi.py`, `panel.py`, `resume_m2.py`, `build_forecasting_research_panel.py`: source-grounded identity recovery and exact-calendar retrospective panel (v0.2 D01-D12, parts of F03-F07).
- `temporal.py`: verified-vintage versus retrospective selection, unresolved revisions, purged label-availability boundaries (F03, F05, F13).
- `baseline.py`: retained general broad ridge/compact/persistence foundation (F08-F09, F15).
- `discovery.py`: full-library descriptive representation, complete spectrum, observed residuals and spectral ridge paths (v0.3 D13-D14).
- `discovery_execution.py`: predeclared nested-development comparisons and complete ledgers (v0.3 D15).
- `tests/test_forecasting*.py`: synthetic engineering tests. They do not certify predictive skill.

## Remaining scope and gates

M3 is NOT complete as a model-development programme. Unimplemented comparisons include elastic net, group weighting/ablation, target-level residual forecasts, nonlinear trees and neural challengers. This registered run has not established forecast intervals, dependence-aware significance or robust subgroup performance. Median-filled geometry still needs missingness/coverage sensitivity; low PC/coverage correlation is not proof of no missingness bias.

M4 is not complete: dynamic-factor/state-space transitions, calibrated historical-analogue retrieval, existing hazard integration, shared global/regional context broadcasting and coherent joint peer simulations remain work. The context-candidate store is retained but not automatically treated as extra countries or silently added without an entity/availability contract.

M5 is not complete: final confirmation was not evaluated by this new experiment, but no historical period is certified as previously unexamined. No real-time predictive validation has been completed.

M6 remains unauthorized: do not merge, promote, alter the production classifier, change current scores or deploy this research branch.

## Reproduction

```bash
python -m pytest tests/test_forecasting*.py -q
python -m src.forecasting.discovery_execution \
  --snapshot /path/to/verified/M2/live \
  --output /path/to/new/research-output
```

The output directory must be new. The workflow pins the M2 archive digest, verifies the internal ledger, records code hashes and retains all outputs. Main outputs: all-predictor admission ledger, discovery spectrum/loadings/residuals, complete exact-profile groups, inner tuning records, out-of-fold forecasts, matched metrics and per-entity error diagnostics. These are research evidence, not publishable live forecasts.
