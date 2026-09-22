# Broad-feature forecasting: execution status

Updated 2026-09-22. Active branch: `research/broad-feature-phase2-transition-2026-09-22`. Parent research PR #28 remains draft. Production baseline `5957ca779dafa21f2e098c819bfb060f43243206` is unchanged.

## Governing requirements

The active requirements are:
- original broad-feature PRD;
- source-grounded data contract v0.2;
- target-independent discovery plan v0.5;
- Phase 2 PRD v0.6.

The owner requirement remains binding: no variable, source, feature family, target or outcome receives artificial emphasis or artificial exclusion merely because it appeared in an earlier model.

## Completed phases

### Phase 0 — canonical data foundation
Complete enough for current research use. Source identities, dimensions, units, status, conflicts and exact annual endpoints are retained with retrospective-vintage caveats.

### Phase 1 — target-independent structural discovery
**Complete.** Validated run `35727510691`; artifact `10694820381`; ZIP SHA256 `e561ab5caa7896ea17172b98b8d85f9bd4269533a4da4702ef0a72dd66c4eb43`.

Key results:
- 46 annual origins through 2026;
- 15,470 predictor representations;
- 10,098 learnable representations in the 2026 reference;
- 68 / 97 / 120 components for 80% / 90% / 95% transformed variance;
- first component share about 13.2%;
- frozen-PCA geometry was too strongly related to data coverage to use directly as the transition state.

### Phase 2A — missing-aware measurement state
**Complete on retrospective development evidence.**

Real-data evidence comes from the completed modelling outputs in run `35745530960`, artifact `10702494849`, SHA256 `5f841b497df55e3498bc799231deffad773caf578939a41fc5c11dbee6c49f7e`. The workflow failed only after modelling, when writing JSON containing a NaN metric; the serialization bug is fixed on the branch without rerunning the models.

Measurement-state result:
- 15,470 representations registered;
- 14,652 eligible model features;
- 4,539,769 genuinely observed cells;
- 0 missing cells imputed into the fit objective;
- 8,783 country-year states;
- selected state rank: **96**;
- raw best held-out reconstruction rank: **128**;
- rank search extended through **320** and no longer ended at the boundary;
- selected measurement mode: **balanced reliability**;
- held-out reconstruction improvement versus zero-state baseline: about **9.9%**.

The Phase 1 coverage problem was materially reduced:
- state distance vs observed coverage: **0.948 → 0.133** Spearman;
- movement vs absolute coverage change: **0.357 → 0.093** Spearman;
- state uncertainty vs coverage: **-0.960** Spearman, as expected when richer information tightens state identification;
- artificial-masking calibration: state error vs uncertainty **0.712** Spearman.

All Phase 2A gates passed.

No supervised banking target, crisis label, production score or production pillar was used to construct the state.

### Phase 2B — target-independent state transitions
**Complete on retrospective development evidence.**

The transition layer forecasts the complete 96-dimensional learned state rather than selected banking variables.

Development design:
- one- and two-year exact-calendar state transitions;
- two outer development windows: 2016–2018 and 2019–2021;
- strict earlier training/inner tuning;
- 1,271 matched development transitions per horizon;
- no-change baseline;
- diagonal autoregression;
- ridge delta;
- reduced-rank delta;
- shared/global-state ridge delta;
- historical-analogue delta.

Aggregate development results:

**1-year horizon**
- no-change state RMSE: **1.2254**
- best challenger: **reduced-rank delta** (numerically tied with full ridge delta)
- best state RMSE: **1.0130**
- relative RMSE improvement: **17.3%**
- challenger beats no-change on about **86.6%** of country-period rows
- average movement-direction cosine: about **0.54**

**2-year horizon**
- no-change state RMSE: **1.3278**
- best challenger: **diagonal autoregression**
- best state RMSE: **1.1032**
- relative RMSE improvement: **16.9%**
- challenger beats no-change on about **90.2%** of country-period rows
- average movement-direction cosine: about **0.54**

The historical-analogue transition model also beat no-change in aggregate but materially underperformed the regularized transition models. That means analogue paths remain useful as an explanatory layer, not the primary forecasting engine at this stage.

These results are retrospective development evidence, not an untouched real-time confirmation. Phase 2A loadings are fitted using the retrospective research panel, so Phase 2B is not yet a vintage-clean historical forecast exercise.

## Phase 2 interpretation

Phase 2 supports the architecture proposed after Phase 1:
- use the broad data to estimate a stable missing-aware state;
- distinguish descriptive state dimension from forecastable dynamics;
- forecast the state rather than a hand-picked outcome list;
- keep historical analogues as a secondary explanatory distribution;
- move next to probabilistic future-state forecasting and uncertainty, subject to later vintage-clean confirmation.

The important empirical result is that the learned state has non-trivial persistence/dynamics beyond simple no-change: regularized transition models improve state RMSE by roughly 17% at both one- and two-year horizons in the registered development windows.

## Next phase

Phase 3 should build probabilistic one- and two-year future-state distributions from the Phase 2 state/transition architecture, including:
- forecast uncertainty;
- shared/global shocks;
- country-specific transition uncertainty;
- reconstructed observable implications;
- relative peer-state simulation;
- historical analogue paths as explanation, not primary forecast.

A future clean confirmation design remains necessary before any production proposal.

## Production firewall

Phase 2 did not alter:
- production crisis classifier;
- current serving risk model;
- current pillar pipeline;
- production source caches;
- Streamlit production code;
- deployed country scores.

No merge, promotion or deployment is authorized.
