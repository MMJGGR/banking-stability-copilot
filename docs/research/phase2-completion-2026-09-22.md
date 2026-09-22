# Phase 2 completion — missing-aware state and target-independent transitions

Date: 2026-09-22  
Branch: `research/broad-feature-phase2-transition-2026-09-22`  
Status: **complete as retrospective research/development evidence**  
Production status: unchanged; no merge, promotion or deployment authorized

## Evidence provenance

- Immutable research-data run: `35134477499`, artifact `10463106988`, ZIP SHA256 `f670f2e7caff532201c1a198130d0f8b34cb04784cb643dae2c5466af9359c9b`.
- Initial rank-boundary run: `35744506248`, artifact `10701178100`, ZIP SHA256 `a1df1c17c326b889453918a71ba9c4a072c295e0c9c4c5ad37bff22e5a610ae7`.
- Completed computation run: `35745530960`, artifact `10702494849`, ZIP SHA256 `5f841b497df55e3498bc799231deffad773caf578939a41fc5c11dbee6c49f7e`.
- Targeted Phase 2 tests in the completed computation run: **15 passed**.
- The source run ended after all model outputs were produced because strict JSON serialization rejected an undefined `NaN` direction statistic for the no-change comparator. The branch now converts undefined metrics to JSON `null`. Finalization reused the completed CSV/model outputs; no model was refitted.

## Boundaries verified

- No selected banking target entered the state or transition model.
- No crisis label or crisis-classifier output entered the state or transition model.
- No production risk or pillar score entered the state or transition model.
- No feature-count cap was imposed.
- No production serving file was changed.
- The selected production crisis classifier was not retrained or modified.
- Git comparison with production baseline `5957ca779dafa21f2e098c819bfb060f43243206` shows only added research workflows, PRDs, research code and tests.

## Phase 2A — missing-aware measurement state

The model used genuinely observed cells only. Missing values did not enter the objective as medians or zeros.

### Selected state

- State dimensions selected: **96**.
- The rank search was extended through **320** dimensions after the first search ended at its boundary.
- The smallest state within the registered tolerance of the best held-out reconstruction remained **96 dimensions**.
- Measurement mode: **balanced reliability weighting**.
- Country-year state rows: **8,783**.
- Model-eligible feature representations: **14,652**.
- Observed model cells: **4,539,769**.
- Missing values imputed before fitting: **0**.

### Reconstruction and coverage

- Held-out reconstruction improvement over the zero-state baseline: **9.93%** on mean row RMSE.
- Phase 1 state-distance versus coverage correlation: **0.948**.
- Phase 2A state-distance versus coverage correlation: **0.133**.
- State movement versus absolute coverage change: **0.093**.
- State uncertainty versus observed coverage: **-0.960**, in the intended direction.

The main Phase 1 problem was therefore materially reduced: the country-state geometry is no longer principally an observation-density map.

### Direct uncertainty check

The state was re-estimated after hiding 30% of observed cells for 400 sampled country-years.

- State-estimation error versus uncertainty proxy, Spearman: **0.712**.
- Mean masked state error: **0.215**.
- Mean uncertainty proxy: **0.457**.

All registered Phase 2A gates passed.

## Phase 2B — target-independent state transitions

The transition models forecast the complete learned state, not selected ratios. They were compared on later development periods with exact one- and two-year state pairs.

Development windows:
- 2016–2018;
- 2019–2021.

Evidence volume:
- **24** model/fold summaries;
- **15,252** matched prediction rows;
- **172** inner tuning comparisons;
- no skipped registered cases.

### One-year state forecast

| Measure | No change | Best challenger |
|---|---:|---:|
| Model | — | Reduced-rank state-change model |
| State RMSE | 1.2254 | **1.0130** |
| Relative RMSE improvement | — | **17.34%** |
| Country-years beating no change | — | **86.62%** |
| Average movement-direction cosine | — | **0.540** |

### Two-year state forecast

| Measure | No change | Best challenger |
|---|---:|---:|
| Model | — | Separate regularized autoregression by state coordinate |
| State RMSE | 1.3278 | **1.1032** |
| Relative RMSE improvement | — | **16.92%** |
| Country-years beating no change | — | **90.17%** |
| Average movement-direction cosine | — | **0.540** |

## Interpretation

Phase 2 supports three conclusions.

1. The broad information set can be represented as a stable, missing-aware state without recreating the severe coverage distortion identified in Phase 1.
2. The state remains materially high-dimensional: 96 dimensions were required even after testing substantially larger alternatives and selecting the smallest near-best state.
3. The state is not only descriptive. Simple regularized transition models forecast the complete future state materially better than assuming no movement over both one- and two-year horizons.

The results support proceeding to Phase 3: probabilistic future-state forecasts, uncertainty intervals and coherent peer-state simulations.

## Limitations

- This is retrospective latest-vintage research, not a real-time historical-vintage backtest.
- The development windows are not a final untouched confirmation period.
- The 96-dimensional state is a measurement representation, not 96 user-facing risk factors.
- Transition improvements do not yet prove calibrated future probability distributions.
- Historical analogues and named banking/crisis overlays remain later layers.
- Nothing in this evidence authorizes production integration.
