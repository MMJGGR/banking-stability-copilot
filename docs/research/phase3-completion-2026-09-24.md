# Phase 3 completion — probabilistic future-state forecasting

Date: 24 September 2026  
Branch: `research/broad-feature-phase3-probabilistic-2026-09-23`  
Status: **complete as retrospective research/development evidence**  
Production status: unchanged; no merge, promotion or deployment authorized

## What failed during development

The statistical analysis did not fail. The initial full local execution exceeded the notebook/process time window because the numerical libraries opened too many BLAS/OpenMP CPU threads while simultaneously holding the large joint-simulation arrays.

The same registered 800-draw execution completed successfully after limiting numerical libraries to two threads:

```bash
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
```

Completed execution profile:

- elapsed time: approximately **34.6 seconds**;
- peak resident memory: approximately **1.34 GiB**;
- latest simulation draws: **800 per horizon**;
- peer-neighbour draws: **250 per horizon**.

A deterministic runner, `scripts/run_phase3_research.sh`, now applies these limits. This was an execution/resource-control issue, not evidence that the model, dataset or probability design failed.

## Phase 2 inputs reproduced

Phase 3 kept the completed 96-dimensional Phase 2 measurement state fixed.

The one- and two-year point models were reproduced from preserved Phase 2 evidence with maximum absolute metric difference:

`1.607329824793169e-10`

This establishes that Phase 3 started from the registered Phase 2 models rather than silently retuning them.

## Rolling forecast evidence

- rolling out-of-time forecast rows: **15,290**;
- distinct target years: **40**;
- leave-target-year-out calibration rows: **30,580**;
- latest country states simulated: **213**;
- state dimensions: **96**;
- provider-projection rows read by the baseline: **0**.

The registered point models remained:

- one year: ridge state-change model, `alpha = 100`;
- two years: coordinate-wise regularized autoregression, `alpha = 0.1`.

## Probabilistic calibration

Probability regions were estimated using historical residual vectors and assessed by leaving each target year out of its own calibration pool.

### One-year horizon

| Model | 50% coverage | 80% coverage | 95% coverage | Mean 80% radius | Mean 95% radius |
|---|---:|---:|---:|---:|---:|
| No change | 49.94% | 80.03% | 94.95% | 0.4219 | 0.5393 |
| Phase 3 transition | **50.15%** | **79.95%** | **94.95%** | **0.3176** | **0.3988** |

At essentially the same empirical coverage, the transition forecast’s regions were:

- approximately **24.7% narrower** at 80%;
- approximately **26.0% narrower** at 95%.

### Two-year horizon

| Model | 50% coverage | 80% coverage | 95% coverage | Mean 80% radius | Mean 95% radius |
|---|---:|---:|---:|---:|---:|
| No change | 49.98% | 79.85% | 94.94% | 0.4439 | 0.5582 |
| Phase 3 transition | **49.78%** | **79.93%** | **94.91%** | **0.3595** | **0.4480** |

At essentially the same empirical coverage, the transition forecast’s regions were:

- approximately **19.0% narrower** at 80%;
- approximately **19.8% narrower** at 95%.

All registered 80% and 95% calibration acceptance gates passed.

## What Phase 3 produces

For each sufficiently supported latest country state, Phase 3 exports:

- fixed-model one- and two-year point forecasts;
- coordinate-level 10th, 25th, 50th, 75th and 90th percentiles;
- 50%, 80% and 95% state-movement radii;
- probability of relative peer-percentile improvement or deterioration;
- probability of moving closer to or farther from the contemporary peer centre;
- likely future nearest peers and their simulation frequencies;
- historical analogue paths;
- standardized observable implications through the Phase 2 measurement loadings.

The joint simulations use one shared historical shock draw across countries in a scenario plus country-specific residual draws. This preserves common cross-country shocks rather than forecasting every country independently.

## Uncertainty treatment

The output distinguishes:

1. uncertainty about the current estimated state;
2. shared future shocks affecting many countries;
3. country-specific transition uncertainty.

The first implementation uses empirical historical residual resampling rather than assuming a 96-dimensional normal distribution.

## WEO 2026–2031 projections

Current-vintage WEO provider projections are excluded from:

- state estimation;
- transition fitting;
- calibration;
- realized outcomes;
- baseline Phase 3 simulation.

They are retained separately as optional provider scenarios/benchmarks under PRD amendment v0.8. No WEO-conditioned scenario was executed in this baseline phase.

## Tests and boundaries

Local targeted tests passed:

- Phase 3 probabilistic tests: **4 passed**;
- provider-projection separation tests: **3 passed**.

Verified boundaries:

- selected banking targets read: **0**;
- crisis labels read: **0**;
- production scores read: **0**;
- provider-projection rows read by baseline: **0**;
- production files changed: **0**;
- production classifier retrained: **false**;
- final untouched confirmation evaluated: **false**.

No GitHub Actions workflow was used for iterative Phase 3 development.

## Interpretation

Phase 3 extends the Phase 2 architecture from point predictions to calibrated probability distributions. The transition model provides materially narrower uncertainty regions than a no-change forecast without sacrificing empirical coverage in the retrospective development design.

This supports the architecture:

**broad observed data → missing-aware state → learned transition → calibrated future-state distribution → coherent peer simulation**

## Limitations

- The state remains based on latest-vintage retrospective history, not vintage-clean real-time data.
- Calibration is development evidence, not an untouched final confirmation.
- Current-state perturbation directions use empirical transition-residual directions because Phase 2 retained masking-error magnitudes but not full masking-error vectors.
- Observable implications are standardized directional changes, not exact raw future indicator levels.
- The 96 internal dimensions are not intended as 96 user-facing risk factors.
- Nothing in this evidence authorizes production integration.

## Evidence hashes

- `phase3-summary.json`: `231ce7bc803b5136934b7f04409c593bbcfa0b9bebdf1471475f3ba93c9dc05b`
- `output-checksums.json`: `29603f4d95bfa1b5547a988ca7b70319eb5f1f63c1be16ff4aae0fd506f5c9a2`
