# PRD v0.7 — Phase 3: probabilistic future-state forecasting

Date: 2026-09-23  
Status: active research requirements  
Branch: `research/broad-feature-phase3-probabilistic-2026-09-23`  
Parent Phase 2 head: `a38075a0eee52d50f28d33034c24003895386a4b`  
Production changes: none authorized

## 1. Purpose

Phase 2 established a broad, missing-aware 96-dimensional banking-system state and showed that its one- and two-year movement can be forecast more accurately than assuming no change in retrospective development tests.

Phase 3 converts those point forecasts into **probability distributions** and coherent peer simulations.

The intended output is not only:

> This country is expected to move to point X.

It is:

> This is the most likely future state, this is the plausible range around it, this part of the uncertainty comes from limited current information, this part comes from unpredictable future movement, and these are the peer and historical paths that provide context.

No named banking ratio, crisis outcome, production pillar or feature family defines the Phase 3 state or forecast.

## 2. Fixed Phase 2 inputs

Phase 3 starts from the completed Phase 2 research evidence:

- 96-dimensional fixed-loading measurement state;
- 8,783 country-year states;
- 14,652 eligible feature representations;
- country-year information and uncertainty diagnostics;
- measurement loadings, feature reliability and robust scaling metadata;
- registered one- and two-year transition model comparisons.

The Phase 2 measurement state is treated as fixed for this phase. Phase 3 does not refit the broad measurement architecture merely to improve probabilistic forecast metrics.

## 3. Point-forecast models

Phase 3 uses the point-model families selected by Phase 2 development evidence:

### One-year horizon

Use the regularized full-state change model:

\[
\widehat z_{t+1} = z_t + f_1(z_t)
\]

The Phase 2 “reduced-rank” winner selected all 96 output dimensions and was numerically identical to the full ridge state-change model. Phase 3 therefore describes it accurately as the **ridge state-change model**, rather than claiming dimensional reduction that did not occur.

Registered penalty: `alpha = 100`.

### Two-year horizon

Use the coordinate-wise regularized autoregressive model:

\[
\widehat z_{j,t+2} = a_j + b_j z_{j,t}
\]

Registered penalty: `alpha = 0.1`.

The model families and penalties are fixed from Phase 2. Phase 3 does not retune them against its own forecast-distribution results.

## 4. Rolling historical forecast errors

Phase 3 must generate expanding-window, out-of-time point forecasts across the available history.

For each eligible forecast-origin year:

1. train only on earlier exact-calendar state transitions;
2. forecast the complete later state;
3. convert predictions back to the common Phase 2 state coordinate system;
4. retain the complete residual vector;
5. record country, origin year, target year, horizon, current-state uncertainty and observed-information measures.

These rolling errors are used to estimate forecast uncertainty. They are retrospective because the Phase 2 measurement state itself was estimated from the latest-vintage panel; they are not described as vintage-clean real-time predictions.

## 5. Three uncertainty components

Phase 3 separates uncertainty into three conceptually different sources.

### 5.1 Current-state identification uncertainty

The present country state is estimated from incomplete and noisy observations.

Use the Phase 2 artificial-masking evidence to calibrate a monotone mapping from the country-year state-uncertainty proxy to expected state-estimation error.

This is uncertainty about **where the country is now**.

### 5.2 Shared future shock uncertainty

Countries can move together because of global or widespread conditions.

For each historical target year, decompose forecast residuals into a cross-country mean residual vector. In joint simulation, one common residual draw is shared across all countries in a scenario.

This is uncertainty about **future shocks that affect many banking systems together**.

The number of historical common-shock years is limited. Use empirical year-block resampling and report that limitation; do not estimate an unrestricted 96 × 96 common-shock covariance from a handful of years.

### 5.3 Country-specific transition uncertainty

After removing the target-year common residual, retain each country-period's remaining residual vector.

In simulation, country-specific residual draws are sampled independently across countries but preserve the observed cross-dimensional residual pattern within each draw.

Residual sampling should be conditioned on broad information quality where sample size permits, using state-uncertainty or observed-information groups. Conditioning must not create tiny unstable buckets; fall back to the broader horizon pool when necessary.

## 6. Empirical simulation rather than assumed normality

The first Phase 3 model uses empirical residual resampling rather than assuming all 96-dimensional forecast errors are multivariate Gaussian.

Each future-state simulation combines:

1. a draw around the current estimated state;
2. the fixed Phase 2 point-transition model;
3. one shared historical shock draw for all countries;
4. one country-specific residual-vector draw per country.

This preserves skewness, fat tails and cross-dimensional residual relationships present in historical errors.

A parametric Gaussian/state-space challenger may be added later, but it is not the default merely for mathematical convenience.

## 7. Calibration tests

Probabilistic forecasts must be evaluated separately from point accuracy.

### 7.1 Leave-one-target-year-out calibration

When assessing historical target year Y, construct the forecast-error distribution without using residuals from Y.

Report at minimum:

- 50%, 80% and 95% joint-state radial coverage;
- average interval/radius width;
- coverage by horizon;
- coverage by current-state uncertainty quartile;
- coverage by data-coverage quartile;
- point RMSE and movement-direction similarity;
- distribution of probability integral / rank diagnostics where practical.

### 7.2 No-change comparison

The probabilistic transition forecast must be compared with an empirical no-change distribution constructed under the same calibration design.

A narrower interval is not automatically better. Coverage and sharpness must be reported together.

### 7.3 Failure rule

If the registered 80% and 95% regions materially under-cover in the leave-year-out test, Phase 3 must inflate/calibrate the distributions or stop. It must not publish overconfident probability cones.

## 8. Latest one- and two-year forecasts

Fit the fixed point models using all eligible historical transitions through the latest Phase 2 state.

For every sufficiently supported latest country state, export:

- current state year;
- forecast year;
- point future state;
- simulated state draws or reproducible simulation seed/parameters;
- 10th, 25th, 50th, 75th and 90th percentile for every state coordinate;
- median and 80%/95% radial movement range;
- probability of moving farther from versus closer to the contemporary peer center;
- current-state information quality;
- forecast-distribution quality flag.

A Phase 3 forecast is a research output, not a risk rating.

## 9. Joint peer simulation

All countries in a scenario share the same common-shock draw.

For each country report:

- distribution of future state-distance percentile relative to peers;
- probability of relative percentile improvement/deterioration;
- likely future nearest peers and the frequency with which each appears;
- probability that the current nearest peer remains the nearest peer;
- expected absolute movement versus relative movement.

Relative position is not risk direction unless a later supervised interpretation layer establishes that relationship.

## 10. Historical analogue explanation

Analogues remain secondary to the point/probabilistic transition engine.

For each latest country state:

1. find earlier country-years with similar state and recent velocity;
2. exclude future information from analogue selection;
3. require minimum information support;
4. report analogue distance, source year and uncertainty;
5. report the analogue's realized one- and two-year state movement where available.

Analogue outcomes are explanatory precedents, not the main probability generator.

## 11. Observable implications

Phase 3 may translate forecast state changes through the Phase 2 measurement loadings.

Because the preserved Phase 2 evidence contains state loadings and feature scales but does not persist every feature intercept, the first Phase 3 output is **change-based**:

\[
\Delta \widehat x_j \propto \lambda_j^\top \Delta z
\]

Export expected standardized feature-representation changes and their uncertainty. Do not claim exact future raw indicator levels where the inverse measurement mapping is incomplete or economically nonlinear.

Group observable implications by source and indicator metadata for analyst interpretation, without converting those groups into constraints on the underlying state.

## 12. Phase 3 acceptance criteria

Phase 3 closes when:

1. the fixed Phase 2 point models are reproduced from preserved evidence;
2. rolling out-of-time residual vectors are generated for both horizons;
3. current-state, shared-shock and country-specific uncertainty are separated;
4. leave-target-year-out 50%/80%/95% calibration is reported;
5. forecast distributions are calibrated or explicitly stopped;
6. joint latest peer simulations are exported;
7. analogue explanations and observable-change implications are exported;
8. every output is reproducible from immutable Phase 2 inputs and declared seeds;
9. no supervised banking/crisis target defines the state or forecast;
10. production application, scores, classifier and serving artifacts are unchanged.

Phase 3 completion is still retrospective research evidence. It does not authorize production integration.

## 13. Execution policy

Use local execution against the preserved Phase 2 evidence while developing.

Do not use GitHub Actions for iterative debugging. One consolidated repository checkpoint may be used only after Phase 3 is closure-ready and after reviewing whether it is necessary.

No fresh IMF or World Bank retrieval is required for Phase 3 architecture work.
