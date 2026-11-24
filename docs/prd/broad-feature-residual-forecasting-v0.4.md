# PRD amendment v0.4 — forecast changes, preserve breadth, control duplicate influence

Date: 2026-09-22. Applies with v0.1-v0.3 on `research/broad-feature-forecasting-2026-09-16`. This amendment is committed before its implementation and fitting. PR #28 remains draft. No production merge, deployment, crisis-classifier retraining or serving-score change is authorized.

## Why this stage

The v0.3 target-independent discovery confirms that the broad dataset cannot be represented adequately by two imposed pillars: 103 components are required for 90% of transformed cross-sectional variance at the 2023 reference, with 9,644 varying inputs and 4,292 exact transformed profiles. The first direct level forecasts using broad ridge and PCA+residual ridge did not beat persistence on aggregate MAE for any of the six target/horizon pairs. This negative result is evidence about those specifications, not evidence that the broad information set is useless.

Persistence is structurally strong for banking ratios. The next development question is therefore whether the broad information set explains the *future change away from the current observed target*, while reducing accidental over-weighting of duplicated source representations. Do not reduce the input universe to a hand-picked list.

## D16 — residual-to-persistence target

For each registered target/horizon and matched cohort, define the supervised response as:

`future observed target - same-series persistence value at the forecast origin`.

Persistence remains the zero-change forecast. Broad models predict the change, and the level forecast equals persistence plus predicted change. The same observed target, target-break exclusions, outcome-availability purging and retrospective lag policy from v0.3 remain binding. No imputed outcomes.

Report MAE/RMSE/bias in original target units for the reconstructed level prediction, and separately report change-prediction error. Do not change the cohort to make a model look better.

## D17 — no feature cap; fold-local learnability only

All model-eligible predictor representations remain available. Inputs may be excluded within a training fold only because they are:
- entirely missing;
- constant after the registered transformation;
- unresolved-unit/quarantined under the v0.2 contract; or
- exact duplicate transformed profiles in a *sensitivity* that retains group membership/provenance.

No top-k feature selection, univariate target screening or manual economic allowlist. A source/family with many correlated series must not receive extra influence merely because it publishes more columns.

## D18 — profile-balanced and source-balanced regularization sensitivities

Run three broad change models on identical rows:
1. **broad residual ridge** — all learnable representations under the existing training-only rank/amount normalization;
2. **exact-profile balanced ridge** — keep every identity in the ledger, but scale each transformed predictor by `1/sqrt(group_size)` for exact training-profile duplicate groups before ridge so a repeated representation cannot multiply influence solely through duplication;
3. **source-balanced ridge** — scale each predictor by `1/sqrt(number of learnable predictors from that source)` before ridge, as a sensitivity to database-size dominance.

These are regularization/weighting sensitivities, not feature deletions. Export the scaling ledger so every original identity remains traceable. Compare all three with persistence and the prior broad-level model.

## D19 — data-selected common-state residual

Retain the v0.3 PCA common-state representation, with component count determined by the registered variance criterion rather than two pillars. Add a residual-to-persistence specification using:
- data-selected common components; and
- the direct residual feature channel under ridge shrinkage.

The direct residual path remains broad and is not restricted to high-loading features. Component variance is descriptive; predictive value is judged only on forward development windows.

## D20 — experiment governance

Use the same outer windows as v0.3 (2016-2018 and 2019-2021), same matched rows, and same inner-only penalty selection rule. Do not inspect 2022-2023 outcomes as confirmation. Record negative results unchanged.

Primary comparison remains level MAE versus persistence on identical rows. Secondary: RMSE, bias, change RMSE, and fraction of rows where absolute error improves versus persistence. Report source/profile scaling effects and learned coefficient concentration, but do not convert coefficients into causal importance.

This stage remains retrospective development evidence. No production candidate is created even if a challenger improves.
