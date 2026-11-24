# PRD amendment v0.3 — broad discovery and forecasting are separate tasks

Date: 2026-09-22. Owner instruction: continue execution on the breakaway branch; do not restrict predictors to a few banking features; allow the data to identify variance and distinctness. Applies with v0.1 and the v0.2 source contract. PR #28 remains draft; no production changes, classifier retraining, merge or deployment are authorized.

## Correction of emphasis

The three existing TARGETS in `panel.py` are observable outcomes for supervised experiments, not the predictor list. The constructor emits every observed registered feature as levels, calendar lags and changes. This breadth must remain. The existing broad ridge library is also uncapped. What is missing is an explicit target-independent discovery path. Add it rather than treating the three outcomes or the old economic/industry pillars as the only definition of useful information.

Maintain three separate quantities: (1) variance/structure explained by a representation, (2) how an observation differs from its peers, (3) predictive skill on later outcomes. None proves another. A loading is not a causal effect, a distinctive country is not necessarily risky, and low-variance predictors must remain eligible in the direct forecasting path.

## Frozen input and provenance

Use the completed September 16 research retrieval/panel from run 35134477499, artifact 10463106988, ZIP SHA256 `f670f2e7caff532201c1a198130d0f8b34cb04784cb643dae2c5466af9359c9b`. This includes the September 16 MFS payload, not just the September 15 serving cache. It is a frozen research vintage, NOT a new September 22 download. Validate the archive and its output-checksums ledger before reading model inputs. The artifact's universe label predates the branch clarification: 214 feature-store entities are not the 201 published-score cohort. Report supplied membership explicitly, preserve non-member context separately, and never count group aggregates as extra independent countries.

The retrospective lag remains one calendar year. Historical release/vintage dates are unverified. All findings remain retrospective development evidence, never an untouched real-time confirmation.

## D13 — Uncapped discovery library

All 5,318 source identities remain registered, including quarantined/unsupported series; every supplied predictor representation is audited. No top-k features, positive-correlation gate, hand-picked economic family or minimum global country coverage. Fold-local all-missing or constant inputs cannot inform a fitted model; record their identities and reasons without deleting them from the library. Invalid numeric values, unresolved units and source break/status constraints are explicit admission gates, not a predictor-count budget. Do not admit raw identifiers, dates, outcome columns or serving scores as explanatory coordinates.

Common-currency and domestic-currency levels differ in comparability. For pooled discovery, currency amounts should be represented through within-entity, prior-history normalization, with unavailable histories reported; never assume converting each column to a global z-score makes different domestic currencies comparable. Keep raw monetary levels as a separately labelled diagnostic sensitivity only. Record original units and transformation policy. Never apply an undocumented SCALE multiplier.

Use training-only unit-insensitive rank/robust transforms and median filling for the first exploratory representation, keeping missingness/coverage diagnostics separate from economic components. Missingness is a legitimate diagnostic but must not automatically be interpreted as economic differentiation. No sign restrictions, development anchor, existing two-pillar directions, positivity requirement or shrinkage toward positive equal weights is used in the discovery PCA.

## D14 — Data-selected representation, retained residual path

Compute the complete nonzero spectrum in row space when wide matrices make feature covariance inefficient. Report cumulative variance and the numbers of components required for 80%, 90% and 95% of the transformed-input variance; do not fix two components or a feature cap. The initial descriptive representation uses the predeclared 90% criterion, with 95% sensitivity. This is a compression parameter, not proof of a true latent-state count. Export all component loadings and all feature contributions, not just a top-ten selection. Display summaries may show leading contributors without removing the rest.

Export separate common-component and residual distinctness diagnostics, feature-level residual contributions, and coverage/imputation shares. The residual channel remains eligible for forecasting; it is not discarded simply because PCA explains little of it. Fit transformations on a declared reference sample and apply frozen transformations to comparisons. Test row/column permutation, pickle round-trip, future-only features, unit rescaling, duplicates and low-variance predictive signals. Include a redundancy sensitivity that collapses exact training-profile duplicates, with the provenance/members retained; do not call repeated frequency/currency versions independent corroboration.

## D15 — First development experiments (registered before fitting)

1. Target-independent discovery: use the panel's latest common forecast-origin year (2023, observations through 2022 under the lag), plus the preceding year transformed on the same reference, explicitly retrospective. Fit on all supplied member entities with available eligible inputs; do not subset to those with observed future NPL/capital/liquidity outcomes. Report absence/coverage instead of silently discarding sparse entities.
2. Initial forecasting development: the three registered outcomes, horizons one and two, with forecast origins 2016-2018 and 2019-2021 as separate outer DEVELOPMENT windows. Training origins precede each window and their assumed target-availability dates must be strictly before the window start. Outcomes for 2022-2023 origins are not evaluated by this experiment. Those years are not certified as never previously inspected.
3. Comparators on identical rows: persistence, compact ridge (target's own level/lag/change only), broad ridge, and a structured broad model with data-selected principal components plus a regularized direct/residual path. A compact comparator is not an input restriction on broad models.
4. Use a small preregistered ridge penalty grid `[10, 100, 1000]` selected on an inner forward development window ending before the outer start with the same label-purging rule. Minimum fitting sample of 20 rows is an estimation diagnostic, not a feature cap. Report any skipped target/fold rather than manufacturing results.
5. Primary diagnostic MAE in outcome units; also RMSE, bias, per-origin and per-entity errors, no-change comparator and observed-label counts. Report broad-minus-compact and broad-minus-persistence on matched samples. No target clipping or outcome winsorization based on validation values. Exclude explicit target breaks; report that cohort choice. Use published observed outcomes only, no imputed labels. Directional accuracy and probabilistic uncertainty are not claimed unless separately implemented and validated.
6. Record exploratory results even when broad models lose. No feature search or repeat tuning in response to outer development results in this registered run. This run does not decide a production winner or evaluate the final confirmation period.

## Execution acceptance and scope

Run the new synthetic tests, the existing research suite, full repository checks and the real frozen-panel job. Confirm unchanged source hashes and production paths. Deliver an uncapped admission ledger, preprocessing audit, spectrum/loadings, source contribution summaries, distinctness/coverage diagnostics, out-of-fold predictions, metrics, chosen parameters and code/environment/input hashes. Publish evidence on PR #28, update the stale implementation status, and distinguish implemented/executed discovery and baseline work from unimplemented nonlinear models, state-space transitions, hazard integration, calibrated intervals and joint peer forecasts.

Discovery and development training explicitly authorized by this instruction do not alter or retrain the selected production crisis classifier. The existing classifier/pillar scores and serving cutoff remain untouched.

References: scikit-learn PCA/PCR vs PLS example (variance need not predict targets), standard preprocessing documentation (feature scales/outliers), ridge documentation and common pitfalls (training-only preprocessing). These motivate controls, not this project's predictive accuracy.
