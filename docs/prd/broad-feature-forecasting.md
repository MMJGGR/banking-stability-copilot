# PRD: Broad-feature, time-aware banking-system forecasting

Version: 0.1 | Written: 2026-09-16 | Owner: Richard Macharia
Status: research requirements drafted from the owner's instructions; implementation authorized on an isolated branch; no production/model promotion authorized by this PRD.
Branch: `research/broad-feature-forecasting-2026-09-16`
Baseline: `5957ca779dafa21f2e098c819bfb060f43243206` (`master`, verified before branching).

## 1. Purpose and decision

Add a modular forecasting research layer to the Banking Stability Copilot. Use the full eligible information universe to learn how banking systems evolve over one and two years, while keeping current-condition assessment, future fundamentals, crisis-onset probability, and relative peer position distinct.

The owner's explicit requirement is broad information access, not an artificially small feature list. A compact model is a comparator, never the maximum permitted information set. Control unsupported model influence through regularization, structure and forward validation. Do not equate more columns with more independent evidence, or presume that a smaller model must win.

This PRD is committed before implementation. The initial delivery is a documented, executable research foundation with tests and source-coverage evidence. The complete forecasting system has subsequent milestones; a passing foundation is not evidence of forecast skill.

## 2. Existing system and reuse boundaries

The baseline serves a cross-sectional two-pillar assessment, KNN imputation, policy adjustments and a preserved legacy logistic crisis overlay. Its recent release establishes artifact integrity and computational reproducibility, not predictive recertification. `artifacts/crisis_validation_summary.json` marks legacy validation `invalid_superseded`.

Reuse and extend, rather than silently replace:
- `src/crisis_panel.py`: country-year origins, feature cutoffs, active-crisis exclusions and censoring contracts.
- `src/crisis_hazard.py`: conditional annual hazard, cumulative incidence and evidence-confidence separation.
- Existing validation/research modules: grouped/forward evaluation, calibration and threshold governance.
- `src/fsic_selection.py`: canonical-series identity and fail-closed ambiguity principles. A broad registry must not be restricted to its current serving feature list.

No modification of `app.py`, serving caches, production manifests, classifier weights, existing score policy or deployment workflows is required for this research foundation. New code belongs in an isolated `src/forecasting/` package and explicitly invoked research scripts. Outputs go to a separate research directory, never `cache/` or serving snapshot paths.

## 3. Users and jobs to be done

A credit/investment analyst needs to distinguish a currently vulnerable banking system from one likely to deteriorate, understand which observations support that outlook, identify historical analogues, and inspect missing-data and model uncertainty. A model reviewer needs to reproduce selection, transformations, splits, predictions and exclusions from immutable inputs. The owner needs evidence that broad information improves decisions beyond transparent baselines before any replacement of served outputs.

Illustrative future view (not a current capability): observed NPL ratio and its date; one-/two-year predictive distributions; separate systemic-onset probabilities; observed versus inferred drivers; historical analogues with subsequent outcomes; absolute risk movement versus relative peer movement; explicit insufficient-evidence state.

## 4. Scope and non-goals

In scope: all discoverable eligible source series; temporal feature library; one-/two-year fundamentals; shared global/regional dynamics; historical analogues; horizon-coherent hazards; uncertainty; fair broad-versus-compact experiments; analyst-facing research reports; reproducible lineage.

Not authorized: automatic production promotion, changes to the current classifier, unconditional forecasts for every country regardless of evidence, guarantees of crisis prediction, causal policy conclusions, or interpreting similarity as contagion. A neural network/backpropagation is an optional challenger, not a requirement or a claim of superior quality. No paid data purchases or new external-account connections are authorized.

Annual modelling is the initial broad-panel cadence. Genuine higher-frequency data may support later extensions. Repeating annual figures across quarters must not be counted as additional independent observations.

## 5. Functional requirements

### Data breadth, identity and availability

**F01 — Open feature universe.** Inventory all five currently bundled core sources (FSIC, FSIBSIS, MFS, WEO, WGI), and record available supplementary research sources separately. No hard-coded maximum predictor count, universal top-k filter, global complete-case requirement or blanket positive-univariate-correlation screen. Enumerate unused/quarantined series and the reason rather than quietly losing them.

**F02 — Canonical identities.** A feature identity includes source, indicator, economic dimensions, frequency, unit and scale when available. Retrieval timestamp and country identifier are metadata, not economic predictors. Different units/frequencies/dimensions remain distinguishable. Exact duplicates may coalesce with a count; conflicting values or missing identity dimensions require quarantine or an explicit governed rule. Preserve labels and lineage. Do not automatically merge distinct source measures because names resemble one another.

**F03 — Vintage-aware contract.** Track observation period, public availability/release date, revision/vintage date, retrieval date and status separately. Selection at origin t may use only values known by t. When historical release/vintage evidence is absent, label the experiment `retrospective_latest_vintage`; a conservative lag assumption does not convert it into a real-time backtest. Unknown release dates must never be relabelled as verified availability. Forward projections can be separately dated inputs, never realized outcomes.

**F04 — Breadth/quality audit.** Report country-by-year/feature-family coverage, history lengths, original frequencies, direct/estimated/imputed status, staleness, conflict counts and publication-date completeness. Count usable forecast-origin/target pairs and distinct crisis episodes, not merely raw rows. Structural inventory can inspect the complete source schema; learned eligibility and selection must use training data only.

**F05 — Missing data.** Retain ragged panels. Preserve missingness and age indicators where justified. Fit imputation and scaling on training data only; no backward fills from future observations or full-sample smoothing in predictors. Never treat imputed outcomes as observed labels. Scale the economic similarity space appropriately; do not reuse the serving pipeline's pre-scaling KNN distance without validation. Metadata must not enter neighbour distances accidentally.

### Targets, representations and learners

**F06 — Targets are not a feature cap.** Start development with observable banking outcomes such as future NPL, capital and liquidity measures. Target names/units and horizons must be registered explicitly; selecting a small initial target set does not restrict its predictors. Learn fundamentals first, not merely future versions of our own score. Crisis-onset targets use the existing governed label/censoring foundation and remain a separate task.

**F07 — Rich temporal representations.** Admit levels, lags, changes, volatility, own-history deviations and selected interactions, with documented economic definitions. All transformations must be calculable at the origin. Data-validity and fold-local identifiability rules are legitimate; arbitrary global predictor caps are not.

**F08 — Comparable model ladder.** Maintain persistence and compact regularized baselines; broad dense ridge; elastic net; common-factor plus direct-feature models; nonlinear boosted-tree challengers; and dynamic-factor/state-space transition candidates. Compare on identical country-origins first, and report additional coverage separately. Each target/horizon can select a different supported architecture. Reuse hazard models rather than duplicating event definitions.

**F09 — Reinforcement without forced influence.** Test a common-factor path and a direct/idiosyncratic path. Correlated indicators may reinforce a real common signal or duplicate errors; the model must be allowed to shrink redundant information. Do not extend the current positive equal-weight-shrunk score by simply appending all columns. Include group-removal/refit experiments because correlated predictors can hide one another's individual importance.

**F10 — Global context and trajectory matching.** Historical analogues are country-origin episodes, matched using conditions and recent dynamics within a training-fitted reference space. Their realized futures must have been available before the query origin. Shared macro/regional factors can connect forecasts. Explicit contagion graphs require observed, time-aligned economic exposures; similarity alone is insufficient.

**F11 — Coherent future space.** Freeze/align feature transformations and score mappings within evaluation vintages. Report absolute fundamentals separately from future relative rank. Whole-peer simulations must use coherent shared shocks, propagate uncertainty, and identify which countries could not be forecast. Do not claim joint forecasts when only independent marginal point forecasts exist.

**F12 — Uncertainty and abstention.** Separate measurement uncertainty, model uncertainty, scenario uncertainty and evidence coverage. Do not present a data-coverage percentage as a statistical confidence level. Return forecast intervals/distributions when validated; mark unsupported forecasts insufficient evidence. Hazard horizons must be coherent: cumulative two-year onset cannot be below cumulative one-year onset.

### Evaluation, reproducibility and governance

**F13 — Time-safe evaluation.** Split by forecast origin, not shuffled rows. Training labels must be observable before the validation origin; purge overlapping unresolved target windows and use explicit embargo/release lag. Country-/episode-group sensitivity and global-crisis clustering must be considered. Fit every imputer, scaler, selector, factor model, calibrator and tuning decision within the training/inner folds.

**F14 — Untouched confirmation.** Before target-dependent experiments, version the origin ranges, target definitions, outcome-availability policy, candidate set, metrics, tuning budget and final confirmation period. Previously inspected periods cannot be called untouched. Final confirmation must not drive feature/model/threshold revisions; failed confirmation requires a new research cycle, not retuning on the same holdout.

**F15 — Measures of success.** Fundamentals: MAE/RMSE, bias, directional skill where defined, interval coverage/width and performance versus persistence. Events: Brier/log loss, calibration, PR-AUC, event recall, false alerts per 100 country-years and alert burden. Relative ranks are secondary. Report country/region/income/coverage/era results and dependence-aware uncertainty. Synthetic results prove implementation behaviour only.

**F16 — Honest admission gates.** No production forecast is admissible solely because unit tests pass. Proposed release standard: improvement over persistence and the strongest compact comparator on the prespecified primary metric; uncertainty analysis that does not support a material aggregate degradation; acceptable calibration/alert burden and no unexamined material subgroup deterioration. Exact numerical tolerances and primary metrics must be frozen in an experiment plan before final evaluation. Inconclusive results remain research; the broad model is allowed to lose.

**F17 — Immutable evidence.** Record source checksums, code commit, feature-library version, target/split configuration, exclusions, model parameters, random seeds, environment, out-of-fold prediction ledger, scores and limitations. Read source snapshots without unpickling model artifacts just to audit raw data. New research outputs must not overwrite existing outputs silently. Compare production paths against the branching baseline.

**F18 — Separate release authority.** Keep a draft PR. Any future live integration needs named owner approval, complete model-validation evidence, compatible source-and-model bundle, clean regression checks and rollback. This PRD and branch are not that approval.

## 6. Technical deliverables and interfaces

- A source adapter/inventory layer creates stable identities without selecting a small economic feature set. All source formats must be inventoried; unsupported formats are explicit failures, not empty success.
- A canonical observation contract carries `country_code`, `feature_id`, `observation_period`, `value`, `available_at`, `vintage_at`, `status`, and source identity/lineage. Public release/vintage fields may be unknown and are never synthesized as verified facts.
- A research panel uses existing `forecast_origin_year` terminology, separates predictors from outcomes and retains exclusions. Adapters bridge the existing crisis research interfaces without altering them.
- Split utilities validate unique country-origins and label availability. Fold-local model pipelines accept wide/ragged numeric features with metadata separate.
- A registry of experiments separates compact/broad/structured candidate families. No candidate is automatically promoted.
- First outputs are human-readable Markdown plus machine-readable JSON/CSV. UI changes are outside the initial implementation.

## 7. Implementation milestones and completion evidence

| Milestone | Deliverable | Acceptance evidence | Initial status |
|---|---|---|---|
| M0 | This PRD committed before code; isolated branch pinned to baseline | Documentation-only first commit | This document |
| M1 | Full core-source inventory, canonical identity/conflict rules, availability audit | All five source formats; deterministic reordered-input tests; real snapshot audit; no source mutation | Not started |
| M2 | Broad temporal panel and development baseline harness | Leakage/revision/target-window tests; train-only preprocessing; persistence/compact/broad runs on identical origins | Not started |
| M3 | Registered broad-versus-structured experiments | Frozen plan; group ablations; coverage and computational diagnostics; development OOF ledger | Not started |
| M4 | Dynamic states, analogue layer and hazard research integration | Horizon consistency, filtered-not-smoothed predictors, uncertainty and subgroup checks | Not started |
| M5 | Confirmation, analyst report and independent review | Untouched forward evidence and documented pass/fail; explicit limitations | Not started |
| M6 | Optional live integration | Separate owner-approved promotion and rollback | Not authorized |

Work should close the earliest demonstrable milestone, with an implementation-status file mapping requirement IDs to tests/artifacts. Do not label M2-M5 complete because an inventory or prototype exists. A branch is not a background job: report completed work and remaining milestones at each handoff.

## 8. Acceptance cases for the foundation

1. Adding a valid new series increases the registry without editing a serving feature allowlist.
2. Adding duplicate rows or permuting source rows does not change selected values; unresolved conflicts are visible and never resolved by last-row order.
3. Same indicator code with different units or economic dimensions remains separate.
4. A revision published after a forecast origin cannot alter a verified-vintage feature for that origin.
5. Unknown availability is rejected in real-time mode and retained only with an explicit retrospective designation.
6. Future-target/projection metadata cannot enter ordinary realized predictors or outcomes by accident.
7. Training preprocessing is unchanged when validation values change; labels extending into validation are purged.
8. A feature unavailable throughout training is identified as unlearnable in that fold, not deleted from the global library.
9. Source hashes, serving files and production `master` remain unchanged after research execution.
10. An incomplete source inventory or failed statistical experiment cannot report success/promotion readiness.

## 9. Risks and open decisions

- Rich raw counts do not establish independent episode count or adequate feature histories. The earlier eight-variable screen is contextual only and is not a feature cap.
- Recent-vintage historical records may contain revisions unavailable in real time. Retrospective findings must be labelled accordingly.
- Normalized caches may have lost source dimensions. Conflicting series need raw metadata recovery rather than invented frequency precedence.
- Sparsity/definition changes may vary sharply by country and era; broad models must not silently evaluate only well-reported countries.
- Dense models can overfit; compact models can miss dispersed effects. Resolve with matched forward experiments.
- Preexisting crisis-label validation limitations and previously failed forward tests remain binding, not erased by a new architecture.
- Final target roster, publication-lag policies, held-out years and operational alert costs require an experiment-plan decision before final evaluation, not before the isolated foundation is built.

## 10. References and authority

Owner instructions in this conversation are authoritative for breadth, isolation and PRD-first sequencing. Repository references above are pinned conceptually to the stated baseline; the data manifest is the authority for the served snapshot, not potentially stale prose.

Implementation references (consulted 2026-09-16):
- Scikit-learn, leakage and preprocessing: https://scikit-learn.org/stable/common_pitfalls.html
- Scikit-learn, ridge/elastic-net mechanisms: https://scikit-learn.org/stable/modules/linear_model.html
- Statsmodels, large/mixed-frequency dynamic-factor framework: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html

These references motivate candidates and controls; they are not evidence that this project's forecasts are effective.
