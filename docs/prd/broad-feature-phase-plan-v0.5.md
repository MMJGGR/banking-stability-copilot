# PRD v0.5 — Data-led architecture and natural research phases

Date: 2026-09-22
Status: active research requirements
Branch: `research/broad-feature-phase2-transition-2026-09-22`
Parent research branch: `research/broad-feature-forecasting-2026-09-16`
Production changes: none authorized

## Owner correction

No variable, indicator, feature family, risk measure, target or outcome receives artificial emphasis or artificial exclusion merely because it appeared in an earlier model or experiment.

The architecture begins from the full eligible information set. Data-validity rules may quarantine unusable observations; fold-local all-missing or constant features may be temporarily unlearnable; neither is an economic judgement about importance.

Earlier experiments that used a small number of supervised banking outcomes remain historical research evidence only. They do not define the architecture, predictor universe, output universe or research priority. PRD v0.4 residual-to-persistence work is deferred until a later supervised phase and must not drive Phase 1.

## Architectural objective

Build a banking-system state and transition architecture in which:

1. the information space is discovered from the broad dataset;
2. latent dimensions and country distinctness emerge from the data rather than prescribed pillars;
3. temporal structure and trajectories are established before supervised outcome selection;
4. future-state forecasting is multivariate / state based rather than anchored to a small hand-picked target list;
5. crisis/event overlays, named banking ratios and other supervised outcomes are introduced only later as validation or interpretation tasks when justified by data and use case;
6. production remains separate until clean forward validation and owner approval.

## Natural phases

### Phase 0 — canonical data foundation
Status: substantially complete.

Purpose:
- preserve source identities, dimensions, frequencies, units, transformations and source status;
- retain all eligible source identities;
- separate countries from aggregate/context entities;
- preserve missingness, conflicts and structural breaks;
- maintain retrospective-vintage caveats.

No modelling conclusion is implied by Phase 0.

### Phase 1 — target-independent structural and trajectory discovery
Status: **complete**. Validated run `35727510691`, evidence artifact `10694820381`.

Phase 1 showed that the information set is materially high-dimensional, annual PCA axes rotate through time, and the first frozen-PCA geometry is too strongly related to data coverage to be used directly as the transition state.

Purpose:
Discover the geometry and temporal organization of the full eligible information set without reading supervised targets, crisis labels, production risk scores or the production pillar definitions.

Phase 1 answers:
- How many independent dimensions are present?
- Which observed series reinforce the same underlying structure?
- Which series are distinct rather than duplicated?
- Which latent dimensions are stable through time?
- Which dimensions are transient or regime-specific?
- How do countries move through the discovered state space?
- Which country trajectories are similar without assuming similarity means risk?
- How sensitive is the geometry to source size, duplicate profiles, missingness and unit treatment?

No feature is selected because it predicts a chosen outcome in Phase 1.

### Phase 2 — stable measurement state, then predictable transitions
Status: **active**. Detailed requirements: [broad-feature-phase2-v0.6.md](broad-feature-phase2-v0.6.md).

Phase 2 is deliberately split:

#### Phase 2A — missing-aware measurement state
Estimate a stable underlying banking-system state directly from observed cells. Missing values remain missing rather than becoming median-filled pseudo-observations. Loadings are initially fixed across time so a state coordinate has a consistent meaning across years. State dimension is selected from target-independent held-out-observation reconstruction, not a fixed component count.

#### Phase 2B — predictable transition state
After 2A produces a defensible state, determine which combinations of that state actually move predictably. Compare no-change, simple pooled dynamics, reduced-rank transitions, dynamic/state-space transitions and analogue-transition distributions.

The descriptive state dimension and the predictable transition dimension are separate quantities and are both determined from data.

### Phase 3 — multivariate future-state forecasting

Purpose:
Generate one- and two-year distributions for the future banking-system state and reconstruct observable variables where justified.

Outputs:
- expected future state;
- uncertainty;
- movement vector;
- likely peer/analogue region;
- absolute versus relative movement;
- reconstruction diagnostics for observable variables.

Performance must be compared with no-change and simple dynamic baselines.

### Phase 4 — supervised overlays and interpretation

Purpose:
Only after the state architecture is established, test whether particular observable outcomes, crisis events or analyst-defined questions are explained/predicted by the discovered state.

No named outcome is privileged in advance. Outcome sets are registered when used and remain separate from the core state representation.

Possible overlays include systemic-crisis hazards, asset-quality deterioration, funding stress, capitalization changes or other empirically supported outcomes.

### Phase 5 — scenario and joint peer simulation

Purpose:
Simulate coherent future states under shared global/regional shocks and country-specific uncertainty.

Outputs:
- state distributions;
- relative peer movement;
- scenario decomposition;
- uncertainty cones;
- analogue paths.

### Phase 6 — confirmation and possible production integration

Purpose:
Lock model families, preprocessing and metrics before a genuinely forward confirmation period.

Production requires:
- demonstrable incremental value;
- stability across eras and coverage groups;
- calibrated uncertainty;
- clean governance evidence;
- explicit owner approval;
- rollback.

## Phase 1 requirements

### P1.1 — Full eligible feature library

Use all model-eligible representations from the canonical research library.

No:
- top-k filter;
- manually selected economic families;
- positive-correlation screen;
- production-pillar restriction;
- outcome-driven selection;
- minimum global coverage rule used as an importance filter.

Every source identity remains in the ledger even when temporarily unlearnable.

### P1.2 — Target firewall

Phase 1 code must not read:
- target-pairs files;
- crisis labels;
- crisis classifier outputs;
- production risk scores;
- production pillar scores.

Automated tests must fail if Phase 1 requires these inputs.

### P1.3 — Latest structural panel

The old 2023 boundary came from supervised development design and is not binding on target-independent discovery.

Build the Phase 1 structural panel through the latest exact annual information available in the frozen September 16 research vintage.

Rules:
- use exact annual / December endpoints;
- do not fabricate annual values from incomplete 2026 monthly data;
- preserve calendar gaps;
- retain level, lag and change representations where calculable;
- use the broad research country/entity membership only for country trajectories;
- keep group/global/context entities separate rather than treating them as extra countries.

### P1.4 — Unit-safe transformations

Comparable ratios/indexes may use robust/rank transformations.

Currency amounts require causal own-history normalization or another registered economically valid transform.

Unknown units remain quarantined but registered.

No raw-country identifiers or dates enter the geometry as explanatory features.

### P1.5 — Data-selected dimensionality

Compute the complete nonzero spectrum.

Report dimensions required for at least:
- 80%;
- 90%;
- 95%
of transformed variance.

These percentages are diagnostics, not a claim that any threshold is uniquely correct.

Do not force two dimensions.

### P1.6 — Reinforcement and redundancy

Retain all identities and separately measure:
- exact transformed-profile duplicates;
- source-level information concentration;
- feature contribution to common variance;
- residual / idiosyncratic contribution.

Run diagnostic sensitivities that rebalance duplicate-profile and source energy without deleting identities.

A repeated representation may reinforce signal, but replication alone must not be interpreted as independent corroboration.

### P1.7 — Temporal stability

Run the same target-independent discovery across historical annual origins.

For each year report:
- entities represented;
- learnable representations;
- numerical rank;
- dimensions for 80/90/95% variance;
- first-component share;
- source contribution distribution.

Compare adjacent and multi-year latent subspaces using principal-angle / subspace-overlap diagnostics on common learnable features.

No target data may influence stability conclusions.

### P1.8 — Frozen-space trajectories

Fit a declared reference geometry and apply it consistently to historical observations.

For each entity/year report:
- common-state coordinates;
- observed feature coverage;
- distance from reference center;
- year-on-year displacement magnitude;
- acceleration / change in displacement;
- direction persistence when consecutive displacement vectors exist.

These are descriptive trajectories, not risk scores.

### P1.9 — Feature stability

For each feature representation report across years:
- years learnable;
- contribution to common variance;
- median / dispersion of contribution;
- loading stability where comparable;
- source and identity metadata;
- redundancy-group membership.

Do not label a feature important merely because its contribution is large in one year.

### P1.10 — Neighbour / analogue geometry

For each entity-year, identify structurally nearest entity-years using the frozen target-independent space.

Similarity is not contagion, causality, safety or risk.

No future outcomes are used in Phase 1 analogue selection.

### P1.11 — Robustness

Phase 1 must test:
- row-order invariance;
- column-order invariance;
- exact-duplicate sensitivity;
- source-energy sensitivity;
- future-only feature isolation;
- unit rescaling for registered comparable transformations;
- missingness/coverage correlation with leading components;
- reproducibility under fixed numerical settings.

### P1.12 — Phase 1 acceptance

Phase 1 closes only when:

1. no supervised target or risk label is read;
2. the structural panel reaches the latest eligible annual origin in the frozen research vintage;
3. every registered identity has an admission state;
4. yearly spectra and stability diagnostics are exported;
5. entity trajectories are exported;
6. feature stability/redundancy ledgers are exported;
7. target-independent nearest-neighbour/analogue geometry is exported;
8. production paths and classifier bytes are unchanged;
9. results reproduce from immutable inputs;
10. limitations explicitly distinguish variance, distinctness, temporal stability and predictive value.

Phase 1 completion does not authorize Phase 2 promotion or any production change.

## Actions usage

Development should avoid repeated GitHub Actions execution.

Phase 1 implementation occurs on the child research branch without a PR-triggered workflow. Use local/static testing while developing. When the code and PRD are stable, use one consolidated Phase 1 execution/validation run against the existing immutable September 16 research artifact.

No fresh five-source retrieval is required for Phase 1 unless a separate data-refresh decision is made.


## Phase 1 evidence carried into Phase 2

Validated Phase 1 result:
- 46 annual structural origins through 2026;
- 15,470 predictor representations with no feature-count cap;
- 10,098 learnable representations in the 2026 reference;
- 68 / 97 / 120 components for 80% / 90% / 95% transformed variance;
- first component share about 13.2%;
- 11,549 feature-stability records;
- 8,819 entity trajectory rows;
- production unchanged and no supervised target/crisis/risk-score reads.

These findings are requirements for Phase 2 design, not predictive results. In particular, Phase 2 must reduce coverage-driven geometry before historical analogues or future-state forecasts are treated as economically meaningful.
