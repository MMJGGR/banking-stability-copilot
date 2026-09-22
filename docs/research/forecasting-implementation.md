# Broad-feature forecasting: execution status

Updated 2026-09-22. Active branch: `research/broad-feature-phase2-transition-2026-09-22`. Parent research PR #28 remains draft and production baseline `5957ca779dafa21f2e098c819bfb060f43243206` remains unchanged.

## Governing requirements

The active requirements are:
- original broad-feature PRD;
- source-grounded data contract v0.2;
- target-independent discovery plan v0.5;
- **Phase 2 PRD v0.6**, which incorporates the Phase 1 findings.

The owner requirement remains binding: no variable, source, feature family, target or outcome receives artificial emphasis or artificial exclusion merely because it appeared in an earlier model.

## Completed phases

### Phase 0 — canonical data foundation
Complete enough for current research use. Source identities, dimensions, units, status, conflicts and exact annual endpoints are retained with retrospective-vintage caveats.

### Phase 1 — target-independent structural discovery
**Complete.** Validated run `35727510691`; artifact `10694820381`; ZIP SHA256 `e561ab5caa7896ea17172b98b8d85f9bd4269533a4da4702ef0a72dd66c4eb43`.

Verified Phase 1 boundaries:
- 0 supervised targets read;
- 0 crisis labels read;
- 0 production risk scores read;
- no feature-count cap;
- no production modification or crisis-classifier retraining.

Key structural results:
- 46 annual origins through 2026;
- 15,470 predictor representations;
- 10,098 learnable representations in the 2026 reference;
- 68 / 97 / 120 data-derived components for 80% / 90% / 95% transformed variance;
- first component share about 13.2%;
- exact/profile and source-balancing sensitivities leave the state materially high-dimensional;
- frozen-PCA distances and movements remain too related to data coverage to be used directly as the forecasting state;
- annual PCA bases rotate materially through time even though broad feature-contribution patterns are much more stable.

The Phase 1 evidence therefore changes the next architecture: do not forecast annual PCA coordinates directly.

## Active Phase 2

### Phase 2A — missing-aware measurement state
**Implementation in progress.**

Goal: estimate a stable underlying country-year banking state directly from observed cells, without turning missing values into median-filled pseudo-observations.

First implementation family:
- regularized missing-aware low-rank measurement model;
- fixed feature loadings across time initially;
- target-independent rank selection using held-out observed-cell reconstruction;
- feature/source residual-reliability diagnostics;
- country-year state estimates and information/uncertainty diagnostics;
- explicit comparison of state geometry with data coverage.

The full eligible feature library remains registered. Feature exclusion is permitted only for data-contract or fold-local learnability reasons, not because a variable is deemed economically unimportant.

### Phase 2B — predictable transition state
Not started. It begins only after Phase 2A produces a defensible state.

Goal: determine which combinations of the richer measurement state actually move predictably through time. The forecastable transition rank is selected from forward state prediction, not static explained variance.

Initial comparators:
- no-change state;
- simple pooled autoregressive dynamics;
- reduced-rank transitions;
- dynamic/state-space transitions;
- analogue transition distributions.

## Deferred work

Earlier supervised level/residual experiments remain historical development evidence only. They do not define Phase 2 architecture and are deferred until later supervised overlays.

Historical analogue interpretation is also deferred until the Phase 2A state demonstrates that distance is not primarily data-coverage distance.

## Production firewall

Phase 2 may not alter:
- production crisis classifier;
- current serving risk model;
- current pillar pipeline;
- production source caches;
- Streamlit production code;
- deployed country scores.

No merge/promotion/deployment is authorized.

## GitHub Actions policy

Do not use Actions as the development loop. Phase 2 development should use local/synthetic validation and the existing immutable September 16 research artifact. Use one consolidated Actions/repository validation only when a Phase 2 subphase is ready to close.
