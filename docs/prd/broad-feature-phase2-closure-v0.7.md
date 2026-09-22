# PRD v0.7 — Phase 2 closure and Phase 3 handoff

Date: 2026-09-22  
Status: **Phase 2 complete as retrospective research/development evidence**  
Branch: `research/broad-feature-phase2-transition-2026-09-22`  
Production changes: none authorized

This addendum closes the Phase 2 requirements in PRD v0.6 and defines the evidence-based handoff to Phase 3. It does not authorize production integration.

## Phase 2 acceptance result

### Phase 2A — stable missing-aware state

Accepted.

- The model used observed cells only; missing cells were not converted into median or zero observations.
- The full eligible feature library remained registered without a feature-count cap.
- State rank was selected from held-out observed-cell reconstruction.
- The rank search was extended through 320 dimensions after the first run ended at its boundary.
- The smallest near-best measurement state was **96 dimensions**.
- The selected final measurement treatment used balanced reliability weighting.
- 8,783 country-year states were estimated from 4,539,769 observed cells across 14,652 eligible feature representations.
- Held-out reconstruction improved approximately 9.9% over the zero-state benchmark.
- State-distance versus coverage correlation fell from 0.948 in Phase 1 to 0.133.
- Movement versus coverage-change correlation fell to 0.093.
- Direct masking calibration produced 0.712 Spearman correlation between state error and estimated uncertainty.

All Phase 2A gates passed.

### Phase 2B — predictable state transitions

Accepted as retrospective development evidence.

The transition layer forecast the complete 96-dimensional state and did not use named banking outcomes or crisis labels.

- Development windows: 2016–2018 and 2019–2021.
- Horizons: one and two years.
- One-year best challenger: reduced-rank state-change model, improving RMSE by 17.3% versus no change.
- Two-year best challenger: separate regularized autoregression by state coordinate, improving RMSE by 16.9% versus no change.
- The best challengers beat no change on 86.6% and 90.2% of matched country-years, respectively.
- Average predicted-versus-realized movement-direction cosine was approximately 0.54 at both horizons.
- Historical-analogue transitions beat no change but underperformed the regularized transition models; analogues remain an explanatory layer rather than the primary engine.

## Architecture decision

The final architecture continues to distinguish:

1. **Measurement state:** a broad, stable representation of what is observed about the banking system.
2. **Predictable transition:** the part of that state whose future movement can be estimated.
3. **Probabilistic future state:** the distribution of possible one- and two-year positions, not just a point forecast.
4. **Analogue explanation:** historical cases used to explain and contextualize the formal forecast.
5. **Supervised overlays:** named banking outcomes and crisis risk added later without defining the underlying state.

The 96-dimensional measurement state is not a proposed user interface and is not interpreted as 96 named risk factors.

## Phase 3 requirements created by Phase 2

Phase 3 should build probability distributions around the state-transition forecasts.

Required elements:

- one- and two-year predictive state distributions;
- separately estimated common/global and country-specific transition shocks;
- calibrated interval or region coverage on later development data;
- mandatory no-change and simple autoregression baselines;
- joint peer simulation so relative position is not estimated from independent country forecasts;
- observable reconstruction diagnostics without imposing a privileged outcome list;
- uncertainty propagation from both current-state identification and future transition noise;
- analogue paths presented as explanation, not substituted for the formal probability model.

## Phase 3 gates

Phase 3 is not complete until:

1. state-distribution intervals are calibrated on time-ordered development data;
2. common and country-specific uncertainty are distinguished;
3. one- and two-year point forecasts continue to beat the registered simple baselines;
4. joint peer simulations preserve coherent shared shocks;
5. output uncertainty appropriately widens for weakly identified current states;
6. no supervised banking or crisis target is used to define the core state distribution;
7. no final confirmation period is consumed during development;
8. production remains unchanged.

## Evidence and limitations

Completed computation evidence:
- run `35745530960`;
- artifact `10702494849`;
- ZIP SHA256 `5f841b497df55e3498bc799231deffad773caf578939a41fc5c11dbee6c49f7e`.

The source run completed all measurement and transition computations, then stopped while serializing an undefined direction statistic for the no-change comparator. Undefined metrics are now serialized as JSON `null`; the completed output files were finalized without refitting models.

Remaining limitations:
- current-vintage historical data, not verified historical vintages;
- retrospective development windows, not a final untouched confirmation;
- no calibrated probabilistic future-state distribution yet;
- no production merge, deployment or model replacement authorized.
