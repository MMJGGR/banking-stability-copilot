# PRD v0.6 — Phase 2: stable measurement state and predictable transitions

Date: 2026-09-22
Status: active research requirements
Branch: `research/broad-feature-phase2-transition-2026-09-22`
Parent validated Phase 1 head: `5d6dcb55b1f9f3ecce9477491099a8512f9fe928`
Production changes: none authorized

## 1. Why Phase 2 is being redesigned

Phase 1 confirmed that the broad banking dataset is genuinely high-dimensional. In the 2026 reference cross-section, 10,098 representations were learnable and roughly 97 data-derived dimensions were required to explain 90% of transformed cross-sectional variance. Exact-profile and source-energy sensitivities reduced this only modestly.

Phase 1 also exposed two reasons not to forecast those PCA coordinates directly:

1. **Coverage bias:** countries observe very different shares of the broad feature space. Distances and apparent movement in the frozen PCA space are strongly related to how much data is observed.
2. **Rotating coordinates:** the exact PCA axes change materially through time even though the underlying feature-contribution pattern is much more stable.

Therefore Phase 2 is split into two natural subphases.

- **Phase 2A:** estimate a stable, missing-aware underlying banking-system state.
- **Phase 2B:** learn which parts of that state move predictably and how.

This preserves the original objective — broad vector state → trajectory → future state → historical analogues — but replaces a brittle median-filled PCA state with a time-consistent probabilistic state.

## 2. Owner principle remains binding

No variable, indicator, source, feature family, target or outcome receives artificial priority or artificial exclusion because of earlier modelling choices.

The full eligible feature library remains available to the measurement model.

Features may be unavailable to a particular fit only for data-contract reasons:
- unresolved unit or source identity;
- invalid numeric value;
- explicit break/status exclusion;
- no observations in the relevant training period;
- zero variation in the relevant training period.

These are data-validity constraints, not economic judgements.

No supervised banking target, crisis label, production risk score or production pillar score may be used to construct or tune the Phase 2A state.

## 3. Phase 2A — stable missing-aware measurement state

### 3.1 Objective

Estimate an underlying state for every country-year directly from the observations actually available for that country-year.

A country with sparse observations must receive a less certain state estimate than a country with rich observations. Missing values must not first be converted into thousands of apparently observed median values.

Conceptually:

[
x_{i,t,j} = mu_j + lambda_j^	op z_{i,t} + epsilon_{i,t,j}
]

where:
- (x_{i,t,j}) is one observed feature representation;
- (mu_j) is that representation's baseline;
- (lambda_j) tells us how it relates to the underlying state;
- (z_{i,t}) is the country's underlying banking-system state;
- (epsilon_{i,t,j}) is measurement noise.

Only genuinely observed cells contribute to the fitting loss.

### 3.2 First implementation family

Start with a regularized **missing-aware low-rank measurement model** using alternating least squares / EM-style updates on observed cells only.

Reasons:
- it works with the current very wide, ragged panel;
- it does not require complete-case rows;
- it lets us estimate stable loadings shared across years;
- it creates a direct benchmark before introducing a more complex Kalman/state-space implementation;
- it can later be embedded in a full probabilistic dynamic model.

This is a research baseline, not a commitment that ALS is the final production estimator.

### 3.3 Time consistency

The measurement loadings must initially be shared across time.

This means a state coordinate in 2005 and the same coordinate in 2025 has the same measurement definition.

Only if fixed loadings demonstrably fail should Phase 2A test slowly changing loadings.

We should not allow each year to invent a completely new coordinate system as annual PCA does.

### 3.4 Feature transformation and admission

Reuse the Phase 1 economic-unit rules:
- comparable ratios / indices may use robust or rank-based transforms;
- domestic-currency amounts require causal own-history normalization or another registered economically valid transform;
- unknown units remain registered but quarantined;
- context/global entities remain separate from country trajectories.

Do not use country code, calendar year, production score or target labels as explanatory features.

### 3.5 Learning feature reliability

The model should not assume every feature is equally reliable merely because it exists.

After an initial fit, estimate feature-level residual variance on observed cells and use it as a measurement-reliability diagnostic.

High-noise features should have less influence in a weighted sensitivity.

This is preferable to arbitrarily forcing equal weight by source.

Source-level residual summaries must still be exported to identify source-specific behaviour.

### 3.6 State dimension

Do not hard-code the number of state dimensions.

Choose it from target-independent evidence using observed-cell reconstruction:

1. split genuinely observed cells within training data into fit and validation cells;
2. fit candidate state dimensions without using the hidden validation cells;
3. compare reconstruction error on those held-out observed cells;
4. use the smallest dimension whose validation performance is statistically/economically indistinguishable from the best larger model;
5. if performance is still improving at the largest tested rank, expand the search rather than treating the boundary as final.

The Phase 1 variance spectrum is a guide to computational search, not the selection criterion.

### 3.7 Uncertainty / information quality

For each country-year export:
- number and share of observed eligible inputs;
- effective weighted information;
- reconstruction error on observed cells;
- state-estimation uncertainty or a transparent approximation to it;
- whether the estimate is sufficiently supported for trajectory/analogue use.

Uncertainty must increase when the observed information becomes sparse or noisy.

### 3.8 Coverage-bias diagnostic

Repeat the Phase 1 tests between:
- state distance and observed-share difference;
- state distance from center and own observed share;
- year-on-year state movement and observed-share change.

Phase 2A is not considered suitable for analogues if these relationships indicate that coverage remains the dominant driver of geometry.

Do not force the correlation to zero; real data availability may itself correlate with development. The requirement is to demonstrate that the state contains economic structure beyond coverage and report the sensitivity transparently.

### 3.9 Stability diagnostic

Compare the fixed-loadings state with Phase 1 annual PCA.

Report:
- reconstruction performance by year;
- state stability for countries with overlapping observations;
- loading stability by construction;
- residual patterns by year/source;
- whether a fixed measurement definition materially degrades in particular eras.

If structural breaks are severe, test explicit regime or slowly varying-loading challengers rather than silently refitting an unrelated state every year.

## 4. Phase 2B — discover the predictable transition state

### 4.1 Objective

Once a defensible measurement state (z_{i,t}) exists, determine which combinations of that state contain useful information about future movement.

Static descriptive dimension and forecastable dimension are separate quantities.

### 4.2 Baselines and challengers

Compare on later state observations:

1. **No change:** (z_{t+1}=z_t).
2. **Mean reversion / pooled AR:** each state coordinate follows a simple regularized autoregressive rule.
3. **Reduced-rank transition:** learn a lower-dimensional transition map from the full state.
4. **Dynamic-factor/state-space transition:** common transition dynamics plus country uncertainty.
5. **Historical analogue transition distribution:** similar past states/velocities imply an empirical distribution of next states.

Only after those baselines should nonlinear sequence models be considered.

### 4.3 Predictable rank

Do not assume the 90%-variance rank is the transition rank.

Select the transition rank using forward prediction of the complete future state, with time-ordered validation.

If only a small subset of state combinations are predictable, retain the richer measurement state while using the smaller transition state for forecasting.

### 4.4 Common shocks

Separate:
- global/common movement shared across many countries;
- country-specific movement.

Do not model every country as statistically independent when the data shows shared transitions.

Regional structure may be introduced only if it improves out-of-sample state prediction; region labels do not automatically define the latent state.

## 5. Phase 2 validation

Phase 2 remains target-independent.

Primary Phase 2A metrics:
- held-out observed-cell reconstruction MAE/RMSE;
- reconstruction improvement versus feature median / simple low-rank baselines;
- uncertainty calibration under artificial masking;
- sensitivity of state geometry to data coverage;
- stability through time.

Primary Phase 2B metrics:
- future-state error versus no-change;
- future-state error versus pooled simple AR;
- directional similarity of predicted vs realized state movement;
- uncertainty coverage once probabilistic transitions are available.

Validation must be time ordered for transition models.

No crisis labels or selected banking outcome values may be used to choose the state dimension or transition model.

## 6. Phase 2A first implementation slice

The first executable slice should deliver:

1. a target-independent Phase 2 panel reusing the immutable September 16 research inputs;
2. an observed-cell sparse matrix with no median-filled pseudo-observations;
3. a fixed-loading missing-aware low-rank model;
4. rank-selection diagnostics using held-out observed cells;
5. feature/source residual-reliability diagnostics;
6. country-year states;
7. information/uncertainty diagnostics;
8. coverage-bias comparisons against Phase 1;
9. reconstruction diagnostics by year and source;
10. no production changes.

Phase 2B coding starts only after the 2A state passes those basic quality checks.

## 7. Actions and execution policy

Do not use GitHub Actions as the development loop.

Phase 2 development and small synthetic/local tests should run outside Actions where possible.

Use one consolidated repository/real-data validation checkpoint only after a Phase 2 subphase is ready to close.

No fresh five-source retrieval is required merely to develop Phase 2; reuse immutable research inputs unless a separate source-refresh decision is made.

## 8. Production firewall

Phase 2 may not modify or retrain:
- the production crisis classifier;
- the serving risk model;
- the current pillar pipeline;
- the production source caches;
- Streamlit production code;
- deployed country scores.

Any future production proposal is a separate owner-approved milestone.


## 9. Implementation note

The first Phase 2A implementation uses PyTorch only as an efficient optimizer for a **linear** masked matrix-factorization model. The model itself remains a transparent low-rank measurement equation with feature loadings, country-year state coordinates and feature intercepts. This does not introduce a nonlinear neural-network architecture or change the model family described above.

The implementation is required to:
- use only genuinely observed cells in the fit objective;
- keep missing values absent from the loss;
- canonicalize the otherwise arbitrary factor rotation/sign before storing states;
- expose held-out reconstruction error, feature residual variance and state-information diagnostics;
- stop rather than declare a rank if the best reconstruction remains at the largest rank tested.


## 10. Phase 2A evidence-driven clarification: state uncertainty versus reconstruction error

The first full real-data Phase 2A execution showed that the proposed state-identification uncertainty score can decline strongly as information coverage rises while still being negatively correlated with held-out measurement reconstruction error.

These quantities answer different questions:

- **State-identification uncertainty:** how strongly the observed feature set pins down the latent banking-system state.
- **Held-out reconstruction error:** how well the shared measurement model reconstructs omitted observed indicators.

A dense country-year may be well pinned down in latent-state space while also containing more idiosyncratic/noisy indicators that are harder to reconstruct. Conversely, a sparse row may contain only a small set of easy-to-reconstruct indicators while its latent state is still weakly identified.

Therefore:
- both diagnostics remain mandatory and are reported separately;
- Phase 2A requires state-identification uncertainty to worsen as information becomes materially sparser/noisier;
- held-out reconstruction must beat a transparent zero-state baseline;
- Phase 2A no longer requires a positive cross-sectional correlation between state-identification uncertainty and held-out reconstruction error.

Future probabilistic forecasting must still calibrate forecast intervals separately; the Phase 2A uncertainty score is an information/identification diagnostic, not yet a fully calibrated posterior interval.
