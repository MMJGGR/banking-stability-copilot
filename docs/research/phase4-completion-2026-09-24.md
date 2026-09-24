# Phase 4 completion — interpretation and supervised overlays

Date: 24 September 2026  
Branch: `research/broad-feature-phase4-interpretation-overlays-2026-09-24`  
Status: **complete as retrospective research/development evidence**  
Production status: unchanged; no merge, promotion or deployment authorized

## Evidence provenance

Consolidated closure run:

- run: `35986163043`;
- validated head: `25107d3f974c7db905692ff071b98a3fb415a7c3`;
- artifact: `10801867846`;
- artifact name: `phase4-interpretation-overlays-35986163043`;
- artifact SHA256: `ab55f002f34fa992d3ade9dc27acdc32bdee379b4b586010dd1357d3b8844e6d`;
- Phase 4 safety tests: **4 passed**;
- every workflow step passed, including immutable-input checks, broad observable validation, official-label crisis overlay, production-path isolation and classifier verification.

The interpretation rotation was executed locally against the preserved Phase 2/3 evidence. Key local evidence hashes:

- `phase4a-summary.json`: `c25c4405b2a6f3eb0171616b9b77da3cb1cabb22838f8e7cec5a444b1a36b73a`;
- `interpretation-rotation.csv`: `a91d539e5d8e4847d27a53238ab5685a342f4db632350b8e626c43eee111f84f`;
- `dimension-interpretation-summary.csv`: `941560220987f1e1bfc2474d42fd708ed0e5294a03d7568dd44a96e3c85a4cd8`.

## Phase 4A — interpretation without changing the model

An orthogonal varimax rotation was applied to the fixed 96-dimensional Phase 2 state and its 14,652 measurement loadings.

Verified properties:

- state dimensions: **96**;
- measurement representations: **14,652**;
- country-year states: **8,783**;
- latest state year: **2026**;
- rotation converged after **73 iterations**;
- orthogonality maximum error: effectively **0**;
- maximum sampled pairwise-distance difference: approximately `2.07e-7`;
- measurement reconstruction difference: **0** to reported numerical precision.

The interpretation coordinates therefore preserve the state geometry and measurement model. They are a different coordinate system, not a refit.

### Interpretability limitation

The rotation did not turn the state into a small set of clean independent themes:

- all 96 interpreted dimensions remained dominated by FSIBSIS measurement energy;
- the median dimension still drew meaningful energy from roughly **775** representations;
- the average dominant-source share was approximately **56%**;
- machine-generated theme labels were often too literal or noisy to publish without analyst review.

Accordingly, the 96 internal dimensions should not be exposed as 96 named user-facing risk factors. The useful interpretation layer is country-specific attribution: which state directions and observable series explain a country’s current distinctness and expected movement.

## Phase 4B — broad observable-outcome validation

The fixed state-transition models were evaluated against the complete eligible annual level universe rather than a selected outcome list.

Evidence population:

- registered level identities: **4,924**;
- observed/evaluated level identities: **1,847**;
- observed historical level cells: **692,406**;
- entities: **214**;
- outcome/horizon evaluations admitted in both registered windows: **1,061**;
- state-predictable outcome/horizon evaluations after both-window consistency and false-discovery control: **14**;
- unique stable identities: **13**;
- stable one-year evaluations: **1**;
- stable two-year evaluations: **13**;
- provider-projection rows used: **0**;
- latest realized outcome year used: **2022**.

The registered windows were 2016–2018 and 2019–2021. Minimum evidence thresholds were 30 rows, 10 countries and two forecast origins per window. False-discovery control used an FDR threshold of 10%.

### Stable observable results

The state forecast showed stable incremental value for a small group of outcomes, chiefly:

- real GDP growth at one and two years;
- banking profitability measures, including return on equity, return on assets and net income, mostly at two years;
- general-government primary balance and output gap at two years;
- exports/imports volume growth at two years;
- one MFS other-depository-corporation claims series at two years.

Examples of pooled RMSE improvement versus persistence:

- real GDP growth, one year: approximately **10.8%**;
- FSIC return on equity/net income after tax, two years: approximately **16.6%**;
- FSIC return on assets/net income before tax, two years: approximately **16.4%**;
- general-government primary balance/GDP, two years: approximately **11.1%**;
- trade-volume measures, two years: approximately **6.7%–7.8%**.

### Broad negative result

The state transition is **not** a universal raw-indicator forecasting engine.

For most individual FSIC, FSIBSIS, MFS, WEO and WGI identities, reconstructing the future observable directly from the common state did not beat persistence consistently. Median improvement was negative in every source/horizon group.

This means the final architecture should:

1. keep the common state for broad system condition, trajectory and joint probability forecasts;
2. present observable implications primarily as directional attribution;
3. use separately validated supervised overlays when an exact raw observable forecast is needed;
4. preserve negative outcome results rather than implying that every source series is predictable from the state.

## Phase 4C — systemic-crisis overlay

The official Laeven–Valencia 1970–2025 episode file was used with a one-to-three-year crisis-onset horizon.

Panel construction:

- eligible country-year rows: **6,980**;
- countries: **214**;
- positive crisis-onset rows: **378**;
- positive rate: approximately **5.42%**;
- systemic episodes used: **161**;
- borderline episodes used: **0**;
- active-crisis exclusions: **492**;
- post-crisis cooldown exclusions: **473**;
- right-censored exclusions: **854**;
- provider-projection rows used: **0**.

Four later development windows executed, generating 5,144 matched test rows and 167 positives.

### Crisis-model result

| Model | Brier score | Log loss | PR AUC | ROC AUC |
|---|---:|---:|---:|---:|
| Historical event-rate baseline | **0.0342** | **0.1652** | 0.0531 | **0.6897** |
| Fixed state only | 0.0446 | 0.2068 | 0.0342 | 0.4619 |
| State + velocity + uncertainty | 0.0512 | 0.2414 | **0.0644** | 0.6366 |

The state-plus-velocity model showed a small improvement in precision-recall ranking but materially worse probability calibration and very poor false-alert burden. Neither challenger improved Brier score or log loss.

### Final fail-closed decision

**Do not advance a state-based crisis overlay.**

The initial development runner wrote an exploratory latest-probability file for the least-bad challenger. That file is rejected and is not an admissible research or production output. The final advancement gate requires an overlay to improve aggregate Brier score and log loss without reducing PR AUC relative to the event-rate baseline. No challenger passes.

Consequences:

- no Phase 4 latest crisis probabilities are approved;
- the locked production crisis classifier remains unchanged;
- the fixed state is not reoriented or refitted using crisis labels;
- a future crisis challenger would need crisis-specific raw information alongside the state and a separately registered validation design.

The production classifier remains 52,580 bytes with SHA256:

`054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee`

A matched historical comparison with the locked production classifier was not possible because matched historical predictions from that locked artifact were not preserved. The production classifier was not retrained to manufacture that comparison.

## Architectural conclusion

Phase 4 supports a layered architecture rather than one universal model:

1. **Broad measurement state:** preserves the information in the wide, missing dataset.
2. **State transition and probability engine:** forecasts the complete banking-system state and peer distribution.
3. **Interpretation layer:** explains country states and movement through observable loadings and analyst-reviewed themes.
4. **Observable overlays:** separate models for exact raw outcomes where broad-screen evidence supports them.
5. **Event overlays:** separately governed rare-event models; the current state-only crisis challenger fails and does not advance.
6. **Provider scenarios:** WEO 2026–2031 projections remain separate conditional context, not observed data or model targets.

The main value of the new architecture remains system-level state, trajectory, uncertainty and relative-peer forecasting. It should not be sold as an accurate forecast for every underlying raw indicator or as a replacement crisis classifier.

## Production firewall

Verified unchanged:

- production app;
- deployed country scores;
- production pillar pipeline;
- serving source caches;
- risk model;
- locked crisis classifier.

No merge, promotion or deployment is authorized.

## Next research milestone

The next natural phase is **shadow integration and prospective confirmation**, not another wholesale redesign:

- freeze the completed research stack;
- create a research serving contract alongside production;
- expose state, trajectory, probabilities and explanations in shadow mode;
- compare future incoming vintages with frozen forecasts;
- retain the current production score and crisis classifier as independent benchmarks;
- evaluate WEO-conditioned scenarios as a visibly separate lane;
- require owner approval before any production substitution.
