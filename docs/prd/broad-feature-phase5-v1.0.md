# PRD v1.0 — Phase 5: shadow integration and prospective confirmation

Date: 24 September 2026  
Status: active research requirements  
Branch: `research/broad-feature-phase5-shadow-confirmation-2026-09-24`  
Parent Phase 4 head: `31968c9596d928df9a12e29b5ea2ea442853213d`  
Production changes: none authorized

## 1. Purpose

Phases 1–4 established and tested a broad banking-system research architecture:

**broad observed data → missing-aware state → learned transition → calibrated future-state distribution → peer simulation → country-specific interpretation**

Phase 5 does not redesign that architecture. It freezes the accepted research stack, exposes it through a separate shadow-serving contract and opens an append-only prospective confirmation ledger.

The shadow output must allow analysts to inspect the new architecture beside production without changing production scores, classifications, crisis probabilities, source caches or application behavior.

Phase 5 has three subphases:

- **5A — freeze and serve:** build a deterministic, versioned research bundle;
- **5B — shadow viewer:** provide a separate read-only interface for analyst review;
- **5C — prospective confirmation:** lock forecasts before later source vintages and realized outcomes become available, then evaluate them without rewriting history.

Phase 5 can complete its engineering/freeze milestone now. Prospective performance confirmation remains open until future observations arrive.

## 2. Frozen evidence inputs

The initial shadow batch is built only from the accepted Phase 2–4 evidence.

### Phase 2

- source run: `35745530960`;
- artifact: `10702494849`;
- ZIP SHA256: `5f841b497df55e3498bc799231deffad773caf578939a41fc5c11dbee6c49f7e`;
- state dimensions: 96;
- country-year states: 8,783;
- source cutoff: 16 September 2026.

### Phase 3

- status: completed probabilistic development milestone;
- latest country states: 213;
- latest forecast rows: 426, covering one- and two-year horizons;
- point-model families and penalties fixed from Phase 2;
- calibrated empirical probability regions accepted;
- baseline provider-projection rows read: zero.

### Phase 4

- consolidated run: `35986163043`;
- artifact: `10801867846`;
- artifact SHA256: `ab55f002f34fa992d3ade9dc27acdc32bdee379b4b586010dd1357d3b8844e6d`;
- interpretation rotation accepted for attribution only;
- broad observable screen accepted with 13 unique stable identities;
- state-based crisis overlay rejected by the fail-closed advancement gate.

### Production benchmark

- production baseline commit: `5957ca779dafa21f2e098c819bfb060f43243206`;
- classifier: 52,580 bytes;
- classifier SHA256: `054811a0b12133592bd22de64e2141c6c969d473963170199cff441e8e689aee`.

The production benchmark is read-only and remains methodologically independent from the research forecast.

## 3. Separate output lanes

The shadow contract must preserve four visibly separate lanes.

### 3.1 Research model baseline

Contains:

- current 96-dimensional research state;
- state-information quality and uncertainty;
- one- and two-year model-only forecast distributions;
- future peer-position distributions;
- nearest-peer frequencies;
- historical analogues;
- country-specific state and movement attribution;
- standardized observable implications.

It must not use WEO 2026–2031 projections.

### 3.2 Production benchmark

Contains only current production outputs available for the same country:

- production risk score;
- production risk category;
- production crisis probability;
- production snapshot/version identifiers.

Production outputs must not be blended into the research state or forecast. Differences are displayed, not reconciled into a combined score.

### 3.3 Provider-projection scenarios

Current-vintage WEO 2026–2031 projections remain a separate provider lane.

The baseline shadow contract may indicate whether provider context exists for a country and forecast year, but it must not inject provider projections into the model-only forecast.

A future WEO-conditioned scenario must have a distinct scenario identifier and be labelled conditional.

### 3.4 Rejected or unavailable overlays

The Phase 4 state-based crisis overlay is rejected.

The serving contract must report:

- `research_crisis_overlay_status = not_approved`;
- no Phase 4 latest crisis probability;
- production crisis probability only inside the production benchmark lane.

Rejected exploratory outputs must never appear in the accepted bundle.

## 4. Time and vintage semantics

The initial shadow batch uses:

- source retrieval/cutoff: 16 September 2026;
- research issue date: 24 September 2026;
- state origin year: 2026;
- forecast years: 2027 and 2028.

The state origin year is a model timing label, not proof that all inputs are final 2026 observations. Under the registered lag convention, the latest state is primarily informed by observations through 2025 and earlier.

WEO observations dated 2025 remain `historical_or_estimate_unverified` unless their actual/estimate status is independently established.

The serving contract must expose:

- `source_cutoff`;
- `issued_at`;
- `state_origin_year`;
- `forecast_year`;
- `horizon_years`;
- `vintage_mode = retrospective_latest_vintage_research`;
- the WEO 2025 status caveat.

## 5. Shadow-serving contract

The accepted bundle consists of a manifest and normalized tables.

### 5.1 Manifest

Required fields include:

- schema version;
- deterministic batch identifier;
- branch/commit and evidence hashes;
- source cutoff and issue date;
- model/status boundaries;
- country and forecast-row counts;
- production overlap counts;
- provider-projection policy;
- crisis-overlay rejection policy;
- accepted file names and SHA256 checksums.

### 5.2 Country-horizon summary

One row per research country and horizon. Required content:

- entity code and name when resolved;
- state origin year and forecast year;
- forecast quality;
- observed feature count/share;
- state uncertainty;
- calibrated 50%/80%/95% movement radii;
- current and future peer-distance percentiles;
- relative improvement/deterioration probabilities;
- probability of moving closer to/farther from the contemporary peer center;
- production benchmark availability and fields;
- provider-scenario availability;
- rejected crisis-overlay status.

### 5.3 Full state and forecast coordinates

The 96-dimensional current state and coordinate quantiles remain available in separate machine-readable tables. They are not presented as 96 user-facing risk factors.

### 5.4 Explanations

Each country receives:

- leading current-state contributions;
- leading forecast-movement contributions for each horizon;
- top observable implications with uncertainty;
- whether an observable identity passed Phase 4 stable-outcome validation;
- likely future peers;
- historical analogues.

Machine-generated dimension labels remain marked `requires_analyst_review`.

### 5.5 Research-only country coverage

Research and production populations differ.

The initial bundle must explicitly report:

- research countries with a production benchmark;
- research-only countries;
- production countries absent from the research forecast;
- no inference that absence means low or high risk.

## 6. Prospective forecast ledger

The initial batch creates an append-only forecast ledger with one locked record per country/horizon.

Required fields include:

- deterministic `forecast_batch_id`;
- entity code;
- horizon and forecast year;
- point/quantile file hash;
- issue date and source cutoff;
- model/version identifiers;
- forecast quality and state-information fields;
- `realization_status = pending`;
- blank realized-state and scoring fields;
- evaluation rule and earliest eligible evaluation date.

The original forecast record must never be overwritten when later data arrive.

Later vintages create separate realization records linked to the original forecast key.

## 7. Prospective evaluation rules

When eligible future observations become available, Phase 5C will evaluate:

- state RMSE/MAE versus no change;
- realized movement direction;
- 50%/80%/95% region coverage;
- probability-region sharpness;
- future peer-percentile calibration;
- stable observable overlays where their realized source status is verified;
- production benchmark changes as context, not as the research target.

Provider projections are never treated as realized outcomes.

A realization may be scored only when:

1. the source observation is dated at or after the target year;
2. the observation is not a provider forecast;
3. source status and identity pass the data contract;
4. the forecast record predates the realization vintage;
5. the evaluation code leaves the original forecast unchanged.

## 8. Shadow viewer

A separate read-only research viewer may be added, but it must:

- use a separate entry point from production;
- read only the frozen shadow bundle;
- display `Research / not production` prominently;
- keep production and research outputs in separate panels;
- suppress the rejected state-based crisis overlay;
- show production crisis probability only as a production benchmark;
- expose uncertainty and data-support measures;
- show WEO projections only in a distinct provider-scenario section;
- make no network writes and no production artifact writes.

No shadow viewer is deployed by this phase unless separately authorized.

## 9. Acceptance criteria for the engineering/freeze milestone

Phase 5A/5B engineering closes when:

1. the bundle is deterministically reproduced from the frozen Phase 2–4 inputs;
2. all 213 latest research countries and 426 country-horizon rows are represented;
3. current state, forecast distributions, explanations, peers and analogues reconcile to source files;
4. provider-projection rows read by the baseline remain zero;
5. rejected crisis probabilities are absent;
6. production benchmark values remain separate and read-only;
7. research/production population differences are explicit;
8. an append-only prospective ledger is generated;
9. file hashes and a machine-readable manifest are exported;
10. production paths and classifier bytes remain unchanged.

Prospective confirmation itself cannot close until future realized data become available.

## 10. Promotion rules

Phase 5 does not authorize production substitution.

A later integration decision requires:

- at least one genuinely prospective observation cycle;
- acceptable prospective calibration and point accuracy;
- stable analyst interpretation;
- operational monitoring and rollback;
- explicit owner approval.

Until then, the current production model and classifier remain the served system of record.

## 11. Execution policy

Use local execution against the existing Phase 2–4 evidence during development.

Do not use GitHub Actions as the iterative development loop. At most one consolidated repository checkpoint may be used after the shadow bundle and ledger are closure-ready, and only if local evidence is insufficient.
