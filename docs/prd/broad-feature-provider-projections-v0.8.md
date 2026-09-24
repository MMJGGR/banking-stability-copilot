# PRD amendment v0.8 — Provider projections are a separate data lane

Date: 2026-09-24  
Status: binding amendment to Phase 3  
Branch: `research/broad-feature-phase3-probabilistic-2026-09-23`  
Production changes: none authorized

## 1. Finding

The immutable 16 September 2026 IMF WEO response contains annual values from 1980 through 2031. Relative to the research cutoff of 16 September 2026, the rows for 2026–2031 are future provider projections rather than realized observations.

Verified current-vintage WEO projection inventory:

- provider-projection rows: **46,388**;
- years: **2026–2031**;
- entities: **204**;
- indicators: **145**;
- source version: `IMF.RES:WEO(9.0.0)`;
- invalid projection rows: **0**.

Rows by year:

| Projection year | Rows |
|---:|---:|
| 2026 | 7,907 |
| 2027 | 7,868 |
| 2028 | 7,657 |
| 2029 | 7,652 |
| 2030 | 7,652 |
| 2031 | 7,652 |

The existing canonical research build already excluded these rows from Phase 1 and Phase 2 by recording them as `after_cutoff`. They did not enter the measurement state, transition targets, calibration residuals, or realized outcomes. This amendment improves the semantics: future WEO rows are not merely “rejected”; they are retained in a separate provider-projection ledger.

## 2. Three distinct lanes

### Lane A — observed or historical data

Purpose:
- construct the historical measurement state;
- construct exact-calendar transitions;
- fit and calibrate the research models;
- evaluate realized outcomes.

Rules:
- provider projections may never enter this lane;
- missing values remain missing;
- historical publication/vintage limitations remain explicit;
- current-vintage WEO values at or before the cutoff remain `historical_or_estimate_unverified` unless a separate source-vintage ledger proves actual/estimate status.

### Lane B — provider projections

Purpose:
- optional external scenario context;
- comparison benchmark;
- analyst display;
- explicitly conditional forecasts.

Rules:
- data role is `provider_projection`;
- model admission is `scenario_only_not_measurement_or_target`;
- projections are preserved by provider, source version, retrieval date, indicator, entity, projection year, unit, scale, and all available source dimensions;
- projections are never treated as realized values merely because their year later becomes historical;
- revisions between WEO vintages are tracked as provider forecast revisions, not historical-data revisions.

Allowed uses:
- `scenario_context`;
- `provider_benchmark`;
- `analyst_display`;
- `conditional_forecast_input`.

Forbidden uses:
- measurement-state fitting;
- transition targets;
- transition calibration;
- realized outcomes;
- model selection;
- historical backtests using a later WEO vintage.

### Lane C — Banking Copilot model forecasts

Purpose:
- unconditional one- and two-year future-state distributions;
- later conditional scenarios;
- peer and analogue simulations.

Rules:
- model forecasts are generated independently of current-vintage WEO projections by default;
- model forecasts and WEO projections must be displayed separately;
- no automatic averaging or blending;
- any WEO-conditioned forecast is labelled conditional and remains separate from the model-only baseline.

## 3. Phase 3 treatment

The Phase 3 baseline forecast remains independent of WEO 2026–2031 projections.

The Phase 3 completion report must record:

- `provider_projection_rows_read = 0` for baseline fitting, calibration, and simulation;
- the existence of the separate provider-projection ledger;
- whether any optional provider-conditioned scenario was executed.

A later scenario module may map WEO projection changes into the learned state through the Phase 2 measurement loadings, but only as an explicitly conditional path. It must not overwrite the baseline future-state forecast.

## 4. Historical validation rule

Current 2026-vintage WEO projections cannot be inserted into historical backtests as though they were available in earlier years.

A vintage-clean provider-projection comparison requires the original WEO vintage that existed at each historical forecast origin. If those vintages are unavailable, historical provider-projection performance is `not verified`.

## 5. 2025 WEO caveat

The current WEO feed does not expose a complete actual-versus-estimate boundary for every indicator. Therefore 2025 WEO observations remain `historical_or_estimate_unverified` in the retrospective state architecture.

This does not contaminate the registered Phase 2 development transitions, which end before the latest years, but it is a limitation of the latest 2026-origin state and must remain visible.

## 6. Acceptance criteria

1. All WEO rows after the research cutoff are stored outside historical/model-training tables.
2. Projection records retain complete source identity and vintage metadata.
3. Code fails closed if provider projections are requested as training data, transition targets, calibration data, or realized outcomes.
4. Baseline Phase 3 reports zero provider-projection rows read.
5. Optional provider-conditioned scenarios are separate outputs, never silent replacements.
6. Production remains unchanged.
