# WEO provider-projection audit — 24 September 2026

Branch: `research/broad-feature-phase3-probabilistic-2026-09-23`  
Research source cutoff: 16 September 2026  
Source: `IMF.RES:WEO(9.0.0)`  
Production changes: none

## Finding

The immutable WEO response used by the research store contains annual values extending through 2031. Rows dated after the 16 September 2026 source cutoff are provider projections, not realized observations.

| Projection year | Rows |
|---:|---:|
| 2026 | 7,907 |
| 2027 | 7,868 |
| 2028 | 7,657 |
| 2029 | 7,652 |
| 2030 | 7,652 |
| 2031 | 7,652 |
| **Total** | **46,388** |

Additional inventory:

- entities represented: **204**;
- indicators represented: **145**;
- invalid projection rows: **0**;
- duplicate projection identities: **0**;
- raw WEO rows inspected: **361,736**;
- metadata-only rows: **3**.

## Was Phase 1 or Phase 2 contaminated?

No. The canonical research pipeline already marked these rows `after_cutoff`, so they did not enter:

- the historical measurement state;
- Phase 1 structural discovery;
- Phase 2 state fitting;
- Phase 2 transition targets;
- uncertainty calibration;
- realized outcomes.

Phase 3 baseline fitting and simulation also record `provider_projection_rows_read = 0`.

The change made in this audit is semantic and operational: the rows are no longer regarded merely as rejected future observations. They are retained in a dedicated provider-projection ledger for optional scenario use.

## Binding treatment

The research architecture now keeps three separate data lanes.

### 1. Historical / observed lane

Used for state estimation, transitions, calibration and realized outcomes. Provider projections are prohibited.

### 2. Provider-projection lane

The WEO 2026–2031 values are labelled:

- `data_role = provider_projection`;
- `model_admission = scenario_only_not_measurement_or_target`.

Allowed uses:

- scenario context;
- provider benchmark;
- analyst display;
- explicitly conditional forecast input.

Forbidden uses:

- measurement-state fitting;
- transition targets;
- transition calibration;
- realized outcomes;
- model selection;
- historical backtests using a later WEO vintage.

### 3. Banking Copilot forecast lane

The Copilot’s baseline one- and two-year future-state distributions remain independent of current-vintage WEO projections. There is no automatic blending or averaging.

A later WEO-conditioned scenario may be produced, but it must be labelled conditional and remain separate from the model-only baseline.

## Important 2025 caveat

The current WEO feed does not expose a complete actual-versus-estimate boundary for every indicator. Therefore WEO observations at or before the cutoff, particularly 2025, remain `historical_or_estimate_unverified` unless source-vintage evidence establishes their status.

This does not affect the registered Phase 2 development windows, which ended earlier. It does mean that the latest 2026-origin state may incorporate current-vintage 2025 WEO values whose actual/estimate status is not fully verified.

## Historical validation rule

Current 2026-vintage WEO projections cannot be inserted into earlier historical backtests as though they had been known at those dates. A valid provider-projection backtest requires the original WEO vintage available at each historical forecast origin.

## Evidence

The local audit exports:

- full 2026–2031 projection ledger;
- historical/status-unverified sample;
- machine-readable summary;
- projection-use policy.

Summary SHA256: `cb96968da47e51f675fa1f7c656db2cf2efd9eaef2ce01792a65d85791913325`.
