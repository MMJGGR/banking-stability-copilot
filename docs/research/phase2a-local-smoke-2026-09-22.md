# Phase 2A local engineering smoke test

Date: 2026-09-22
Branch: `research/broad-feature-phase2-transition-2026-09-22`
Scope: synthetic engineering validation only; **not real-data evidence and not predictive validation**.

The Phase 2A masked linear measurement prototype was exercised locally on a synthetic ragged panel with:
- 20 entities;
- 10 annual observations per entity;
- 36 feature representations;
- three underlying simulated factors;
- about 40% randomly missing cells.

The implementation:
- trained only on observed cells;
- held out a deterministic subset of observed cells for reconstruction;
- preserved at least one training observation for each fitted row/feature;
- estimated country-year states without filling missing cells;
- produced information/uncertainty diagnostics from the fitted loadings and observed cells.

Observed engineering behaviour:
- higher-rank candidates reconstructed held-out cells better than a one-factor model;
- when reconstruction was still improving at the largest rank tested, the rank search correctly returned `rank_search_boundary_reached = true` rather than treating the grid boundary as the final dimension;
- state-distance versus observed-share Spearman correlation in this synthetic smoke was approximately **0.025**;
- state-uncertainty versus observed-share Spearman correlation was approximately **-0.821**, which is the intended direction: sparser observations imply more uncertainty;
- year-on-year state movement versus absolute coverage change was approximately **-0.037** in the synthetic smoke.

These values are properties of the synthetic test design, not claims about the banking dataset.

The next evidence checkpoint is a single consolidated real-data Phase 2A run against the immutable September 16 research artifact after code review. Phase 2B remains blocked until Phase 2A passes its real-data state/coverage diagnostics.
