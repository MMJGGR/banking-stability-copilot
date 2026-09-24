#!/usr/bin/env bash
set -euo pipefail

: "${PHASE2_RESULTS:?Set PHASE2_RESULTS to the frozen Phase 2 results directory}"
: "${PHASE3_RESULTS:?Set PHASE3_RESULTS to the frozen Phase 3 results directory}"
: "${PHASE4_RESULTS:?Set PHASE4_RESULTS to the frozen Phase 4 results directory}"
: "${SHADOW_OUTPUT:?Set SHADOW_OUTPUT to a new output directory}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

arguments=(
  --phase2 "$PHASE2_RESULTS"
  --phase3 "$PHASE3_RESULTS"
  --phase4 "$PHASE4_RESULTS"
  --output "$SHADOW_OUTPUT"
  --issued-at "${SHADOW_ISSUED_AT:-2026-09-24}"
  --research-commit "${RESEARCH_MODEL_COMMIT:-31968c9596d928df9a12e29b5ea2ea442853213d}"
  --production-commit "${PRODUCTION_COMMIT:-5957ca779dafa21f2e098c819bfb060f43243206}"
)

if [[ -n "${PRODUCTION_REFERENCE:-}" ]]; then
  arguments+=(--production-reference "$PRODUCTION_REFERENCE")
fi
if [[ -n "${WEO_PROVIDER_PROJECTIONS:-}" ]]; then
  arguments+=(--provider-projections "$WEO_PROVIDER_PROJECTIONS")
fi

python -m src.forecasting.phase5_shadow "${arguments[@]}"
