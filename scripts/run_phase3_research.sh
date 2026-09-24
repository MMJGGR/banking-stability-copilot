#!/usr/bin/env bash
set -euo pipefail

# Large state-simulation matrices can trigger excessive BLAS thread fan-out.
# Two numerical threads completed the registered 800-draw run deterministically
# while avoiding local memory/timeout failures.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

python -m src.forecasting.phase3_probabilistic "$@"
