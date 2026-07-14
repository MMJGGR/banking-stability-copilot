"""Smoke-test serving artifacts before promotion.

Run after a candidate snapshot is built (and again during promotion) to catch
artifact-content defects that checksums and coverage stats cannot see — for
example the 2026-07-10 incident where a candidate shipped with every
country_name blank and all existing gates passed it.

Usage:
    python -m src.scripts.smoke_test_artifacts

Exits non-zero when any check fails.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.config import BASE_DIR, CACHE_DIR
from src.country_names import fill_missing_country_names
from src.model_store import load_data_manifest, load_model_artifact
from src.snapshot_manifest import sha256_file

MINIMUM_COUNTRIES = 150
MINIMUM_RAW_NAME_SHARE = 0.90


def _check(condition: bool, message: str, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)


def manifest_artifact_failures(manifest: dict, repository_root=BASE_DIR) -> list[str]:
    """Return missing, unsafe, or checksum-invalid manifest entries."""
    root = Path(repository_root).resolve()
    failures = []
    for relative_path, metadata in manifest.get("artifacts", {}).items():
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"unsafe path: {relative_path}")
            continue
        expected = metadata.get("sha256")
        if not candidate.is_file():
            failures.append(f"missing: {relative_path}")
        elif not expected or sha256_file(candidate) != expected:
            failures.append(f"checksum: {relative_path}")
    return failures


def run_smoke_tests() -> int:
    failures: list = []

    print("Serving-artifact smoke tests")
    print("=" * 60)

    manifest = load_data_manifest()
    _check(bool(manifest), "data manifest present", failures)
    if manifest:
        _check(
            manifest.get("snapshot_status") == "verified",
            f"manifest snapshot_status is 'verified' "
            f"(got {manifest.get('snapshot_status')!r})",
            failures,
        )
        checksum_failures = manifest_artifact_failures(manifest)
        _check(
            not checksum_failures,
            "all manifest artifacts are present, safely rooted, and checksum-valid"
            + (f" ({checksum_failures[:3]})" if checksum_failures else ""),
            failures,
        )

    try:
        model = load_model_artifact()
    except Exception as exc:
        _check(False, f"risk model loads and passes checksum ({exc})", failures)
        print(f"\n{len(failures)} smoke-test failure(s)")
        return 1
    _check(True, "risk model loads and passes checksum", failures)

    scores = model["country_scores"]
    _check(
        len(scores) >= MINIMUM_COUNTRIES,
        f"at least {MINIMUM_COUNTRIES} countries scored (got {len(scores)})",
        failures,
    )
    _check(
        scores["risk_score"].between(1, 10).all(),
        "all risk scores within [1, 10]",
        failures,
    )

    names = scores["country_name"].fillna("").astype(str).str.strip()
    raw_share = float((names != "").mean()) if len(names) else 0.0
    _check(
        raw_share >= MINIMUM_RAW_NAME_SHARE,
        f"artifact carries display names for >= {MINIMUM_RAW_NAME_SHARE:.0%} "
        f"of countries (got {raw_share:.0%})",
        failures,
    )
    display = fill_missing_country_names(
        scores[["country_code", "country_name"]].copy(),
        fallback_to_code=True,
    )
    display_names = display["country_name"].fillna("").astype(str).str.strip()
    _check(
        bool((display_names != "").all()),
        "no country renders with a blank display name",
        failures,
    )

    # Raw/imputed sidecar coherence: imputation must only fill gaps, and the
    # sidecar must come from the same build as the feature frame.
    try:
        crisis = pd.read_parquet(f"{CACHE_DIR}/crisis_features.parquet")
        imputed = pd.read_parquet(f"{CACHE_DIR}/imputed_features.parquet")
        crisis = crisis.set_index("country_code")
        imputed = imputed.set_index("country_code")
        shared_countries = crisis.index.intersection(imputed.index)
        shared_columns = [
            column
            for column in imputed.columns
            if column in crisis.columns
            and not column.endswith("_year")
            and pd.api.types.is_numeric_dtype(imputed[column])
        ]
        _check(len(shared_columns) > 0, "raw and imputed frames share feature columns", failures)
        mismatches = []
        for column in shared_columns:
            raw_values = pd.to_numeric(crisis.loc[shared_countries, column], errors="coerce")
            imputed_values = pd.to_numeric(imputed.loc[shared_countries, column], errors="coerce")
            observed = raw_values.notna() & imputed_values.notna()
            if observed.any() and not np.allclose(
                raw_values[observed], imputed_values[observed], rtol=1e-6, atol=1e-8
            ):
                mismatches.append(column)
        _check(
            not mismatches,
            "imputed sidecar preserves every observed raw value "
            + (f"(mismatched: {mismatches[:5]})" if mismatches else ""),
            failures,
        )
    except FileNotFoundError as exc:
        _check(False, f"feature frames present ({exc})", failures)

    print()
    if failures:
        print(f"{len(failures)} smoke-test failure(s):")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
