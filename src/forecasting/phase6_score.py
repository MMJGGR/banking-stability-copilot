"""Leakage-safe Phase 6 replacement score construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_STEP = 0.10
MIN_WEIGHT = 0.10


def peer_percentile(signal: pd.Series, origins: pd.Series) -> pd.Series:
    frame = pd.DataFrame({
        "signal": pd.to_numeric(signal, errors="raise").to_numpy(float),
        "origin": pd.to_numeric(origins, errors="raise").to_numpy(int),
    })
    return frame.groupby("origin", observed=True).signal.rank(
        pct=True, method="average"
    )


def historical_percentile(
    signal: pd.Series,
    countries: pd.Series,
    origins: pd.Series,
    *,
    prior_reference: pd.DataFrame | None = None,
    shrinkage_observations: float = 5.0,
) -> pd.Series:
    """Expanding own-history percentile; no later signal can enter an earlier row."""
    frame = pd.DataFrame({
        "signal": pd.to_numeric(signal, errors="raise").to_numpy(float),
        "country": countries.astype(str).to_numpy(),
        "origin": pd.to_numeric(origins, errors="raise").to_numpy(int),
        "position": np.arange(len(signal)),
    }).sort_values(["origin", "country", "position"])
    histories: dict[str, list[float]] = {}
    if prior_reference is not None and len(prior_reference):
        required = {"country", "origin", "signal"}
        if not required <= set(prior_reference):
            raise ValueError(f"prior_reference missing {sorted(required-set(prior_reference))}")
        for country, group in prior_reference.sort_values("origin").groupby(
            "country", observed=True
        ):
            histories[str(country)] = group.signal.astype(float).tolist()
    output = np.full(len(frame), 0.5, dtype=float)
    for origin, origin_rows in frame.groupby("origin", observed=True, sort=True):
        staged: list[tuple[str, float]] = []
        for row in origin_rows.itertuples(index=False):
            history = histories.get(str(row.country), [])
            if history:
                percentile = float(
                    np.searchsorted(np.sort(np.asarray(history)), row.signal, side="right")
                    / len(history)
                )
                weight = len(history) / (len(history) + shrinkage_observations)
                value = weight * percentile + (1 - weight) * 0.5
            else:
                value = 0.5
            output[int(row.position)] = value
            staged.append((str(row.country), float(row.signal)))
        # Same-origin peers cannot enter each other's own-history reference.
        for country, value in staged:
            histories.setdefault(country, []).append(value)
    return pd.Series(output, index=signal.index)


def weight_grid(step: float = WEIGHT_STEP, minimum: float = MIN_WEIGHT):
    units = int(round(1 / step))
    minimum_units = int(round(minimum / step))
    for peer in range(minimum_units, units + 1):
        for history in range(minimum_units, units - peer + 1):
            absolute = units - peer - history
            if absolute >= minimum_units:
                yield peer / units, history / units, absolute / units


def select_weights(frame: pd.DataFrame) -> dict:
    required = {"peer", "history", "absolute", "y"}
    if not required <= set(frame):
        raise ValueError(f"score tuning frame missing {sorted(required-set(frame))}")
    best = None
    for peer_weight, history_weight, absolute_weight in weight_grid():
        index = (
            peer_weight * frame.peer
            + history_weight * frame.history
            + absolute_weight * frame.absolute
        ).clip(1e-8, 1 - 1e-8)
        y = frame.y.to_numpy(float)
        probability = index.to_numpy(float)
        brier = float(np.mean((y - probability) ** 2))
        logloss = float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability)))
        candidate = (brier, logloss, -absolute_weight, peer_weight, history_weight)
        if best is None or candidate < best[0]:
            best = (
                candidate,
                {
                    "peer_weight": peer_weight,
                    "history_weight": history_weight,
                    "absolute_weight": absolute_weight,
                    "tuning_brier": brier,
                    "tuning_log_loss": logloss,
                },
            )
    if best is None:
        raise ValueError("No valid replacement-score weight combination")
    return best[1]


def _components(frame: pd.DataFrame, prior_reference: pd.DataFrame | None = None):
    result = frame.copy().reset_index(drop=True)
    result["peer"] = peer_percentile(result.proba, result.origin)
    result["history"] = historical_percentile(
        result.proba,
        result.country,
        result.origin,
        prior_reference=prior_reference,
    )
    result["absolute"] = result.proba
    return result


def build_oof_scores(validation_result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select score weights inside each outer fold's training predictions."""
    scores, weight_rows = [], []
    for fold in sorted(validation_result.ledger.outer_fold.unique()):
        test = validation_result.ledger.loc[
            validation_result.ledger.outer_fold.eq(fold)
        ].copy()
        tuning = validation_result.tuning_ledger.loc[
            validation_result.tuning_ledger.outer_fold.eq(fold)
        ].copy()
        tune_components = _components(tuning)
        selected = select_weights(tune_components)
        reference = tune_components[["country", "origin", "proba"]].rename(
            columns={"proba": "signal"}
        )
        test_components = _components(test, prior_reference=reference)
        for key, value in selected.items():
            if key.endswith("_weight"):
                test_components[key] = value
        test_components["replacement_index"] = (
            selected["peer_weight"] * test_components.peer
            + selected["history_weight"] * test_components.history
            + selected["absolute_weight"] * test_components.absolute
        ).clip(0, 1)
        test_components["replacement_score"] = 1 + 9 * test_components.replacement_index
        test_components["risk_category"] = score_category(
            test_components.replacement_score
        )
        scores.append(test_components)
        weight_rows.append({"outer_fold": int(fold), **selected})
    return pd.concat(scores, ignore_index=True), pd.DataFrame(weight_rows)


def score_category(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[0, 2, 4, 6, 8, 10.000001],
        labels=[
            "1-2: Very Low Risk",
            "3-4: Low Risk",
            "5-6: Moderate Risk",
            "7-8: High Risk",
            "9-10: Very High Risk",
        ],
        include_lowest=True,
    ).astype(str)


def score_diagnostics(scores: pd.DataFrame) -> dict:
    data = scores.copy()
    data["category_order"] = pd.cut(
        data.replacement_score,
        bins=[0, 2, 4, 6, 8, 10.000001],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )
    category = data.groupby("category_order", observed=True).agg(
        rows=("y", "size"),
        positives=("y", "sum"),
        event_rate=("y", "mean"),
        mean_score=("replacement_score", "mean"),
    ).reset_index()
    valid = category.loc[category.rows.ge(10)]
    probability = data.replacement_index.clip(1e-8, 1 - 1e-8)
    y = data.y.to_numpy(float)
    return {
        "rows": len(data),
        "positives": int(data.y.sum()),
        "brier": float(np.mean((y - probability.to_numpy()) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
        "score_event_rate_monotonic": bool(
            valid.event_rate.is_monotonic_increasing
        ) if len(valid) > 1 else False,
        "category_table": category.astype(object).where(
            pd.notna(category), None
        ).to_dict("records"),
        "mean_peer_weight": float(data.peer_weight.mean()),
        "mean_history_weight": float(data.history_weight.mean()),
        "mean_absolute_weight": float(data.absolute_weight.mean()),
    }


def latest_components(
    latest_probability: pd.Series,
    countries: pd.Series,
    origin_year: int,
    historical_oof: pd.DataFrame,
    weights: pd.Series | dict,
) -> pd.DataFrame:
    latest = pd.DataFrame({
        "country": countries.astype(str).to_numpy(),
        "origin": int(origin_year),
        "proba": pd.to_numeric(latest_probability, errors="raise").to_numpy(float),
    })
    latest["peer"] = peer_percentile(latest.proba, latest.origin)
    reference = historical_oof[["country", "origin", "proba"]].rename(
        columns={"proba": "signal"}
    )
    latest["history"] = historical_percentile(
        latest.proba,
        latest.country,
        latest.origin,
        prior_reference=reference,
    )
    latest["absolute"] = latest.proba
    peer_weight = float(weights["peer_weight"])
    history_weight = float(weights["history_weight"])
    absolute_weight = float(weights["absolute_weight"])
    latest["peer_weight"] = peer_weight
    latest["history_weight"] = history_weight
    latest["absolute_weight"] = absolute_weight
    latest["replacement_index"] = (
        peer_weight * latest.peer
        + history_weight * latest.history
        + absolute_weight * latest.absolute
    ).clip(0, 1)
    latest["replacement_score"] = 1 + 9 * latest.replacement_index
    latest["risk_category"] = score_category(latest.replacement_score)
    return latest
