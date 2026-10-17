"""Fit an Isolation Forest to numeric tabular data and rank unusual rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Detection:
    """Results aligned with the input row order. Lower scores are stranger."""

    scores: np.ndarray
    flags: np.ndarray
    main_signals: list[str]


def detect(
    rows: Sequence[dict[str, object]],
    features: Sequence[str],
    *,
    contamination: float = 0.05,
    seed: int = 42,
) -> Detection:
    """Fit Isolation Forest and return anomaly scores, flags, and top signals.

    ``main_signals`` is a model-agnostic hint: for each flagged row, it reports
    the feature furthest from its median in robust-IQR units. It is not a
    causal explanation of the Isolation Forest prediction.
    """
    if not rows:
        raise ValueError("No data rows were provided.")
    if not features:
        raise ValueError("Select at least one numeric feature.")
    if not 0.0 < contamination <= 0.5:
        raise ValueError("Contamination must be greater than 0 and at most 0.5.")

    matrix = np.array(
        [[_as_number(row.get(feature), feature) for feature in features] for row in rows],
        dtype=float,
    )
    if np.isinf(matrix).any():
        raise ValueError("Infinite values are not supported; clean the CSV first.")
    if np.isnan(matrix).all(axis=0).any():
        bad = [features[i] for i, all_missing in enumerate(np.isnan(matrix).all(axis=0)) if all_missing]
        raise ValueError("Feature(s) contain no numeric values: " + ", ".join(bad))
    if len(rows) < 3:
        raise ValueError("At least three rows are required to fit the detector.")

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    pipeline.fit(matrix)
    forest = pipeline.steps[-1][1]
    transformed = pipeline[:-1].transform(matrix)
    scores = forest.score_samples(transformed)
    flags = forest.predict(transformed) == -1
    signals = _top_signals(matrix, features, flags)
    return Detection(scores=scores, flags=flags, main_signals=signals)


def _as_number(value: object, feature: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature {feature!r} contains a non-numeric value: {value!r}") from exc


def _top_signals(
    matrix: np.ndarray, features: Sequence[str], flags: np.ndarray
) -> list[str]:
    medians = np.nanmedian(matrix, axis=0)
    q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
    scales = q75 - q25
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    result: list[str] = []
    for row, flagged in zip(matrix, flags):
        if not flagged:
            result.append("")
            continue
        distance = np.abs(row - medians) / scales
        distance[~np.isfinite(distance)] = -1
        result.append(str(features[int(np.argmax(distance))]))
    return result
