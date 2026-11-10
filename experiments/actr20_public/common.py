"""Shared ACTR-20 public benchmark adapter used by TimeSafe experiments.

The external ACTR package is research-only and is not vendored into TimeSafe.
Before importing its Python code, this adapter validates every file listed in the
package's SHA-256 manifest.  The experiment therefore executes the exact signed-off
artifact rather than an arbitrary directory that merely has the same layout.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_BENCHMARK_SHA256 = (
    "dcc663779842c5701440f16015ceb36fe0555bc125f33f8b8f8e53ca9a02e128"
)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(root: str | Path) -> dict[str, Any]:
    """Verify all files declared by an ACTR ``MANIFEST.sha256``.

    The manifest is treated as an external artifact contract. Paths must remain
    inside ``root``; missing files, malformed lines, duplicate paths, or digest
    mismatches are fatal. Extra files are allowed because users may extract the
    package beside generated results, but they are not part of the verified input.
    """

    package_root = Path(root).expanduser().resolve()
    manifest = package_root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing ACTR manifest: {manifest}")

    checked: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed manifest line {line_number}: {raw_line!r}"
            ) from exc
        relative = relative.strip()
        if relative.startswith("./"):
            relative = relative[2:]
        if len(expected) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected):
            raise ValueError(
                f"Malformed SHA-256 on manifest line {line_number}: {expected!r}"
            )
        if relative in seen:
            raise ValueError(f"Duplicate manifest path: {relative!r}")
        seen.add(relative)

        candidate = (package_root / relative).resolve()
        try:
            candidate.relative_to(package_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes package root: {relative!r}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {relative}")
        observed = file_sha256(candidate)
        if observed.lower() != expected.lower():
            raise RuntimeError(
                f"Manifest digest mismatch for {relative}: "
                f"observed={observed}, expected={expected.lower()}"
            )
        checked.append(relative)

    if not checked:
        raise ValueError("ACTR manifest contains no file records.")

    verification_path = package_root / "PACKAGE_VERIFICATION.json"
    if not verification_path.is_file():
        raise FileNotFoundError("ACTR package is missing PACKAGE_VERIFICATION.json.")
    package_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if package_verification.get("status") != "PASS":
        raise RuntimeError("The external ACTR package does not declare PASS status.")
    declared_benchmark = package_verification.get("package", {}).get("benchmark_sha256")
    if declared_benchmark != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError(
            "Unexpected ACTR benchmark identity in PACKAGE_VERIFICATION.json."
        )

    return {
        "status": "PASS",
        "manifest": manifest.name,
        "checked_files": len(checked),
        "benchmark_sha256": declared_benchmark,
    }


def register_actr_package(actr_root: str | Path) -> Path:
    """Verify and register the external ACTR source tree."""

    root = Path(actr_root).expanduser().resolve()
    source = root / "src"
    data = root / "data" / "actr_deidentified_benchmark.npz"
    license_path = root / "LICENSE_RESEARCH_ONLY.md"
    if not source.is_dir() or not data.is_file() or not license_path.is_file():
        raise FileNotFoundError(
            "ACTR public package must contain src/, "
            "data/actr_deidentified_benchmark.npz, and LICENSE_RESEARCH_ONLY.md"
        )
    verify_sha256_manifest(root)
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    return root


def load_actr_data(actr_root: str | Path):
    root = register_actr_package(actr_root)
    from actr_public import load_benchmark

    return root, load_benchmark(root / "data" / "actr_deidentified_benchmark.npz")


def benchmark_to_panel(data, *, feedback_lag_sessions: int = 25) -> pd.DataFrame:
    """Convert ACTR benchmark arrays to a labeled panel used by model audits.

    ``target_available`` follows the public ACTR feedback convention: a target
    issued at anonymous session ``s`` becomes usable at session ``s + 25``.  The
    audit treats that availability as ``pre_open`` on the availability session,
    which is conservative relative to the prior session's after-close maturity.
    """

    index = pd.MultiIndex.from_arrays(
        [data.date_id.astype(int), data.asset_id.astype(int)],
        names=["date_id", "asset_id"],
    )
    panel = pd.DataFrame(
        {
            "year": data.year.astype(int),
            "cs_score": data.cs_score.astype(float),
            "ts_score": data.ts_score.astype(float),
            "target_rank": data.target_rank.astype(float),
        },
        index=index,
    )
    for column in range(data.cross_embedding.shape[1]):
        panel[f"cross_{column:02d}"] = data.cross_embedding[:, column]
    for column in range(data.temporal_embedding.shape[1]):
        panel[f"temporal_{column:02d}"] = data.temporal_embedding[:, column]

    sessions = pd.Index(np.unique(data.date_id.astype(int))).sort_values()
    position = {int(value): idx for idx, value in enumerate(sessions)}
    available: list[float] = []
    for date_id in panel.index.get_level_values("date_id"):
        target = position[int(date_id)] + int(feedback_lag_sessions)
        available.append(float(sessions[target]) if target < len(sessions) else np.nan)
    panel["target_available"] = available
    return panel.sort_index()


def anchor_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Public ACTR cross-anchor feature family."""

    return [
        "cs_score",
        "ts_score",
        *[column for column in panel.columns if column.startswith("cross_")],
    ]


def model_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Bounded ACTR-style joint feature family used by the audit workers."""

    return [
        *anchor_feature_columns(panel),
        *[column for column in panel.columns if column.startswith("temporal_")][:12],
    ]


def make_public_estimator(*, seed: int = 20261110, max_iter: int = 45):
    """Create the same low-capacity learner family used by public ACTR."""

    from actr_public.learners import make_low_capacity_regressor

    return make_low_capacity_regressor(
        seed=seed,
        max_iter=max_iter,
        max_depth=2,
        min_samples_leaf=80,
        learning_rate=0.05,
        l2_regularization=2.0,
    )


def _fit_model(estimator: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Any:
    try:
        estimator.fit(x, y, sample_weight=weights)
    except (TypeError, ValueError):
        estimator.fit(x, y, gradientboostingregressor__sample_weight=weights)
    return estimator


def _date_equal_weights(date_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(date_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def _prediction_series(frame: pd.DataFrame, checkpoint: int, values: np.ndarray) -> pd.Series:
    current = frame.index.get_level_values("date_id") == checkpoint
    index = frame.index[current]
    series = pd.Series(values, index=index, name="prediction", dtype=float)
    if len(series) >= 2:
        series = series.rank(method="average", pct=True)
    return series


def fit_predict_legal(estimator: Any, frame: pd.DataFrame, checkpoint: int) -> pd.Series:
    """Fit only on labels available by the current ACTR decision session."""

    dates = frame.index.get_level_values("date_id").to_numpy(dtype=int)
    available = frame["target_available"].to_numpy(dtype=float)
    legal_label = frame["target_available"].notna().to_numpy() & (
        available <= float(checkpoint)
    )
    train = (dates < int(checkpoint)) & legal_label & frame["target_rank"].notna().to_numpy()
    current = dates == int(checkpoint)
    if train.sum() < 500 or current.sum() < 3:
        return pd.Series(dtype=float, index=frame.index[:0], name="prediction")
    features = model_feature_columns(frame)
    weights = _date_equal_weights(dates[train])
    _fit_model(
        estimator,
        frame.loc[train, features].to_numpy(dtype=float),
        frame.loc[train, "target_rank"].to_numpy(dtype=float),
        weights,
    )
    prediction = estimator.predict(frame.loc[current, features].to_numpy(dtype=float))
    return _prediction_series(frame, int(checkpoint), np.asarray(prediction, dtype=float))


def fit_predict_immature_labels(
    estimator: Any, frame: pd.DataFrame, checkpoint: int
) -> pd.Series:
    """Negative control: train on every materialized past target, mature or not."""

    dates = frame.index.get_level_values("date_id").to_numpy(dtype=int)
    train = (dates < int(checkpoint)) & frame["target_rank"].notna().to_numpy()
    current = dates == int(checkpoint)
    if train.sum() < 500 or current.sum() < 3:
        return pd.Series(dtype=float, index=frame.index[:0], name="prediction")
    features = model_feature_columns(frame)
    weights = _date_equal_weights(dates[train])
    _fit_model(
        estimator,
        frame.loc[train, features].to_numpy(dtype=float),
        frame.loc[train, "target_rank"].to_numpy(dtype=float),
        weights,
    )
    prediction = estimator.predict(frame.loc[current, features].to_numpy(dtype=float))
    return _prediction_series(frame, int(checkpoint), np.asarray(prediction, dtype=float))


def fit_predict_full_sample(estimator: Any, frame: pd.DataFrame, checkpoint: int) -> pd.Series:
    """Negative control: ignore the checkpoint and fit on all materialized labels."""

    dates = frame.index.get_level_values("date_id").to_numpy(dtype=int)
    train = frame["target_rank"].notna().to_numpy()
    current = dates == int(checkpoint)
    if train.sum() < 500 or current.sum() < 3:
        return pd.Series(dtype=float, index=frame.index[:0], name="prediction")
    features = model_feature_columns(frame)
    weights = _date_equal_weights(dates[train])
    _fit_model(
        estimator,
        frame.loc[train, features].to_numpy(dtype=float),
        frame.loc[train, "target_rank"].to_numpy(dtype=float),
        weights,
    )
    prediction = estimator.predict(frame.loc[current, features].to_numpy(dtype=float))
    return _prediction_series(frame, int(checkpoint), np.asarray(prediction, dtype=float))
