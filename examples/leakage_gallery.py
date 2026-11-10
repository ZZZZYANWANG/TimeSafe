"""Compact examples of legal transforms, hidden leaks, and timing contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from timesafe_quant import (
    CausalPipeline,
    LabelContract,
    PipelineStep,
    audit_label_maturity,
    audit_pipeline_prefix_invariance,
    audit_prefix_invariance,
    build_label_intervals,
    make_report,
    render_markdown,
)
from timesafe_quant.demo import make_demo_panel


def causal_ema(data: pd.DataFrame) -> pd.Series:
    return (
        data["return"]
        .groupby(level="asset", group_keys=False)
        .ewm(span=20, adjust=False)
        .mean()
        .rename("ema")
    )


def leaking_global_zscore(data: pd.DataFrame) -> pd.Series:
    x = data["return"]
    return ((x - x.mean()) / x.std(ddof=0)).rename("global_zscore")


def leaking_centered_mean(data: pd.DataFrame) -> pd.Series:
    return (
        data["return"]
        .groupby(level="asset", group_keys=False)
        .rolling(window=11, center=True, min_periods=1)
        .mean()
        .rename("centered_mean")
    )


def leaking_full_sample_pca(data: pd.DataFrame) -> pd.Series:
    features = pd.DataFrame(index=data.index)
    features["r"] = data["return"]
    features["log_volume"] = np.log1p(data["volume"])
    matrix = features.to_numpy(dtype=float)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
    loading = vectors[0]
    if loading[np.argmax(np.abs(loading))] < 0:
        loading = -loading
    return pd.Series(matrix @ loading, index=data.index, name="pc1")


def leaking_backfill(data: pd.DataFrame) -> pd.Series:
    sparse = data["return"].where(data.groupby(level="asset").cumcount() % 7 != 0)
    return sparse.groupby(level="asset", group_keys=False).bfill().rename("backfilled")


def expanding_zscore(data: pd.DataFrame) -> pd.Series:
    grouped = data["return"].groupby(level="asset")
    mean = grouped.expanding(min_periods=20).mean().droplevel(0)
    std = grouped.expanding(min_periods=20).std(ddof=0).droplevel(0)
    return ((data["return"] - mean) / std.replace(0.0, np.nan)).rename("expanding_zscore")


def main() -> None:
    panel = make_demo_panel(sessions=120, assets=5)
    transforms = {
        "causal EMA": causal_ema,
        "expanding z-score": expanding_zscore,
        "global z-score (leak)": leaking_global_zscore,
        "centered rolling mean (leak)": leaking_centered_mean,
        "full-sample PCA (leak)": leaking_full_sample_pca,
        "future backfill (leak)": leaking_backfill,
    }

    for name, transform in transforms.items():
        result = audit_prefix_invariance(
            transform,
            panel,
            time_level="date",
            checkpoint_policy="all",
            min_history=40,
            transform_name=name,
        )
        print(
            f"{name:34s} {result.status.value:13s} "
            f"coverage={result.coverage:.1%} first={result.first_violation}"
        )

    pipeline = CausalPipeline(
        [
            PipelineStep("raw_return", lambda frame: frame[["return"]]),
            PipelineStep(
                "global_standardization",
                lambda frame: (frame - frame.mean()) / frame.std(ddof=0),
            ),
            PipelineStep("causal_smoothing", lambda frame: frame.ewm(span=10).mean()),
        ]
    )
    stage_result = audit_pipeline_prefix_invariance(
        pipeline,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=40,
    )
    print(f"first failing stage: {stage_result.first_failing_stage}")

    calendar = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    contract = LabelContract(
        execution_offset_sessions=1,
        holding_period_sessions=20,
        safety_gap_sessions=4,
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    valid_intervals = intervals[intervals["interval_valid"]].reset_index(drop=True)
    maturity_result = audit_label_maturity(
        valid_intervals,
        contract=contract,
        as_of=calendar[-1],
        as_of_event="after_close",
        calendar=calendar,
    )

    report = make_report(
        title="TimeSafe leakage gallery",
        pipeline=stage_result,
        maturity=maturity_result,
    )
    print("\n" + render_markdown(report))


if __name__ == "__main__":
    main()
