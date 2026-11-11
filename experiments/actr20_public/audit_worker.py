"""Isolated TimeSafe worker for one ACTR-20 audit.

Each invocation performs exactly one bounded audit in a fresh interpreter.  This
avoids cross-audit native-thread state retained by repeated scikit-learn
histogram-boosting fits on some BLAS/OpenMP combinations while preserving the
actual public ACTR learner and data distribution.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    atomic_write_json,
    benchmark_to_panel,
    build_audit_identity,
    fit_predict_full_sample,
    fit_predict_immature_labels,
    fit_predict_legal,
    load_actr_data,
    make_public_estimator,
)
from timesafe_quant import (
    AvailabilityRule,
    LabelContract,
    audit_label_maturity,
    audit_refit_availability_invariance,
    audit_refit_determinism,
    audit_prefix_invariance,
    audit_refit_prefix_invariance,
    build_label_intervals,
    purged_walk_forward_masks,
)

from audit_plan import AUDIT_NAMES


def _thread_limit_context():
    """Use sklearn's mature threadpoolctl dependency when available."""

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - sklearn normally provides it
        return nullcontext()
    return threadpool_limits(limits=1)


def _load_audit_panel(actr_root: str | Path, experiment_run_id: str):
    root, data = load_actr_data(actr_root)
    panel = benchmark_to_panel(data)
    # Pre-registered subset: all sessions and the first 24 anonymized assets.
    # The full benchmark is reproduced separately on every row.
    audit_assets = np.sort(panel.index.get_level_values("asset_id").unique())[:24]
    panel = panel[panel.index.get_level_values("asset_id").isin(audit_assets)].copy()
    sessions = np.sort(panel.index.get_level_values("date_id").unique().to_numpy(dtype=int))
    checkpoints = [value for value in (1000, 1200, 1450, 1700) if value in set(sessions)]
    identity = build_audit_identity(
        root=root,
        data=data,
        panel=panel,
        audit_assets=audit_assets,
        sessions=sessions,
        checkpoints=checkpoints,
        experiment_run_id=experiment_run_id,
    )
    return panel, sessions, audit_assets, checkpoints, identity


def run_one(
    audit: str, actr_root: str | Path, experiment_run_id: str
) -> dict[str, Any]:
    if audit not in AUDIT_NAMES:
        raise ValueError(f"Unknown audit {audit!r}; choose one of {AUDIT_NAMES}")

    panel, sessions, audit_assets, checkpoints, identity = _load_audit_panel(
        actr_root, experiment_run_id
    )

    def factory() -> Any:
        return make_public_estimator(seed=20261110, max_iter=45)

    rule = AvailabilityRule(
        value_column="target_rank",
        available_at_column="target_available",
        available_event="pre_open",
    )

    with _thread_limit_context():
        if audit == "mature_feedback_prefix":
            from actr_public.feedback import build_mature_feedback
            from actr_public.ranking import date_rank
            import pandas as pd

            def feedback_transform(frame):
                dates = frame.index.get_level_values("date_id").to_numpy(dtype=int)
                assets = frame.index.get_level_values("asset_id").to_numpy(dtype=int)
                anchor = date_rank(
                    dates,
                    0.60 * frame["cs_score"].to_numpy(dtype=float)
                    + 0.40 * frame["ts_score"].to_numpy(dtype=float),
                )
                values, names = build_mature_feedback(
                    date_id=dates,
                    asset_id=assets,
                    anchor=anchor,
                    target_rank=frame["target_rank"].to_numpy(dtype=float),
                    availability_lag=25,
                )
                return pd.DataFrame(values, index=frame.index, columns=names)

            # Exhaustive evidence on a bounded, pre-registered 921-session window.
            # This keeps the exact ACTR feedback transform cheap enough to test every
            # eligible checkpoint while the model refits remain broad sampled audits.
            feedback_panel = panel[
                panel.index.get_level_values("date_id") <= 920
            ].copy()
            result = audit_prefix_invariance(
                feedback_transform,
                feedback_panel,
                time_level="date_id",
                checkpoint_policy="all",
                min_history=900,
                transform_name="actr_mature_feedback",
            )
        elif audit == "legal_refit":
            result = audit_refit_prefix_invariance(
                factory,
                fit_predict_legal,
                panel,
                time_level="date_id",
                checkpoints=checkpoints,
                min_history=900,
                audit_name="actr_legal_refit",
            )
        elif audit == "full_sample_negative_control":
            result = audit_refit_prefix_invariance(
                factory,
                fit_predict_full_sample,
                panel,
                time_level="date_id",
                checkpoints=checkpoints,
                min_history=900,
                audit_name="actr_full_sample_negative_control",
            )
        elif audit == "legal_availability":
            result = audit_refit_availability_invariance(
                factory,
                fit_predict_legal,
                panel,
                rules=[rule],
                time_level="date_id",
                checkpoints=checkpoints,
                min_history=900,
                decision_event="close",
                audit_name="actr_legal_availability",
            )
        elif audit == "immature_label_negative_control":
            result = audit_refit_availability_invariance(
                factory,
                fit_predict_immature_labels,
                panel,
                rules=[rule],
                time_level="date_id",
                checkpoints=checkpoints,
                min_history=900,
                decision_event="close",
                audit_name="actr_immature_label_negative_control",
            )
        elif audit == "refit_determinism":
            result = audit_refit_determinism(
                factory,
                fit_predict_legal,
                panel,
                time_level="date_id",
                checkpoints=checkpoints[:2],
                min_history=900,
                repetitions=2,
                audit_name="actr_refit_determinism",
            )
        else:
            contract = LabelContract(
                decision_event="close",
                execution_offset_sessions=1,
                execution_event="open",
                holding_period_sessions=20,
                exit_event="close",
                label_available_event="after_close",
                safety_gap_sessions=4,
            )
            intervals = build_label_intervals(sessions, calendar=sessions, contract=contract)
            train_mask, test_mask = purged_walk_forward_masks(
                intervals,
                test_start=950,
                test_end=1175,
                contract=contract,
                calendar=sessions,
            )
            result = audit_label_maturity(
                intervals,
                contract=contract,
                as_of=int(sessions[-1]),
                train_mask=train_mask,
                test_mask=test_mask,
                calendar=sessions,
            )

    return {
        "audit": audit,
        "identity": identity,
        "audit_asset_count": int(len(audit_assets)),
        "audit_rows": int(len(panel)),
        "checkpoints": [item.checkpoint for item in getattr(result, "checkpoints", [])],
        "result": result.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, choices=AUDIT_NAMES)
    parser.add_argument("--actr-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    # Set before model fits; run_all also sets these before interpreter startup.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    payload = run_one(args.audit, args.actr_root, args.run_id)
    atomic_write_json(args.output, payload)
    print(json.dumps({"audit": args.audit, "status": payload["result"]["status"]}))


if __name__ == "__main__":
    main()
