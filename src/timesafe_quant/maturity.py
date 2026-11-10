"""Label maturity, overlap, purge, and embargo calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._pandas import scalar_jsonable
from .types import (
    AuditStatus,
    Finding,
    LabelContractSummary,
    LabelMaturityAuditResult,
)


@dataclass(frozen=True)
class LabelContract:
    """Session-based label timing contract.

    Parameters
    ----------
    execution_offset_sessions:
        Number of sessions from the decision session to entry. For a signal formed after
        close and executed at the next open, use 1.
    holding_period_sessions:
        Number of sessions in the realized return interval. Entry at t+1 and exit at the
        close of t+20 corresponds to 20.
    label_publication_lag_sessions:
        Additional delay before the realized label is observable by the research process.
    safety_gap_sessions:
        Conservative extra purge buffer.
    """

    execution_offset_sessions: int = 1
    holding_period_sessions: int = 20
    label_publication_lag_sessions: int = 0
    safety_gap_sessions: int = 0
    decision_event: str = "close"
    execution_event: str = "open"
    exit_event: str = "close"

    def __post_init__(self) -> None:
        if self.execution_offset_sessions < 0:
            raise ValueError("execution_offset_sessions must be non-negative.")
        if self.holding_period_sessions <= 0:
            raise ValueError("holding_period_sessions must be positive.")
        if self.label_publication_lag_sessions < 0:
            raise ValueError("label_publication_lag_sessions must be non-negative.")
        if self.safety_gap_sessions < 0:
            raise ValueError("safety_gap_sessions must be non-negative.")

    @property
    def label_start_offset_sessions(self) -> int:
        return self.execution_offset_sessions

    @property
    def label_end_offset_sessions(self) -> int:
        return self.execution_offset_sessions + self.holding_period_sessions - 1

    @property
    def label_available_offset_sessions(self) -> int:
        return self.label_end_offset_sessions + self.label_publication_lag_sessions

    @property
    def recommended_purge_sessions(self) -> int:
        return self.label_available_offset_sessions + self.safety_gap_sessions

    def summary(self) -> LabelContractSummary:
        return LabelContractSummary(
            execution_offset_sessions=self.execution_offset_sessions,
            holding_period_sessions=self.holding_period_sessions,
            label_publication_lag_sessions=self.label_publication_lag_sessions,
            safety_gap_sessions=self.safety_gap_sessions,
            label_start_offset_sessions=self.label_start_offset_sessions,
            label_end_offset_sessions=self.label_end_offset_sessions,
            label_available_offset_sessions=self.label_available_offset_sessions,
            recommended_purge_sessions=self.recommended_purge_sessions,
        )


def build_label_intervals(
    decision_sessions: pd.Index | list[Any],
    *,
    calendar: pd.Index | list[Any] | None = None,
    contract: LabelContract | None = None,
) -> pd.DataFrame:
    """Map each decision session to entry, label end, and label availability sessions."""

    contract = contract or LabelContract()
    decisions = pd.Index(decision_sessions)
    sessions = pd.Index(calendar if calendar is not None else decisions).drop_duplicates().sort_values()
    if not sessions.is_unique:
        raise ValueError("calendar must contain unique sessions.")

    positions = sessions.get_indexer(decisions)
    if (positions < 0).any():
        missing = decisions[positions < 0]
        raise ValueError(f"Decision sessions missing from calendar: {list(missing[:5])}")

    def shifted(offset: int) -> pd.Index:
        target = positions + offset
        values: list[Any] = []
        missing_value: Any = pd.NaT if isinstance(sessions, pd.DatetimeIndex) else None
        for position in target:
            values.append(sessions[position] if position < len(sessions) else missing_value)
        if isinstance(sessions, pd.DatetimeIndex):
            return pd.DatetimeIndex(values)
        return pd.Index(values)

    frame = pd.DataFrame(
        {
            "decision_time": decisions.to_numpy(),
            "label_start": shifted(contract.label_start_offset_sessions),
            "label_end": shifted(contract.label_end_offset_sessions),
            "label_available": shifted(contract.label_available_offset_sessions),
        }
    )
    frame["interval_valid"] = frame[["label_start", "label_end", "label_available"]].notna().all(axis=1)
    return frame


def audit_label_maturity(
    intervals: pd.DataFrame,
    *,
    contract: LabelContract | None = None,
    as_of: Any | None = None,
    train_mask: pd.Series | np.ndarray | None = None,
    test_mask: pd.Series | np.ndarray | None = None,
    embargo_sessions: int = 0,
    calendar: pd.Index | list[Any] | None = None,
) -> LabelMaturityAuditResult:
    """Audit label availability and train/test interval overlap."""

    contract = contract or LabelContract()
    required = {"decision_time", "label_start", "label_end", "label_available"}
    missing_columns = required.difference(intervals.columns)
    if missing_columns:
        raise ValueError(f"intervals missing required columns: {sorted(missing_columns)}")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative.")

    valid = intervals["label_available"].notna()
    if as_of is None:
        mature = valid.copy()
    else:
        mature = valid & (intervals["label_available"] <= as_of)
    immature = ~mature

    overlap_rows = 0
    findings: list[Finding] = []
    if train_mask is not None or test_mask is not None:
        if train_mask is None or test_mask is None:
            raise ValueError("train_mask and test_mask must be supplied together.")
        train_mask_array = np.asarray(train_mask, dtype=bool)
        test_mask_array = np.asarray(test_mask, dtype=bool)
        if len(train_mask_array) != len(intervals) or len(test_mask_array) != len(intervals):
            raise ValueError("train_mask and test_mask must match intervals length.")
        if np.any(train_mask_array & test_mask_array):
            raise ValueError("train_mask and test_mask must be disjoint.")

        test_intervals = intervals.loc[test_mask_array & valid, ["decision_time", "label_end"]]
        train_intervals = intervals.loc[train_mask_array & valid, ["label_start", "label_end", "decision_time"]]
        overlap = np.zeros(len(train_intervals), dtype=bool)
        if not test_intervals.empty and not train_intervals.empty:
            for _, test_row in test_intervals.iterrows():
                overlap |= (
                    (train_intervals["label_start"] <= test_row["label_end"])
                    & (train_intervals["label_end"] >= test_row["decision_time"])
                ).to_numpy(dtype=bool)

            if embargo_sessions and calendar is not None:
                sessions = pd.Index(calendar).drop_duplicates().sort_values()
                test_end = test_intervals["label_end"].max()
                end_position = sessions.get_indexer([test_end])[0]
                if end_position >= 0:
                    embargo_end_position = min(end_position + embargo_sessions, len(sessions) - 1)
                    embargo_end = sessions[embargo_end_position]
                    overlap |= (
                        (train_intervals["decision_time"] > test_end)
                        & (train_intervals["decision_time"] <= embargo_end)
                    ).to_numpy(dtype=bool)
        overlap_rows = int(overlap.sum())

    rows = len(intervals)
    mature_rows = int(mature.sum())
    immature_rows = int(immature.sum())

    if immature_rows:
        findings.append(
            Finding(
                code="TS-MATURITY-101",
                status=AuditStatus.FAIL,
                message="One or more labels were not observable at the declared audit time.",
                details={"as_of": scalar_jsonable(as_of), "immature_rows": immature_rows},
            )
        )
    else:
        findings.append(
            Finding(
                code="TS-MATURITY-001",
                status=AuditStatus.PASS,
                message="All audited labels were mature at the declared audit time.",
                details={"as_of": scalar_jsonable(as_of), "mature_rows": mature_rows},
            )
        )

    if overlap_rows:
        findings.append(
            Finding(
                code="TS-PURGE-101",
                status=AuditStatus.FAIL,
                message="Training label intervals overlap the test interval; purge is insufficient.",
                details={
                    "overlap_rows": overlap_rows,
                    "recommended_purge_sessions": contract.recommended_purge_sessions,
                },
            )
        )
    elif train_mask is not None:
        findings.append(
            Finding(
                code="TS-PURGE-001",
                status=AuditStatus.PASS,
                message="No training label interval overlaps the test interval.",
                details={"embargo_sessions": embargo_sessions},
            )
        )

    status = AuditStatus.PASS
    if immature_rows or overlap_rows:
        status = AuditStatus.FAIL

    return LabelMaturityAuditResult(
        status=status,
        contract=contract.summary(),
        rows=rows,
        mature_rows=mature_rows,
        immature_rows=immature_rows,
        split_overlap_rows=overlap_rows,
        findings=findings,
        metadata={"as_of": scalar_jsonable(as_of), "embargo_sessions": embargo_sessions},
    )


def purged_walk_forward_masks(
    intervals: pd.DataFrame,
    *,
    test_start: Any,
    test_end: Any,
    embargo_sessions: int = 0,
    calendar: pd.Index | list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create past-only train and test masks with label-overlap purge.

    Training candidates are decisions strictly before ``test_start``. A candidate survives
    only if its label is available before ``test_start``. Test rows are decisions in the
    inclusive ``[test_start, test_end]`` interval.
    """

    required = {"decision_time", "label_available"}
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(f"intervals missing required columns: {sorted(missing)}")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative.")

    test_mask = (
        (intervals["decision_time"] >= test_start)
        & (intervals["decision_time"] <= test_end)
    ).to_numpy(dtype=bool)
    train_mask = (
        (intervals["decision_time"] < test_start)
        & intervals["label_available"].notna()
        & (intervals["label_available"] < test_start)
    ).to_numpy(dtype=bool)

    _ = embargo_sessions, calendar
    return train_mask, test_mask
