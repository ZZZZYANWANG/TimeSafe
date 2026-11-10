"""Label maturity, event ordering, overlap, purge, and embargo calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


class SessionEvent(str, Enum):
    """Supported within-session event order."""

    PRE_OPEN = "pre_open"
    OPEN = "open"
    INTRADAY = "intraday"
    CLOSE = "close"
    AFTER_CLOSE = "after_close"


_EVENT_ORDER = {
    SessionEvent.PRE_OPEN.value: 0,
    SessionEvent.OPEN.value: 1,
    SessionEvent.INTRADAY.value: 2,
    SessionEvent.CLOSE.value: 3,
    SessionEvent.AFTER_CLOSE.value: 4,
}


def _normalize_event(value: str | SessionEvent) -> str:
    event = value.value if isinstance(value, SessionEvent) else str(value)
    if event not in _EVENT_ORDER:
        raise ValueError(
            f"Unsupported session event {event!r}; choose one of {sorted(_EVENT_ORDER)}."
        )
    return event


def _point_key(session_offset: int, event: str) -> tuple[int, int]:
    return session_offset, _EVENT_ORDER[event]


def _point_le_scalar(
    left_session: Any,
    left_event: str,
    right_session: Any,
    right_event: str,
) -> bool:
    return bool(
        left_session < right_session
        or (
            left_session == right_session
            and _EVENT_ORDER[left_event] <= _EVENT_ORDER[right_event]
        )
    )


def _point_lt_scalar(
    left_session: Any,
    left_event: str,
    right_session: Any,
    right_event: str,
) -> bool:
    return bool(
        left_session < right_session
        or (
            left_session == right_session
            and _EVENT_ORDER[left_event] < _EVENT_ORDER[right_event]
        )
    )


def _series_point_le(
    left_session: pd.Series,
    left_event: pd.Series,
    right_session: Any,
    right_event: str,
) -> pd.Series:
    return (left_session < right_session) | (
        (left_session == right_session)
        & (left_event.map(_EVENT_ORDER) <= _EVENT_ORDER[right_event])
    )


def _series_point_lt(
    left_session: pd.Series,
    left_event: pd.Series,
    right_session: Any,
    right_event: str,
) -> pd.Series:
    return (left_session < right_session) | (
        (left_session == right_session)
        & (left_event.map(_EVENT_ORDER) < _EVENT_ORDER[right_event])
    )


@dataclass(frozen=True)
class LabelContract:
    """Session- and event-aware label timing contract."""

    execution_offset_sessions: int = 1
    holding_period_sessions: int = 20
    label_publication_lag_sessions: int = 0
    safety_gap_sessions: int = 0
    decision_event: str | SessionEvent = SessionEvent.CLOSE
    execution_event: str | SessionEvent = SessionEvent.OPEN
    exit_event: str | SessionEvent = SessionEvent.CLOSE
    label_available_event: str | SessionEvent = SessionEvent.AFTER_CLOSE

    def __post_init__(self) -> None:
        if self.execution_offset_sessions < 0:
            raise ValueError("execution_offset_sessions must be non-negative.")
        if self.holding_period_sessions <= 0:
            raise ValueError("holding_period_sessions must be positive.")
        if self.label_publication_lag_sessions < 0:
            raise ValueError("label_publication_lag_sessions must be non-negative.")
        if self.safety_gap_sessions < 0:
            raise ValueError("safety_gap_sessions must be non-negative.")

        decision_event = _normalize_event(self.decision_event)
        execution_event = _normalize_event(self.execution_event)
        exit_event = _normalize_event(self.exit_event)
        available_event = _normalize_event(self.label_available_event)
        object.__setattr__(self, "decision_event", decision_event)
        object.__setattr__(self, "execution_event", execution_event)
        object.__setattr__(self, "exit_event", exit_event)
        object.__setattr__(self, "label_available_event", available_event)

        if _point_key(self.execution_offset_sessions, execution_event) <= _point_key(
            0, decision_event
        ):
            raise ValueError(
                "Execution must occur strictly after the decision. "
                f"Received decision=(t,{decision_event}) and "
                f"execution=(t+{self.execution_offset_sessions},{execution_event})."
            )
        if _point_key(self.label_end_offset_sessions, exit_event) <= _point_key(
            self.execution_offset_sessions, execution_event
        ):
            raise ValueError("Label exit must occur strictly after execution.")
        if _point_key(self.label_available_offset_sessions, available_event) < _point_key(
            self.label_end_offset_sessions, exit_event
        ):
            raise ValueError("Label availability cannot precede the label exit event.")

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
            decision_event=str(self.decision_event),
            execution_event=str(self.execution_event),
            exit_event=str(self.exit_event),
            label_available_event=str(self.label_available_event),
            label_start_offset_sessions=self.label_start_offset_sessions,
            label_end_offset_sessions=self.label_end_offset_sessions,
            label_available_offset_sessions=self.label_available_offset_sessions,
            recommended_purge_sessions=self.recommended_purge_sessions,
        )


def _shift_sessions(
    sessions: pd.Index,
    positions: np.ndarray,
    offset: int,
) -> pd.Index:
    target = positions + offset
    missing_value: Any = pd.NaT if isinstance(sessions, pd.DatetimeIndex) else None
    values = [
        sessions[position] if 0 <= position < len(sessions) else missing_value
        for position in target
    ]
    if isinstance(sessions, pd.DatetimeIndex):
        return pd.DatetimeIndex(values)
    return pd.Index(values)


def build_label_intervals(
    decision_sessions: pd.Index | list[Any],
    *,
    calendar: pd.Index | list[Any] | None = None,
    contract: LabelContract | None = None,
) -> pd.DataFrame:
    """Map decisions to entry, exit, availability, and safety-gap endpoints."""

    contract = contract or LabelContract()
    decisions = pd.Index(decision_sessions)
    sessions = pd.Index(
        calendar if calendar is not None else decisions
    ).drop_duplicates().sort_values()
    if not sessions.is_unique:
        raise ValueError("calendar must contain unique sessions.")

    positions = sessions.get_indexer(decisions)
    if (positions < 0).any():
        missing = decisions[positions < 0]
        raise ValueError(f"Decision sessions missing from calendar: {list(missing[:5])}")

    frame = pd.DataFrame(
        {
            "decision_time": decisions.to_numpy(),
            "decision_event": str(contract.decision_event),
            "label_start": _shift_sessions(
                sessions, positions, contract.label_start_offset_sessions
            ),
            "label_start_event": str(contract.execution_event),
            "label_end": _shift_sessions(
                sessions, positions, contract.label_end_offset_sessions
            ),
            "label_end_event": str(contract.exit_event),
            "label_available": _shift_sessions(
                sessions, positions, contract.label_available_offset_sessions
            ),
            "label_available_event": str(contract.label_available_event),
            "label_available_with_gap": _shift_sessions(
                sessions, positions, contract.recommended_purge_sessions
            ),
            "label_available_with_gap_event": str(contract.label_available_event),
        }
    )
    frame["interval_valid"] = frame[
        ["label_start", "label_end", "label_available"]
    ].notna().all(axis=1)
    frame["gap_valid"] = frame["label_available_with_gap"].notna()
    return frame


def _ensure_event_columns(intervals: pd.DataFrame, contract: LabelContract) -> pd.DataFrame:
    frame = intervals.copy()
    defaults = {
        "decision_event": str(contract.decision_event),
        "label_start_event": str(contract.execution_event),
        "label_end_event": str(contract.exit_event),
        "label_available_event": str(contract.label_available_event),
        "label_available_with_gap_event": str(contract.label_available_event),
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
        frame[column] = frame[column].map(_normalize_event)
    return frame


def _ensure_gap_column(
    intervals: pd.DataFrame,
    *,
    contract: LabelContract,
    calendar: pd.Index | list[Any] | None,
) -> pd.DataFrame:
    if "label_available_with_gap" in intervals:
        return intervals
    if calendar is None:
        if contract.safety_gap_sessions:
            raise ValueError(
                "calendar is required to apply safety_gap_sessions when intervals "
                "lack label_available_with_gap."
            )
        frame = intervals.copy()
        frame["label_available_with_gap"] = frame["label_available"]
        return frame

    sessions = pd.Index(calendar).drop_duplicates().sort_values()
    positions = sessions.get_indexer(pd.Index(intervals["label_available"]))
    frame = intervals.copy()
    values: list[Any] = []
    missing_value: Any = pd.NaT if isinstance(sessions, pd.DatetimeIndex) else None
    for position in positions:
        target = position + contract.safety_gap_sessions
        values.append(
            sessions[target]
            if position >= 0 and target < len(sessions)
            else missing_value
        )
    frame["label_available_with_gap"] = values
    return frame


def audit_label_maturity(
    intervals: pd.DataFrame,
    *,
    contract: LabelContract | None = None,
    as_of: Any | None = None,
    as_of_event: str | SessionEvent = SessionEvent.AFTER_CLOSE,
    train_mask: pd.Series | np.ndarray | None = None,
    test_mask: pd.Series | np.ndarray | None = None,
    embargo_sessions: int = 0,
    calendar: pd.Index | list[Any] | None = None,
) -> LabelMaturityAuditResult:
    """Audit label availability, safety-gap purge, overlap, and optional embargo."""

    contract = contract or LabelContract()
    required = {"decision_time", "label_start", "label_end", "label_available"}
    missing_columns = required.difference(intervals.columns)
    if missing_columns:
        raise ValueError(f"intervals missing required columns: {sorted(missing_columns)}")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative.")

    frame = _ensure_event_columns(intervals, contract)
    frame = _ensure_gap_column(frame, contract=contract, calendar=calendar)
    as_of_event_value = _normalize_event(as_of_event)

    valid = frame["label_available"].notna()
    if as_of is None:
        mature = valid.copy()
    else:
        mature = valid & _series_point_le(
            frame["label_available"],
            frame["label_available_event"],
            as_of,
            as_of_event_value,
        )
    immature = ~mature

    overlap_rows = 0
    interval_overlap_rows = 0
    safety_gap_rows = 0
    embargo_rows = 0
    findings: list[Finding] = []

    if train_mask is not None or test_mask is not None:
        if train_mask is None or test_mask is None:
            raise ValueError("train_mask and test_mask must be supplied together.")
        train_array = np.asarray(train_mask, dtype=bool)
        test_array = np.asarray(test_mask, dtype=bool)
        if len(train_array) != len(frame) or len(test_array) != len(frame):
            raise ValueError("train_mask and test_mask must match intervals length.")
        if np.any(train_array & test_array):
            raise ValueError("train_mask and test_mask must be disjoint.")

        test_rows = frame.loc[test_array & valid]
        train_rows = frame.loc[train_array & valid]
        overlap = np.zeros(len(train_rows), dtype=bool)
        gap_violation = np.zeros(len(train_rows), dtype=bool)
        embargo_violation = np.zeros(len(train_rows), dtype=bool)

        if not test_rows.empty and not train_rows.empty:
            for _, test_row in test_rows.iterrows():
                test_start_session = test_row["decision_time"]
                test_start_event = test_row["decision_event"]
                test_end_session = test_row["label_end"]
                test_end_event = test_row["label_end_event"]

                train_starts_before_test_end = (
                    (train_rows["label_start"] < test_end_session)
                    | (
                        (train_rows["label_start"] == test_end_session)
                        & (
                            train_rows["label_start_event"].map(_EVENT_ORDER)
                            <= _EVENT_ORDER[test_end_event]
                        )
                    )
                )
                train_ends_after_test_start = (
                    (train_rows["label_end"] > test_start_session)
                    | (
                        (train_rows["label_end"] == test_start_session)
                        & (
                            train_rows["label_end_event"].map(_EVENT_ORDER)
                            >= _EVENT_ORDER[test_start_event]
                        )
                    )
                )
                overlap |= (
                    train_starts_before_test_end & train_ends_after_test_start
                ).to_numpy(dtype=bool)

                before_test = _series_point_lt(
                    train_rows["decision_time"],
                    train_rows["decision_event"],
                    test_start_session,
                    test_start_event,
                )
                gap_is_safe = _series_point_lt(
                    train_rows["label_available_with_gap"],
                    train_rows["label_available_with_gap_event"],
                    test_start_session,
                    test_start_event,
                )
                gap_violation |= (before_test & ~gap_is_safe).to_numpy(dtype=bool)

            if embargo_sessions:
                if calendar is None:
                    raise ValueError("calendar is required when embargo_sessions > 0.")
                sessions = pd.Index(calendar).drop_duplicates().sort_values()
                latest_test_end = test_rows["label_end"].max()
                position = sessions.get_indexer([latest_test_end])[0]
                if position < 0:
                    raise ValueError("Test label end is missing from calendar.")
                embargo_end_position = min(
                    position + embargo_sessions, len(sessions) - 1
                )
                embargo_end = sessions[embargo_end_position]
                after_test = train_rows["decision_time"] > latest_test_end
                before_embargo_end = train_rows["decision_time"] <= embargo_end
                embargo_violation |= (
                    after_test & before_embargo_end
                ).to_numpy(dtype=bool)

        interval_overlap_rows = int(overlap.sum())
        safety_gap_rows = int(gap_violation.sum())
        embargo_rows = int(embargo_violation.sum())
        overlap_rows = int((overlap | gap_violation | embargo_violation).sum())

    rows = len(frame)
    mature_rows = int(mature.sum())
    immature_rows = int(immature.sum())

    if immature_rows:
        findings.append(
            Finding(
                code="TS-MATURITY-101",
                status=AuditStatus.FAIL,
                message="One or more labels were not observable at the declared audit time.",
                details={
                    "as_of": scalar_jsonable(as_of),
                    "as_of_event": as_of_event_value,
                    "immature_rows": immature_rows,
                },
            )
        )
    else:
        findings.append(
            Finding(
                code="TS-MATURITY-001",
                status=AuditStatus.PASS,
                message="All audited labels were mature at the declared audit time.",
                details={
                    "as_of": scalar_jsonable(as_of),
                    "as_of_event": as_of_event_value,
                    "mature_rows": mature_rows,
                },
            )
        )

    if overlap_rows:
        findings.append(
            Finding(
                code="TS-PURGE-101",
                status=AuditStatus.FAIL,
                message=(
                    "Training rows violate label-overlap, safety-gap, or embargo rules."
                ),
                details={
                    "violating_train_rows": overlap_rows,
                    "interval_overlap_rows": interval_overlap_rows,
                    "safety_gap_rows": safety_gap_rows,
                    "embargo_rows": embargo_rows,
                    "recommended_purge_sessions": contract.recommended_purge_sessions,
                },
            )
        )
    elif train_mask is not None:
        findings.append(
            Finding(
                code="TS-PURGE-001",
                status=AuditStatus.PASS,
                message="No training row violates overlap, safety-gap, or embargo rules.",
                details={
                    "embargo_sessions": embargo_sessions,
                    "safety_gap_sessions": contract.safety_gap_sessions,
                },
            )
        )

    status = AuditStatus.FAIL if immature_rows or overlap_rows else AuditStatus.PASS
    return LabelMaturityAuditResult(
        status=status,
        contract=contract.summary(),
        rows=rows,
        mature_rows=mature_rows,
        immature_rows=immature_rows,
        split_overlap_rows=overlap_rows,
        findings=findings,
        metadata={
            "as_of": scalar_jsonable(as_of),
            "as_of_event": as_of_event_value,
            "embargo_sessions": embargo_sessions,
            "interval_overlap_rows": interval_overlap_rows,
            "safety_gap_rows": safety_gap_rows,
            "embargo_rows": embargo_rows,
        },
    )


def purged_walk_forward_masks(
    intervals: pd.DataFrame,
    *,
    test_start: Any,
    test_end: Any,
    contract: LabelContract | None = None,
    calendar: pd.Index | list[Any] | None = None,
    embargo_sessions: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a past-only walk-forward split with the declared safety gap.

    Post-test embargo is not meaningful for a strictly past-only training set. A
    non-zero ``embargo_sessions`` therefore raises instead of being silently ignored;
    use :func:`purged_train_test_masks` for arbitrary train/test candidates.
    """

    if embargo_sessions:
        raise ValueError(
            "embargo_sessions is not applicable to a past-only walk-forward split; "
            "use purged_train_test_masks for arbitrary candidate training rows."
        )
    contract = contract or LabelContract()
    required = {"decision_time", "label_available"}
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(f"intervals missing required columns: {sorted(missing)}")

    frame = _ensure_event_columns(intervals, contract)
    frame = _ensure_gap_column(frame, contract=contract, calendar=calendar)
    test_mask = (
        (frame["decision_time"] >= test_start)
        & (frame["decision_time"] <= test_end)
    ).to_numpy(dtype=bool)

    safe_before_test = _series_point_lt(
        frame["label_available_with_gap"],
        frame["label_available_with_gap_event"],
        test_start,
        str(contract.decision_event),
    )
    train_mask = (
        (frame["decision_time"] < test_start)
        & frame["label_available_with_gap"].notna()
        & safe_before_test
    ).to_numpy(dtype=bool)
    return train_mask, test_mask


def purged_train_test_masks(
    intervals: pd.DataFrame,
    *,
    candidate_train_mask: pd.Series | np.ndarray,
    test_mask: pd.Series | np.ndarray,
    contract: LabelContract | None = None,
    embargo_sessions: int = 0,
    calendar: pd.Index | list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Purge arbitrary candidate training rows and apply a post-test embargo."""

    contract = contract or LabelContract()
    candidate = np.asarray(candidate_train_mask, dtype=bool)
    tests = np.asarray(test_mask, dtype=bool)
    if len(candidate) != len(intervals) or len(tests) != len(intervals):
        raise ValueError("candidate_train_mask and test_mask must match intervals length.")
    if np.any(candidate & tests):
        raise ValueError("candidate_train_mask and test_mask must be disjoint.")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative.")

    frame = _ensure_event_columns(intervals, contract)
    frame = _ensure_gap_column(frame, contract=contract, calendar=calendar)
    test_rows = frame.iloc[np.flatnonzero(tests)]
    clean = candidate.copy()
    if test_rows.empty:
        return clean, tests

    for position in np.flatnonzero(candidate):
        train_row = frame.iloc[position]
        violates = False
        for _, test_row in test_rows.iterrows():
            intervals_overlap = _point_le_scalar(
                train_row["label_start"],
                train_row["label_start_event"],
                test_row["label_end"],
                test_row["label_end_event"],
            ) and _point_le_scalar(
                test_row["decision_time"],
                test_row["decision_event"],
                train_row["label_end"],
                train_row["label_end_event"],
            )
            if intervals_overlap:
                violates = True
                break

            train_precedes_test = _point_lt_scalar(
                train_row["decision_time"],
                train_row["decision_event"],
                test_row["decision_time"],
                test_row["decision_event"],
            )
            if train_precedes_test:
                gap_safe = _point_lt_scalar(
                    train_row["label_available_with_gap"],
                    train_row["label_available_with_gap_event"],
                    test_row["decision_time"],
                    test_row["decision_event"],
                )
                if not gap_safe:
                    violates = True
                    break
        if violates:
            clean[position] = False

    if embargo_sessions:
        if calendar is None:
            raise ValueError("calendar is required when embargo_sessions > 0.")
        sessions = pd.Index(calendar).drop_duplicates().sort_values()
        latest_test_end = test_rows["label_end"].max()
        position = sessions.get_indexer([latest_test_end])[0]
        if position < 0:
            raise ValueError("Test label end is missing from calendar.")
        embargo_end = sessions[min(position + embargo_sessions, len(sessions) - 1)]
        post_test_embargo = (
            (frame["decision_time"] > latest_test_end)
            & (frame["decision_time"] <= embargo_end)
        ).to_numpy(dtype=bool)
        clean &= ~post_test_embargo

    return clean, tests
