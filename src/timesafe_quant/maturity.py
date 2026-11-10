"""Label maturity, contract validation, purge, safety-gap, and embargo logic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._pandas import scalar_jsonable
from .timing import (
    EVENT_ORDER,
    EVENTS_PER_SESSION,
    SessionEvent,
    canonical_calendar,
    encode_points,
    normalize_event,
    point_key,
    series_point_le,
    shift_sessions,
)
from .types import (
    AuditStatus,
    Finding,
    LabelContractSummary,
    LabelMaturityAuditResult,
)

INTERVAL_SCHEMA_VERSION = 1
CONTRACT_FINGERPRINT_COLUMN = "_timesafe_contract_fingerprint"
CONTRACT_SCHEMA_COLUMN = "_timesafe_contract_schema"


@dataclass(frozen=True)
class LabelContract:
    """Session- and event-aware forward-label contract.

    ``holding_period_sessions`` counts trading-session holding intervals.  A later
    event in the entry session can complete the first interval.  If the requested
    exit event is equal to or earlier than the entry event, the first valid exit is
    in the following session.  Examples:

    - ``open(t+1) -> close``, one session: exit at ``close(t+1)``;
    - ``close(t+1) -> close``, one session: exit at ``close(t+2)``;
    - ``close(t) decision -> open(t+1) entry -> close``, twenty sessions:
      exit at ``close(t+20)``.
    """

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

        decision_event = normalize_event(self.decision_event)
        execution_event = normalize_event(self.execution_event)
        exit_event = normalize_event(self.exit_event)
        available_event = normalize_event(self.label_available_event)
        object.__setattr__(self, "decision_event", decision_event)
        object.__setattr__(self, "execution_event", execution_event)
        object.__setattr__(self, "exit_event", exit_event)
        object.__setattr__(self, "label_available_event", available_event)

        if point_key(self.execution_offset_sessions, execution_event) <= point_key(
            0, decision_event
        ):
            raise ValueError(
                "Execution must occur strictly after the decision. "
                f"Received decision=(t,{decision_event}) and "
                f"execution=(t+{self.execution_offset_sessions},{execution_event})."
            )
        if point_key(self.label_end_offset_sessions, exit_event) <= point_key(
            self.execution_offset_sessions, execution_event
        ):
            raise ValueError("Label exit must occur strictly after execution.")
        if point_key(self.label_available_offset_sessions, available_event) < point_key(
            self.label_end_offset_sessions, exit_event
        ):
            raise ValueError("Label availability cannot precede the label exit event.")

    @property
    def label_start_offset_sessions(self) -> int:
        return self.execution_offset_sessions

    @property
    def label_end_offset_sessions(self) -> int:
        rolls_to_next_session = int(
            EVENT_ORDER[str(self.exit_event)] <= EVENT_ORDER[str(self.execution_event)]
        )
        return (
            self.execution_offset_sessions
            + self.holding_period_sessions
            - 1
            + rolls_to_next_session
        )

    @property
    def label_available_offset_sessions(self) -> int:
        return self.label_end_offset_sessions + self.label_publication_lag_sessions

    @property
    def recommended_purge_sessions(self) -> int:
        return self.label_available_offset_sessions + self.safety_gap_sessions

    def payload(self) -> dict[str, Any]:
        return {
            "contract_schema_version": INTERVAL_SCHEMA_VERSION,
            "execution_offset_sessions": self.execution_offset_sessions,
            "holding_period_sessions": self.holding_period_sessions,
            "label_publication_lag_sessions": self.label_publication_lag_sessions,
            "safety_gap_sessions": self.safety_gap_sessions,
            "decision_event": str(self.decision_event),
            "execution_event": str(self.execution_event),
            "exit_event": str(self.exit_event),
            "label_available_event": str(self.label_available_event),
            "label_start_offset_sessions": self.label_start_offset_sessions,
            "label_end_offset_sessions": self.label_end_offset_sessions,
            "label_available_offset_sessions": self.label_available_offset_sessions,
            "recommended_purge_sessions": self.recommended_purge_sessions,
        }

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

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
            contract_schema_version=INTERVAL_SCHEMA_VERSION,
            fingerprint=self.fingerprint,
        )


@dataclass(frozen=True)
class IntervalContractValidation:
    """Evidence that an interval table matches its active contract."""

    status: AuditStatus
    fingerprint_present: bool
    fingerprint_match: bool | None
    recomputed: bool
    mismatched_rows: int
    mismatched_columns: tuple[str, ...]
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "fingerprint_present": self.fingerprint_present,
            "fingerprint_match": self.fingerprint_match,
            "recomputed": self.recomputed,
            "mismatched_rows": self.mismatched_rows,
            "mismatched_columns": list(self.mismatched_columns),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def build_label_intervals(
    decision_sessions: pd.Index | list[Any],
    *,
    calendar: pd.Index | list[Any] | None = None,
    contract: LabelContract | None = None,
) -> pd.DataFrame:
    """Map decisions to entry, exit, availability, and safety-gap endpoints."""

    contract = contract or LabelContract()
    decisions = pd.Index(decision_sessions)
    sessions = canonical_calendar(calendar if calendar is not None else decisions)
    positions = sessions.get_indexer(decisions)
    if (positions < 0).any():
        missing = decisions[positions < 0]
        raise ValueError(f"Decision sessions missing from calendar: {list(missing[:5])}")

    frame = pd.DataFrame(
        {
            "decision_time": decisions.to_numpy(),
            "decision_event": str(contract.decision_event),
            "label_start": shift_sessions(
                sessions, positions, contract.label_start_offset_sessions
            ),
            "label_start_event": str(contract.execution_event),
            "label_end": shift_sessions(
                sessions, positions, contract.label_end_offset_sessions
            ),
            "label_end_event": str(contract.exit_event),
            "label_available": shift_sessions(
                sessions, positions, contract.label_available_offset_sessions
            ),
            "label_available_event": str(contract.label_available_event),
            "label_available_with_gap": shift_sessions(
                sessions, positions, contract.recommended_purge_sessions
            ),
            "label_available_with_gap_event": str(contract.label_available_event),
            CONTRACT_SCHEMA_COLUMN: INTERVAL_SCHEMA_VERSION,
            CONTRACT_FINGERPRINT_COLUMN: contract.fingerprint,
        }
    )
    frame["interval_valid"] = frame[
        ["label_start", "label_end", "label_available"]
    ].notna().all(axis=1)
    frame["gap_valid"] = frame["label_available_with_gap"].notna()
    frame.attrs[CONTRACT_SCHEMA_COLUMN] = INTERVAL_SCHEMA_VERSION
    frame.attrs[CONTRACT_FINGERPRINT_COLUMN] = contract.fingerprint
    return frame


def _normalize_interval_events(
    intervals: pd.DataFrame, contract: LabelContract
) -> pd.DataFrame:
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
        frame[column] = frame[column].map(normalize_event)
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

    sessions = canonical_calendar(calendar)
    positions = sessions.get_indexer(pd.Index(intervals["label_available"]))
    frame = intervals.copy()
    frame["label_available_with_gap"] = shift_sessions(
        sessions, positions, contract.safety_gap_sessions
    )
    return frame


def _series_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(left) or pd.api.types.is_datetime64_any_dtype(
        right
    ):
        left_values = pd.to_datetime(left, errors="coerce")
        right_values = pd.to_datetime(right, errors="coerce")
        return left_values.eq(right_values) | (left_values.isna() & right_values.isna())
    return left.eq(right) | (left.isna() & right.isna())


def validate_label_intervals(
    intervals: pd.DataFrame,
    *,
    contract: LabelContract | None = None,
    calendar: pd.Index | list[Any] | None = None,
    require_recomputed: bool = False,
) -> IntervalContractValidation:
    """Validate interval provenance and, when possible, recompute every endpoint.

    A fingerprint checks that the declared contract did not change.  Recomputing from
    ``decision_time`` and the supplied calendar also detects tampered or stale interval
    values.  Split construction uses ``require_recomputed=True`` so it never relies on a
    declaration alone.
    """

    contract = contract or LabelContract()
    required = {
        "decision_time",
        "label_start",
        "label_end",
        "label_available",
    }
    missing = required.difference(intervals.columns)
    if missing:
        finding = Finding(
            code="TS-CONTRACT-201",
            status=AuditStatus.ERROR,
            message="The interval table is missing required columns.",
            details={"missing_columns": sorted(missing)},
        )
        return IntervalContractValidation(
            status=AuditStatus.ERROR,
            fingerprint_present=False,
            fingerprint_match=None,
            recomputed=False,
            mismatched_rows=0,
            mismatched_columns=tuple(sorted(missing)),
            findings=(finding,),
        )
    if intervals.empty:
        finding = Finding(
            code="TS-CONTRACT-021",
            status=AuditStatus.INCONCLUSIVE,
            message="The interval table is empty; no contract evidence exists.",
        )
        return IntervalContractValidation(
            status=AuditStatus.INCONCLUSIVE,
            fingerprint_present=False,
            fingerprint_match=None,
            recomputed=False,
            mismatched_rows=0,
            mismatched_columns=(),
            findings=(finding,),
        )

    fingerprint_present = CONTRACT_FINGERPRINT_COLUMN in intervals.columns
    fingerprint_match: bool | None = None
    findings: list[Finding] = []
    if fingerprint_present:
        observed = pd.Index(intervals[CONTRACT_FINGERPRINT_COLUMN].dropna().unique())
        fingerprint_match = len(observed) == 1 and observed[0] == contract.fingerprint
        if not fingerprint_match:
            findings.append(
                Finding(
                    code="TS-CONTRACT-101",
                    status=AuditStatus.FAIL,
                    message="The interval table fingerprint does not match the active contract.",
                    details={
                        "expected": contract.fingerprint,
                        "observed": observed.astype(str).tolist(),
                    },
                )
            )
    else:
        attr_value = intervals.attrs.get(CONTRACT_FINGERPRINT_COLUMN)
        if attr_value is not None:
            fingerprint_present = True
            fingerprint_match = str(attr_value) == contract.fingerprint
            if not fingerprint_match:
                findings.append(
                    Finding(
                        code="TS-CONTRACT-101",
                        status=AuditStatus.FAIL,
                        message="The interval table attribute fingerprint does not match the active contract.",
                        details={"expected": contract.fingerprint, "observed": str(attr_value)},
                    )
                )

    recomputed = calendar is not None
    mismatched_columns: list[str] = []
    mismatch_mask = np.zeros(len(intervals), dtype=bool)
    if recomputed:
        expected = build_label_intervals(
            intervals["decision_time"].tolist(), calendar=calendar, contract=contract
        ).reset_index(drop=True)
        observed_frame = _normalize_interval_events(intervals.reset_index(drop=True), contract)
        observed_frame = _ensure_gap_column(
            observed_frame, contract=contract, calendar=calendar
        )
        columns = [
            "decision_time",
            "decision_event",
            "label_start",
            "label_start_event",
            "label_end",
            "label_end_event",
            "label_available",
            "label_available_event",
            "label_available_with_gap",
            "label_available_with_gap_event",
        ]
        for column in columns:
            equal = _series_equal(observed_frame[column], expected[column])
            if not bool(equal.all()):
                mismatched_columns.append(column)
                mismatch_mask |= ~equal.to_numpy(dtype=bool)
        if mismatch_mask.any():
            findings.append(
                Finding(
                    code="TS-CONTRACT-102",
                    status=AuditStatus.FAIL,
                    message="Recomputed interval endpoints do not match the supplied table.",
                    details={
                        "mismatched_rows": int(mismatch_mask.sum()),
                        "mismatched_columns": mismatched_columns,
                    },
                )
            )

    if findings:
        return IntervalContractValidation(
            status=AuditStatus.FAIL,
            fingerprint_present=fingerprint_present,
            fingerprint_match=fingerprint_match,
            recomputed=recomputed,
            mismatched_rows=int(mismatch_mask.sum()),
            mismatched_columns=tuple(mismatched_columns),
            findings=tuple(findings),
        )

    if recomputed:
        finding = Finding(
            code="TS-CONTRACT-001",
            status=AuditStatus.PASS,
            message="The interval table was recomputed from the active contract and calendar.",
            details={"fingerprint": contract.fingerprint},
        )
        status = AuditStatus.PASS
    elif fingerprint_present and fingerprint_match:
        finding = Finding(
            code="TS-CONTRACT-011",
            status=AuditStatus.WARN,
            message=(
                "The contract fingerprint matches, but interval endpoints were not "
                "recomputed because no calendar was supplied."
            ),
            details={"fingerprint": contract.fingerprint},
        )
        status = AuditStatus.WARN
    else:
        finding = Finding(
            code="TS-CONTRACT-021",
            status=AuditStatus.INCONCLUSIVE,
            message=(
                "The interval table has neither recomputed endpoint evidence nor a "
                "matching contract fingerprint."
            ),
        )
        status = AuditStatus.INCONCLUSIVE

    if require_recomputed and not recomputed:
        status = AuditStatus.ERROR
        finding = Finding(
            code="TS-CONTRACT-201",
            status=status,
            message="A trading calendar is required for strict interval validation.",
        )

    return IntervalContractValidation(
        status=status,
        fingerprint_present=fingerprint_present,
        fingerprint_match=fingerprint_match,
        recomputed=recomputed,
        mismatched_rows=0,
        mismatched_columns=(),
        findings=(finding,),
    )


def _merge_encoded_intervals(
    starts: np.ndarray, ends: np.ndarray
) -> list[tuple[int, int]]:
    valid = (starts >= 0) & (ends >= 0)
    pairs = sorted(zip(starts[valid].tolist(), ends[valid].tolist()))
    merged: list[tuple[int, int]] = []
    for start, end in pairs:
        if not merged or start > merged[-1][1]:
            merged.append((int(start), int(end)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], int(end)))
    return merged


def _overlap_with_merged(
    starts: np.ndarray, ends: np.ndarray, merged: list[tuple[int, int]]
) -> np.ndarray:
    out = np.zeros(len(starts), dtype=bool)
    if not merged:
        return out
    merged_starts = np.asarray([item[0] for item in merged], dtype=np.int64)
    merged_ends = np.asarray([item[1] for item in merged], dtype=np.int64)
    valid = (starts >= 0) & (ends >= 0)
    positions = np.searchsorted(merged_starts, ends[valid], side="right") - 1
    valid_positions = positions >= 0
    indices = np.flatnonzero(valid)
    candidate_indices = indices[valid_positions]
    candidate_positions = positions[valid_positions]
    out[candidate_indices] = merged_ends[candidate_positions] >= starts[candidate_indices]
    return out


def _purge_diagnostics(
    frame: pd.DataFrame,
    *,
    candidate: np.ndarray,
    tests: np.ndarray,
    calendar: pd.Index | list[Any],
    embargo_sessions: int,
) -> dict[str, np.ndarray | list[tuple[int, int]]]:
    cal = canonical_calendar(calendar)
    decision_points = encode_points(
        frame["decision_time"], frame["decision_event"], calendar=cal
    )
    label_end_points = encode_points(
        frame["label_end"], frame["label_end_event"], calendar=cal
    )
    available_points = encode_points(
        frame["label_available"],
        frame["label_available_event"],
        calendar=cal,
    )
    gap_points = encode_points(
        frame["label_available_with_gap"],
        frame["label_available_with_gap_event"],
        calendar=cal,
    )
    test_valid = (decision_points >= 0) & (label_end_points >= 0) & (available_points >= 0)
    train_valid = test_valid & (gap_points >= 0)

    test_merged = _merge_encoded_intervals(
        decision_points[tests & test_valid], label_end_points[tests & test_valid]
    )
    base_overlap = _overlap_with_merged(decision_points, label_end_points, test_merged)
    effective_end = np.maximum(label_end_points, gap_points)
    effective_overlap = _overlap_with_merged(decision_points, effective_end, test_merged)
    safety_only = effective_overlap & ~base_overlap

    embargo = np.zeros(len(frame), dtype=bool)
    if embargo_sessions:
        for _, end in test_merged:
            end_session = end // EVENTS_PER_SESSION
            embargo_end_session = min(end_session + embargo_sessions, len(cal) - 1)
            embargo_end_point = (
                embargo_end_session * EVENTS_PER_SESSION
                + EVENT_ORDER[SessionEvent.AFTER_CLOSE.value]
            )
            embargo |= (decision_points > end) & (decision_points <= embargo_end_point)

    invalid_train = ~train_valid
    invalid_test = ~test_valid
    return {
        "test_merged": test_merged,
        "base_overlap": candidate & base_overlap,
        "safety_only": candidate & safety_only,
        "effective_overlap": candidate & effective_overlap,
        "embargo": candidate & embargo,
        "invalid": candidate & invalid_train,
        "invalid_test": tests & invalid_test,
    }


def _require_strict_contract(
    intervals: pd.DataFrame,
    *,
    contract: LabelContract,
    calendar: pd.Index | list[Any] | None,
) -> pd.DataFrame:
    validation = validate_label_intervals(
        intervals,
        contract=contract,
        calendar=calendar,
        require_recomputed=True,
    )
    if validation.status is not AuditStatus.PASS:
        details = validation.to_dict()
        raise ValueError(f"Interval contract validation failed: {details}")
    frame = _normalize_interval_events(intervals, contract)
    return _ensure_gap_column(frame, contract=contract, calendar=calendar)


def purged_train_test_masks(
    intervals: pd.DataFrame,
    *,
    candidate_train_mask: pd.Series | np.ndarray,
    test_mask: pd.Series | np.ndarray,
    contract: LabelContract | None = None,
    embargo_sessions: int = 0,
    calendar: pd.Index | list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Purge arbitrary training candidates against all test information blocks.

    Every training sample is represented by an information interval from its decision
    point through ``label_available_with_gap``.  It is removed when that interval
    overlaps any test decision-to-label-end interval.  A post-test embargo is applied
    after *each* merged test block, not only after the final block.
    """

    contract = contract or LabelContract()
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative.")
    if calendar is None:
        raise ValueError("calendar is required for strict purge construction.")

    candidate = np.asarray(candidate_train_mask, dtype=bool)
    tests = np.asarray(test_mask, dtype=bool)
    if len(candidate) != len(intervals) or len(tests) != len(intervals):
        raise ValueError("candidate_train_mask and test_mask must match intervals length.")
    if np.any(candidate & tests):
        raise ValueError("candidate_train_mask and test_mask must be disjoint.")

    frame = _require_strict_contract(
        intervals, contract=contract, calendar=calendar
    )
    if not tests.any():
        raise ValueError("test_mask must select at least one row.")

    diagnostics = _purge_diagnostics(
        frame,
        candidate=candidate,
        tests=tests,
        calendar=calendar,
        embargo_sessions=embargo_sessions,
    )
    invalid_test = np.asarray(diagnostics["invalid_test"], dtype=bool)
    if invalid_test.any():
        raise ValueError(
            "test_mask includes labels without a complete interval: decision, exit, or availability endpoint is missing."
        )

    remove = (
        np.asarray(diagnostics["effective_overlap"], dtype=bool)
        | np.asarray(diagnostics["embargo"], dtype=bool)
        | np.asarray(diagnostics["invalid"], dtype=bool)
    )
    clean = candidate & ~remove
    return clean, tests


def purged_walk_forward_masks(
    intervals: pd.DataFrame,
    *,
    test_start: Any,
    test_end: Any,
    contract: LabelContract | None = None,
    calendar: pd.Index | list[Any] | None = None,
    embargo_sessions: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a strict past-only walk-forward split.

    A post-test embargo cannot affect a training set restricted to dates before the
    test starts.  Passing a non-zero embargo therefore raises instead of pretending
    that it was applied.
    """

    if embargo_sessions:
        raise ValueError(
            "embargo_sessions is not applicable to a past-only walk-forward split; "
            "use purged_train_test_masks for arbitrary candidate training rows."
        )
    contract = contract or LabelContract()
    if calendar is None:
        raise ValueError("calendar is required for strict walk-forward construction.")
    decision_time = intervals["decision_time"]
    test_mask = ((decision_time >= test_start) & (decision_time <= test_end)).to_numpy(
        dtype=bool
    )
    candidate = (decision_time < test_start).to_numpy(dtype=bool)
    return purged_train_test_masks(
        intervals,
        candidate_train_mask=candidate,
        test_mask=test_mask,
        contract=contract,
        embargo_sessions=0,
        calendar=calendar,
    )


def audit_label_maturity(
    intervals: pd.DataFrame,
    *,
    contract: LabelContract | None = None,
    as_of: Any | None = None,
    as_of_event: str | SessionEvent = SessionEvent.AFTER_CLOSE,
    train_mask: pd.Series | np.ndarray | None = None,
    test_mask: pd.Series | np.ndarray | None = None,
    scope_mask: pd.Series | np.ndarray | None = None,
    embargo_sessions: int = 0,
    calendar: pd.Index | list[Any] | None = None,
) -> LabelMaturityAuditResult:
    """Audit contract consistency, label maturity, purge, gap, and embargo rules."""

    contract = contract or LabelContract()
    validation = validate_label_intervals(
        intervals, contract=contract, calendar=calendar
    )
    findings = list(validation.findings)
    if validation.status in {AuditStatus.FAIL, AuditStatus.ERROR}:
        return LabelMaturityAuditResult(
            status=validation.status,
            contract=contract.summary(),
            rows=len(intervals),
            mature_rows=0,
            immature_rows=len(intervals),
            split_overlap_rows=0,
            findings=findings,
            metadata={"contract_validation": validation.to_dict()},
        )

    frame = _normalize_interval_events(intervals, contract)
    frame = _ensure_gap_column(frame, contract=contract, calendar=calendar)
    as_of_event_value = normalize_event(as_of_event)

    if train_mask is not None or test_mask is not None:
        if train_mask is None or test_mask is None:
            raise ValueError("train_mask and test_mask must be supplied together.")
        train_array = np.asarray(train_mask, dtype=bool)
        test_array = np.asarray(test_mask, dtype=bool)
        if len(train_array) != len(frame) or len(test_array) != len(frame):
            raise ValueError("train_mask and test_mask must match intervals length.")
        if np.any(train_array & test_array):
            raise ValueError("train_mask and test_mask must be disjoint.")
    else:
        train_array = np.zeros(len(frame), dtype=bool)
        test_array = np.zeros(len(frame), dtype=bool)

    if scope_mask is not None:
        scope = np.asarray(scope_mask, dtype=bool)
        if len(scope) != len(frame):
            raise ValueError("scope_mask must match intervals length.")
    elif train_mask is not None:
        scope = train_array | test_array
    else:
        scope = np.ones(len(frame), dtype=bool)

    rows = int(scope.sum())
    if rows == 0:
        findings.append(
            Finding(
                code="TS-MATURITY-021",
                status=AuditStatus.INCONCLUSIVE,
                message="The audit scope contains no rows.",
            )
        )
        return LabelMaturityAuditResult(
            status=AuditStatus.INCONCLUSIVE,
            contract=contract.summary(),
            rows=0,
            mature_rows=0,
            immature_rows=0,
            split_overlap_rows=0,
            findings=findings,
            metadata={"contract_validation": validation.to_dict()},
        )

    valid = frame["label_available"].notna() & frame["label_end"].notna()
    if as_of is None:
        mature = valid.copy()
    else:
        mature = valid & series_point_le(
            frame["label_available"],
            frame["label_available_event"],
            as_of,
            as_of_event_value,
        )
    immature_scope = scope & ~mature.to_numpy(dtype=bool)
    mature_rows = int((scope & mature.to_numpy(dtype=bool)).sum())
    immature_rows = int(immature_scope.sum())

    overlap_rows = 0
    interval_overlap_rows = 0
    safety_gap_rows = 0
    embargo_rows = 0
    invalid_train_rows = 0
    if train_mask is not None and not test_array.any():
        findings.append(
            Finding(
                code="TS-PURGE-021",
                status=AuditStatus.INCONCLUSIVE,
                message="The supplied test mask selects no rows; purge was not exercised.",
            )
        )
    if train_mask is not None and not train_array.any():
        findings.append(
            Finding(
                code="TS-PURGE-022",
                status=AuditStatus.INCONCLUSIVE,
                message="The supplied training mask selects no rows; purge was not exercised.",
            )
        )
    if train_mask is not None:
        if calendar is None:
            findings.append(
                Finding(
                    code="TS-PURGE-201",
                    status=AuditStatus.ERROR,
                    message="calendar is required to audit purge and embargo rules.",
                )
            )
        elif test_array.any():
            diagnostics = _purge_diagnostics(
                frame,
                candidate=train_array,
                tests=test_array,
                calendar=calendar,
                embargo_sessions=embargo_sessions,
            )
            base = np.asarray(diagnostics["base_overlap"], dtype=bool)
            safety = np.asarray(diagnostics["safety_only"], dtype=bool)
            embargo = np.asarray(diagnostics["embargo"], dtype=bool)
            invalid = np.asarray(diagnostics["invalid"], dtype=bool)
            interval_overlap_rows = int(base.sum())
            safety_gap_rows = int(safety.sum())
            embargo_rows = int(embargo.sum())
            invalid_train_rows = int(invalid.sum())
            overlap_rows = int((base | safety | embargo | invalid).sum())

    if immature_rows:
        findings.append(
            Finding(
                code="TS-MATURITY-101",
                status=AuditStatus.FAIL,
                message="One or more scoped labels were unavailable at the declared audit time.",
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
                message="All scoped labels were mature at the declared audit time.",
                details={"mature_rows": mature_rows},
            )
        )

    if overlap_rows:
        findings.append(
            Finding(
                code="TS-PURGE-101",
                status=AuditStatus.FAIL,
                message="Training rows violate overlap, safety-gap, embargo, or validity rules.",
                details={
                    "violating_train_rows": overlap_rows,
                    "interval_overlap_rows": interval_overlap_rows,
                    "safety_gap_rows": safety_gap_rows,
                    "embargo_rows": embargo_rows,
                    "invalid_train_rows": invalid_train_rows,
                },
            )
        )
    elif train_mask is not None and test_array.any() and train_array.any():
        findings.append(
            Finding(
                code="TS-PURGE-001",
                status=AuditStatus.PASS,
                message="No training row violates overlap, safety-gap, embargo, or validity rules.",
                details={
                    "embargo_sessions": embargo_sessions,
                    "safety_gap_sessions": contract.safety_gap_sessions,
                },
            )
        )

    statuses = [finding.status for finding in findings]
    if AuditStatus.ERROR in statuses:
        status = AuditStatus.ERROR
    elif AuditStatus.FAIL in statuses:
        status = AuditStatus.FAIL
    elif AuditStatus.INCONCLUSIVE in statuses:
        status = AuditStatus.INCONCLUSIVE
    elif AuditStatus.WARN in statuses:
        status = AuditStatus.WARN
    else:
        status = AuditStatus.PASS

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
            "invalid_train_rows": invalid_train_rows,
            "contract_validation": validation.to_dict(),
        },
    )
