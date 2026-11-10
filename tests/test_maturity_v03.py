import numpy as np
import pandas as pd
import pytest

from timesafe_quant import (
    AuditStatus,
    LabelContract,
    SessionEvent,
    audit_label_maturity,
    build_label_intervals,
    purged_train_test_masks,
    validate_label_intervals,
)
from timesafe_quant.timing import EVENT_ORDER, EVENTS_PER_SESSION


def _point(session_pos: int, event: str) -> int:
    return session_pos * EVENTS_PER_SESSION + EVENT_ORDER[event]


def _naive_purge(intervals, candidate, tests, calendar, embargo_sessions):
    position = {value: idx for idx, value in enumerate(calendar)}

    def encode(session, event):
        if pd.isna(session) or session not in position:
            return -1
        return _point(position[session], event)

    test_intervals = []
    for idx in np.flatnonzero(tests):
        row = intervals.iloc[idx]
        start = encode(row.decision_time, row.decision_event)
        end = encode(row.label_end, row.label_end_event)
        if start < 0 or end < 0:
            raise ValueError("invalid test interval")
        test_intervals.append((start, end))
    test_intervals.sort()
    merged = []
    for start, end in test_intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    clean = candidate.copy()
    for idx in np.flatnonzero(candidate):
        row = intervals.iloc[idx]
        start = encode(row.decision_time, row.decision_event)
        label_end = encode(row.label_end, row.label_end_event)
        gap_end = encode(
            row.label_available_with_gap,
            row.label_available_with_gap_event,
        )
        if min(start, label_end, gap_end) < 0:
            clean[idx] = False
            continue
        effective_end = max(label_end, gap_end)
        if any(start <= test_end and effective_end >= test_start for test_start, test_end in merged):
            clean[idx] = False
            continue
        if embargo_sessions:
            for _, test_end in merged:
                end_session = test_end // EVENTS_PER_SESSION
                embargo_end = min(end_session + embargo_sessions, len(calendar) - 1)
                if test_end < start <= _point(embargo_end, SessionEvent.AFTER_CLOSE.value):
                    clean[idx] = False
                    break
    return clean


def test_holding_period_respects_entry_and_exit_events():
    open_to_close = LabelContract(
        decision_event="pre_open",
        execution_offset_sessions=0,
        execution_event="open",
        holding_period_sessions=1,
        exit_event="close",
    )
    close_to_close = LabelContract(
        decision_event="close",
        execution_offset_sessions=1,
        execution_event="close",
        holding_period_sessions=1,
        exit_event="close",
    )
    assert open_to_close.label_end_offset_sessions == 0
    assert close_to_close.label_end_offset_sessions == 2


def test_impossible_same_session_execution_is_rejected():
    with pytest.raises(ValueError, match="Execution must occur strictly after"):
        LabelContract(
            decision_event="close",
            execution_offset_sessions=0,
            execution_event="open",
        )


def test_contract_fingerprint_and_endpoint_recomputation_detect_mismatch():
    calendar = pd.bdate_range("2024-01-02", periods=80)
    contract = LabelContract(holding_period_sessions=20, safety_gap_sessions=3)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)

    wrong_contract = LabelContract(holding_period_sessions=10, safety_gap_sessions=3)
    result = validate_label_intervals(
        intervals,
        contract=wrong_contract,
        calendar=calendar,
    )
    assert result.status is AuditStatus.FAIL

    tampered = intervals.copy()
    tampered.loc[0, "label_end"] = calendar[3]
    result = validate_label_intervals(tampered, contract=contract, calendar=calendar)
    assert result.status is AuditStatus.FAIL
    assert result.mismatched_rows == 1
    assert "label_end" in result.mismatched_columns


def test_split_constructor_rejects_stale_intervals():
    calendar = pd.bdate_range("2024-01-02", periods=80)
    contract = LabelContract(holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    intervals.loc[5, "label_available"] = calendar[5]
    candidate = np.ones(len(intervals), dtype=bool)
    tests = np.zeros(len(intervals), dtype=bool)
    tests[40:43] = True
    candidate[tests] = False
    with pytest.raises(ValueError, match="Interval contract validation failed"):
        purged_train_test_masks(
            intervals,
            candidate_train_mask=candidate,
            test_mask=tests,
            contract=contract,
            calendar=calendar,
        )


def test_embargo_is_applied_after_each_disjoint_test_block():
    calendar = pd.bdate_range("2024-01-02", periods=60)
    contract = LabelContract(
        decision_event="pre_open",
        execution_offset_sessions=0,
        execution_event="open",
        holding_period_sessions=1,
        exit_event="close",
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    tests = np.zeros(len(intervals), dtype=bool)
    tests[[10, 30]] = True
    candidate = ~tests
    clean, _ = purged_train_test_masks(
        intervals,
        candidate_train_mask=candidate,
        test_mask=tests,
        contract=contract,
        embargo_sessions=2,
        calendar=calendar,
    )
    for idx in [11, 12, 31, 32]:
        assert not clean[idx]
    assert clean[13]
    assert clean[33]


def test_purge_matches_independent_oracle_across_many_contracts():
    rng = np.random.default_rng(20261110)
    calendar = pd.bdate_range("2020-01-02", periods=90)
    events = ["pre_open", "open", "intraday", "close", "after_close"]

    checked = 0
    while checked < 120:
        decision_event = str(rng.choice(events[:-1]))
        execution_offset = int(rng.integers(0, 3))
        execution_event = str(rng.choice(events))
        holding = int(rng.integers(1, 7))
        exit_event = str(rng.choice(events))
        publication_lag = int(rng.integers(0, 3))
        safety = int(rng.integers(0, 4))
        try:
            contract = LabelContract(
                decision_event=decision_event,
                execution_offset_sessions=execution_offset,
                execution_event=execution_event,
                holding_period_sessions=holding,
                exit_event=exit_event,
                label_publication_lag_sessions=publication_lag,
                safety_gap_sessions=safety,
            )
        except ValueError:
            continue

        intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
        tests = np.zeros(len(intervals), dtype=bool)
        start_a = int(rng.integers(15, 30))
        start_b = int(rng.integers(45, 60))
        tests[start_a : start_a + 2] = True
        tests[start_b : start_b + 3] = True
        candidate = ~tests
        embargo = int(rng.integers(0, 4))
        clean, _ = purged_train_test_masks(
            intervals,
            candidate_train_mask=candidate,
            test_mask=tests,
            contract=contract,
            embargo_sessions=embargo,
            calendar=calendar,
        )
        expected = _naive_purge(intervals, candidate, tests, calendar, embargo)
        np.testing.assert_array_equal(clean, expected)
        checked += 1


def test_tail_test_with_unrealized_label_is_rejected():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    contract = LabelContract(holding_period_sessions=10)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    tests = np.zeros(len(intervals), dtype=bool)
    tests[-2:] = True
    candidate = ~tests
    with pytest.raises(ValueError, match="complete interval"):
        purged_train_test_masks(
            intervals,
            candidate_train_mask=candidate,
            test_mask=tests,
            contract=contract,
            calendar=calendar,
        )


def test_maturity_scope_does_not_fail_on_unselected_tail_rows():
    calendar = pd.bdate_range("2024-01-02", periods=50)
    contract = LabelContract(holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    scope = np.zeros(len(intervals), dtype=bool)
    scope[10:20] = True
    result = audit_label_maturity(
        intervals,
        contract=contract,
        scope_mask=scope,
        as_of=calendar[30],
        calendar=calendar,
    )
    assert result.status is AuditStatus.PASS
    assert result.rows == 10
    assert result.immature_rows == 0


def test_empty_maturity_scope_is_inconclusive():
    calendar = pd.bdate_range("2024-01-02", periods=30)
    contract = LabelContract(holding_period_sessions=3)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    result = audit_label_maturity(
        intervals,
        contract=contract,
        scope_mask=np.zeros(len(intervals), dtype=bool),
        calendar=calendar,
    )
    assert result.status is AuditStatus.INCONCLUSIVE


def test_test_label_with_unavailable_publication_lag_is_rejected():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    contract = LabelContract(
        holding_period_sessions=3,
        label_publication_lag_sessions=5,
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    tests = np.zeros(len(intervals), dtype=bool)
    tests[-5] = True  # label end exists, publication endpoint does not
    candidate = ~tests
    with pytest.raises(ValueError, match="complete interval"):
        purged_train_test_masks(
            intervals,
            candidate_train_mask=candidate,
            test_mask=tests,
            contract=contract,
            calendar=calendar,
        )
