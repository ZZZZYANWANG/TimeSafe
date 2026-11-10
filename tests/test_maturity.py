import numpy as np
import pandas as pd
import pytest

from timesafe_quant import (
    AuditStatus,
    LabelContract,
    audit_label_maturity,
    build_label_intervals,
    purged_train_test_masks,
    purged_walk_forward_masks,
)


def test_contract_offsets_and_events():
    contract = LabelContract(
        execution_offset_sessions=1,
        holding_period_sessions=20,
        label_publication_lag_sessions=2,
        safety_gap_sessions=3,
    )
    assert contract.label_start_offset_sessions == 1
    assert contract.label_end_offset_sessions == 20
    assert contract.label_available_offset_sessions == 22
    assert contract.recommended_purge_sessions == 25
    assert contract.decision_event == "close"
    assert contract.execution_event == "open"


def test_impossible_same_day_close_to_open_contract_rejected():
    with pytest.raises(ValueError, match="Execution must occur strictly after"):
        LabelContract(
            execution_offset_sessions=0,
            decision_event="close",
            execution_event="open",
            holding_period_sessions=2,
        )


def test_valid_same_day_open_to_close_contract():
    contract = LabelContract(
        execution_offset_sessions=0,
        decision_event="pre_open",
        execution_event="open",
        holding_period_sessions=1,
        exit_event="close",
    )
    assert contract.label_end_offset_sessions == 0


def test_build_label_intervals_uses_trading_calendar_and_gap():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    contract = LabelContract(
        execution_offset_sessions=1,
        holding_period_sessions=5,
        safety_gap_sessions=2,
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    assert intervals.loc[0, "label_start"] == calendar[1]
    assert intervals.loc[0, "label_end"] == calendar[5]
    assert intervals.loc[0, "label_available_with_gap"] == calendar[7]
    assert not bool(intervals.loc[len(calendar) - 1, "interval_valid"])


def test_maturity_audit_is_event_aware():
    calendar = pd.bdate_range("2024-01-02", periods=20)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=2)
    intervals = build_label_intervals(calendar[:5], calendar=calendar, contract=contract)
    available = intervals.loc[0, "label_available"]
    before = audit_label_maturity(
        intervals.iloc[[0]],
        contract=contract,
        as_of=available,
        as_of_event="close",
        calendar=calendar,
    )
    after = audit_label_maturity(
        intervals.iloc[[0]],
        contract=contract,
        as_of=available,
        as_of_event="after_close",
        calendar=calendar,
    )
    assert before.status is AuditStatus.FAIL
    assert after.status is AuditStatus.PASS


def test_maturity_audit_flags_unavailable_labels():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=10)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    result = audit_label_maturity(
        intervals,
        contract=contract,
        as_of=calendar[20],
        calendar=calendar,
    )
    assert result.status is AuditStatus.FAIL
    assert result.immature_rows > 0


def test_purged_walk_forward_applies_safety_gap():
    calendar = pd.bdate_range("2024-01-02", periods=90)
    contract = LabelContract(
        execution_offset_sessions=1,
        holding_period_sessions=20,
        safety_gap_sessions=4,
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    test_start = calendar[60]
    train, test = purged_walk_forward_masks(
        intervals,
        test_start=test_start,
        test_end=calendar[69],
        contract=contract,
        calendar=calendar,
    )
    kept = intervals.loc[train]
    assert np.all(kept["label_available_with_gap"] < test_start)
    last_train_pos = np.flatnonzero(train)[-1]
    assert 60 - last_train_pos - 1 == contract.recommended_purge_sessions
    result = audit_label_maturity(
        intervals,
        contract=contract,
        train_mask=train,
        test_mask=test,
        calendar=calendar,
    )
    assert result.split_overlap_rows == 0


def test_walk_forward_rejects_silent_embargo_argument():
    calendar = pd.bdate_range("2024-01-02", periods=60)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    with pytest.raises(ValueError, match="not applicable"):
        purged_walk_forward_masks(
            intervals,
            test_start=calendar[30],
            test_end=calendar[35],
            contract=contract,
            calendar=calendar,
            embargo_sessions=3,
        )


def test_general_purged_split_applies_embargo():
    calendar = pd.bdate_range("2024-01-02", periods=100)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    test = (
        (intervals["decision_time"] >= calendar[40])
        & (intervals["decision_time"] <= calendar[44])
    ).to_numpy()
    candidate = ~test
    clean, _ = purged_train_test_masks(
        intervals,
        candidate_train_mask=candidate,
        test_mask=test,
        contract=contract,
        embargo_sessions=3,
        calendar=calendar,
    )
    latest_test_label_end = intervals.loc[test, "label_end"].max()
    embargo_end = calendar[calendar.get_loc(latest_test_label_end) + 3]
    kept_decisions = intervals.loc[clean, "decision_time"]
    assert not (
        (kept_decisions > latest_test_label_end)
        & (kept_decisions <= embargo_end)
    ).any()


def test_insufficient_purge_is_detected():
    calendar = pd.bdate_range("2024-01-02", periods=80)
    contract = LabelContract(
        execution_offset_sessions=1,
        holding_period_sessions=20,
        safety_gap_sessions=4,
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    train = (intervals["decision_time"] < calendar[50]).to_numpy()
    test = (
        (intervals["decision_time"] >= calendar[50])
        & (intervals["decision_time"] <= calendar[59])
    ).to_numpy()
    result = audit_label_maturity(
        intervals,
        contract=contract,
        train_mask=train,
        test_mask=test,
        calendar=calendar,
    )
    assert result.status is AuditStatus.FAIL
    assert result.split_overlap_rows > 0
    assert result.metadata["safety_gap_rows"] > 0
