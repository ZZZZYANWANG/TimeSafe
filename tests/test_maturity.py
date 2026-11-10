import numpy as np
import pandas as pd

from timesafe_quant import (
    AuditStatus,
    LabelContract,
    audit_label_maturity,
    build_label_intervals,
    purged_walk_forward_masks,
)


def test_contract_offsets():
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


def test_build_label_intervals_uses_trading_calendar():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    assert intervals.loc[0, "label_start"] == calendar[1]
    assert intervals.loc[0, "label_end"] == calendar[5]
    assert not bool(intervals.loc[len(calendar) - 1, "interval_valid"])


def test_maturity_audit_flags_unavailable_labels():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=10)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    result = audit_label_maturity(intervals, contract=contract, as_of=calendar[20])
    assert result.status is AuditStatus.FAIL
    assert result.immature_rows > 0


def test_purged_walk_forward_removes_overlapping_train_rows():
    calendar = pd.bdate_range("2024-01-02", periods=80)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=20)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    train, test = purged_walk_forward_masks(
        intervals,
        test_start=calendar[50],
        test_end=calendar[59],
    )
    assert np.all(intervals.loc[train, "label_available"] < calendar[50])
    assert np.all(
        (intervals.loc[test, "decision_time"] >= calendar[50])
        & (intervals.loc[test, "decision_time"] <= calendar[59])
    )
    result = audit_label_maturity(
        intervals,
        contract=contract,
        train_mask=train,
        test_mask=test,
        calendar=calendar,
    )
    assert result.split_overlap_rows == 0


def test_insufficient_purge_is_detected():
    calendar = pd.bdate_range("2024-01-02", periods=80)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=20)
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
