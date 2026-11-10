import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st

from timesafe_quant import LabelContract, build_label_intervals, purged_train_test_masks
from timesafe_quant.timing import EVENT_ORDER, EVENTS_PER_SESSION

EVENTS = tuple(EVENT_ORDER)


def _encode(calendar, session, event):
    if pd.isna(session):
        return -1
    position = calendar.get_indexer([session])[0]
    if position < 0:
        return -1
    return position * EVENTS_PER_SESSION + EVENT_ORDER[event]


@settings(max_examples=80, deadline=None)
@given(
    decision_event=st.sampled_from(EVENTS[:-1]),
    execution_offset=st.integers(min_value=0, max_value=2),
    execution_event=st.sampled_from(EVENTS),
    holding=st.integers(min_value=1, max_value=8),
    exit_event=st.sampled_from(EVENTS),
    publication_lag=st.integers(min_value=0, max_value=2),
    safety_gap=st.integers(min_value=0, max_value=4),
    embargo=st.integers(min_value=0, max_value=3),
    block_a=st.integers(min_value=12, max_value=25),
    block_b=st.integers(min_value=40, max_value=55),
)
def test_purged_rows_are_disjoint_from_every_test_information_block(
    decision_event,
    execution_offset,
    execution_event,
    holding,
    exit_event,
    publication_lag,
    safety_gap,
    embargo,
    block_a,
    block_b,
):
    try:
        contract = LabelContract(
            decision_event=decision_event,
            execution_offset_sessions=execution_offset,
            execution_event=execution_event,
            holding_period_sessions=holding,
            exit_event=exit_event,
            label_publication_lag_sessions=publication_lag,
            safety_gap_sessions=safety_gap,
        )
    except ValueError:
        assume(False)

    calendar = pd.bdate_range("2021-01-04", periods=90)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    tests = np.zeros(len(intervals), dtype=bool)
    tests[block_a : block_a + 2] = True
    tests[block_b : block_b + 3] = True
    candidate = ~tests

    clean, _ = purged_train_test_masks(
        intervals,
        candidate_train_mask=candidate,
        test_mask=tests,
        contract=contract,
        embargo_sessions=embargo,
        calendar=calendar,
    )

    test_ranges = []
    for row in intervals.loc[tests].itertuples():
        test_ranges.append(
            (
                _encode(calendar, row.decision_time, row.decision_event),
                _encode(calendar, row.label_end, row.label_end_event),
            )
        )

    for row in intervals.loc[clean].itertuples():
        start = _encode(calendar, row.decision_time, row.decision_event)
        end = max(
            _encode(calendar, row.label_end, row.label_end_event),
            _encode(
                calendar,
                row.label_available_with_gap,
                row.label_available_with_gap_event,
            ),
        )
        assert start >= 0 and end >= 0
        for test_start, test_end in test_ranges:
            assert not (start <= test_end and end >= test_start)
