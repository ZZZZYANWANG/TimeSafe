"""Deterministic independent oracle for purge and embargo masks.

This deliberately avoids TimeSafe's internal interval-merging helpers.  It guards the
optimized implementation even when the optional Hypothesis dependency is unavailable.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from timesafe_quant import LabelContract, build_label_intervals, purged_train_test_masks
from timesafe_quant.timing import EVENT_ORDER, EVENTS_PER_SESSION, SessionEvent, encode_points


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def test_purged_masks_match_independent_randomized_oracle():
    rng = random.Random(20261111)
    executed = 0

    for _ in range(400):
        size = rng.randint(45, 160)
        calendar = pd.Index(range(size))
        try:
            contract = LabelContract(
                decision_event=rng.choice(
                    ["pre_open", "open", "intraday", "close", "after_close"]
                ),
                execution_offset_sessions=rng.randint(0, 3),
                execution_event=rng.choice(
                    ["pre_open", "open", "intraday", "close", "after_close"]
                ),
                holding_period_sessions=rng.randint(1, 25),
                exit_event=rng.choice(["open", "intraday", "close", "after_close"]),
                label_publication_lag_sessions=rng.randint(0, 4),
                label_available_event="after_close",
                safety_gap_sessions=rng.randint(0, 5),
            )
        except ValueError:
            continue

        frame = build_label_intervals(calendar, calendar=calendar, contract=contract)
        decision = encode_points(
            frame["decision_time"], frame["decision_event"], calendar=calendar
        )
        label_end = encode_points(
            frame["label_end"], frame["label_end_event"], calendar=calendar
        )
        available = encode_points(
            frame["label_available"],
            frame["label_available_event"],
            calendar=calendar,
        )
        gap_end = encode_points(
            frame["label_available_with_gap"],
            frame["label_available_with_gap_event"],
            calendar=calendar,
        )
        valid_test = (decision >= 0) & (label_end >= 0) & (available >= 0)
        valid_indices = np.flatnonzero(valid_test)
        if len(valid_indices) < 10:
            continue

        test = np.zeros(size, dtype=bool)
        for _ in range(rng.randint(1, 3)):
            start = int(rng.choice(valid_indices))
            for position in range(start, min(size, start + rng.randint(1, 6))):
                if valid_test[position]:
                    test[position] = True
        candidate = ~test
        embargo = rng.randint(0, 6)

        observed, returned_test = purged_train_test_masks(
            frame,
            candidate_train_mask=candidate,
            test_mask=test,
            contract=contract,
            embargo_sessions=embargo,
            calendar=calendar,
        )
        np.testing.assert_array_equal(returned_test, test)

        valid_train = valid_test & (gap_end >= 0)
        expected = candidate & valid_train
        test_blocks = _merge_intervals(
            [(int(decision[i]), int(label_end[i])) for i in np.flatnonzero(test)]
        )

        for index in np.flatnonzero(expected):
            train_end = max(int(label_end[index]), int(gap_end[index]))
            if any(
                int(decision[index]) <= test_end and train_end >= test_start
                for test_start, test_end in test_blocks
            ):
                expected[index] = False

        if embargo:
            for _, test_end in test_blocks:
                end_session = test_end // EVENTS_PER_SESSION
                embargo_end = (
                    min(end_session + embargo, size - 1) * EVENTS_PER_SESSION
                    + EVENT_ORDER[SessionEvent.AFTER_CLOSE.value]
                )
                expected &= ~(
                    (decision > test_end) & (decision <= embargo_end)
                )

        np.testing.assert_array_equal(observed, expected)
        executed += 1

    assert executed >= 250
