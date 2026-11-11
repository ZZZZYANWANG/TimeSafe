"""Event-aware temporal point utilities.

TimeSafe treats a temporal point as ``(session, within_session_event)``.  The
supported event order is intentionally small and explicit so that contracts such
as "decide at close, execute at the same day's open" are rejected rather than
silently coerced.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class SessionEvent(str, Enum):
    """Supported within-session event order."""

    PRE_OPEN = "pre_open"
    OPEN = "open"
    INTRADAY = "intraday"
    CLOSE = "close"
    AFTER_CLOSE = "after_close"


EVENT_ORDER: dict[str, int] = {
    SessionEvent.PRE_OPEN.value: 0,
    SessionEvent.OPEN.value: 1,
    SessionEvent.INTRADAY.value: 2,
    SessionEvent.CLOSE.value: 3,
    SessionEvent.AFTER_CLOSE.value: 4,
}
EVENTS_PER_SESSION = len(EVENT_ORDER)


def normalize_event(value: str | SessionEvent) -> str:
    """Return a validated event name."""

    event = value.value if isinstance(value, SessionEvent) else str(value)
    if event not in EVENT_ORDER:
        raise ValueError(
            f"Unsupported session event {event!r}; choose one of {sorted(EVENT_ORDER)}."
        )
    return event


def point_key(session_offset: int, event: str | SessionEvent) -> tuple[int, int]:
    """Comparable key for a relative temporal point."""

    normalized = normalize_event(event)
    return int(session_offset), EVENT_ORDER[normalized]


def point_lt_scalar(
    left_session: Any,
    left_event: str | SessionEvent,
    right_session: Any,
    right_event: str | SessionEvent,
) -> bool:
    """Strict event-aware comparison for scalar temporal points."""

    left_name = normalize_event(left_event)
    right_name = normalize_event(right_event)
    return bool(
        left_session < right_session
        or (
            left_session == right_session
            and EVENT_ORDER[left_name] < EVENT_ORDER[right_name]
        )
    )


def point_le_scalar(
    left_session: Any,
    left_event: str | SessionEvent,
    right_session: Any,
    right_event: str | SessionEvent,
) -> bool:
    """Non-strict event-aware comparison for scalar temporal points."""

    left_name = normalize_event(left_event)
    right_name = normalize_event(right_event)
    return bool(
        left_session < right_session
        or (
            left_session == right_session
            and EVENT_ORDER[left_name] <= EVENT_ORDER[right_name]
        )
    )


def series_point_lt(
    left_session: pd.Series,
    left_event: pd.Series,
    right_session: Any,
    right_event: str | SessionEvent,
) -> pd.Series:
    """Vectorized strict comparison against one scalar temporal point."""

    right_name = normalize_event(right_event)
    event_rank = left_event.map(lambda value: EVENT_ORDER[normalize_event(value)])
    return (left_session < right_session) | (
        (left_session == right_session) & (event_rank < EVENT_ORDER[right_name])
    )


def series_point_le(
    left_session: pd.Series,
    left_event: pd.Series,
    right_session: Any,
    right_event: str | SessionEvent,
) -> pd.Series:
    """Vectorized non-strict comparison against one scalar temporal point."""

    right_name = normalize_event(right_event)
    event_rank = left_event.map(lambda value: EVENT_ORDER[normalize_event(value)])
    return (left_session < right_session) | (
        (left_session == right_session) & (event_rank <= EVENT_ORDER[right_name])
    )


def canonical_calendar(calendar: pd.Index | list[Any]) -> pd.Index:
    """Return a unique, sorted trading calendar."""

    raw = pd.Index(calendar)
    if len(raw) == 0:
        raise ValueError("calendar must not be empty.")
    if bool(pd.isna(raw).any()):
        raise ValueError("calendar must not contain missing sessions.")
    if not raw.is_unique:
        raise ValueError("calendar must contain unique sessions.")
    return raw.sort_values()


def shift_sessions(
    sessions: pd.Index,
    positions: np.ndarray,
    offset: int,
) -> pd.Index:
    """Shift calendar positions, returning missing values beyond calendar bounds."""

    source = np.asarray(positions, dtype=np.int64)
    target = source + int(offset)
    missing_value: Any = pd.NaT if isinstance(sessions, pd.DatetimeIndex) else None
    source_valid = (source >= 0) & (source < len(sessions))
    target_valid = source_valid & (target >= 0) & (target < len(sessions))
    values = [
        sessions[int(position)] if valid else missing_value
        for position, valid in zip(target, target_valid)
    ]
    if isinstance(sessions, pd.DatetimeIndex):
        return pd.DatetimeIndex(values)
    return pd.Index(values)


def encode_points(
    sessions: pd.Series | pd.Index | list[Any],
    events: pd.Series | pd.Index | list[str] | str | SessionEvent,
    *,
    calendar: pd.Index | list[Any],
) -> np.ndarray:
    """Encode temporal points as sortable integers.

    Missing or out-of-calendar sessions are encoded as ``-1``.  Valid points use
    ``session_position * EVENTS_PER_SESSION + event_rank``.
    """

    cal = canonical_calendar(calendar)
    session_index = pd.Index(sessions)
    positions = cal.get_indexer(session_index)

    if isinstance(events, (str, SessionEvent)):
        event_names = np.repeat(normalize_event(events), len(session_index))
    else:
        event_values = list(events)
        if len(event_values) != len(session_index):
            raise ValueError("events must match sessions length.")
        event_names = np.asarray([normalize_event(value) for value in event_values])

    encoded = np.full(len(session_index), -1, dtype=np.int64)
    valid = positions >= 0
    if valid.any():
        ranks = np.asarray([EVENT_ORDER[name] for name in event_names], dtype=np.int64)
        encoded[valid] = positions[valid] * EVENTS_PER_SESSION + ranks[valid]
    return encoded


def decode_point(encoded: int, *, calendar: pd.Index | list[Any]) -> tuple[Any, str]:
    """Decode an integer point produced by :func:`encode_points`."""

    if encoded < 0:
        raise ValueError("encoded point must be non-negative.")
    cal = canonical_calendar(calendar)
    session_position, event_position = divmod(int(encoded), EVENTS_PER_SESSION)
    if session_position >= len(cal):
        raise ValueError("encoded point is outside calendar bounds.")
    inverse = {rank: name for name, rank in EVENT_ORDER.items()}
    return cal[session_position], inverse[event_position]
