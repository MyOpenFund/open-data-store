from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from open_data_store.market.asof import asof, latest, sessions

UTC = UTC


def _table(rows):
    return pa.table(
        {
            "k": pa.array([r[0] for r in rows], pa.string()),
            "valid_date": pa.array([r[1] for r in rows], pa.date32()),
            "known_at": pa.array([r[2] for r in rows], pa.timestamp("us", tz="UTC")),
            "v": pa.array([r[3] for r in rows], pa.int64()),
        }
    )


def test_future_valid_date_excluded():
    t = _table(
        [
            ("a", date(2024, 5, 1), datetime(2024, 5, 1, 12, tzinfo=UTC), 1),
            ("a", date(2024, 5, 2), datetime(2024, 5, 1, 13, tzinfo=UTC), 2),
        ]
    )
    out = asof(t, ["k"], datetime(2024, 5, 1, 23, tzinfo=UTC))
    assert out.column("v").to_pylist() == [1]


def test_correction_visible_only_after_known_at():
    t = _table(
        [
            ("a", date(2024, 5, 1), datetime(2024, 5, 1, 12, tzinfo=UTC), 1),
            ("a", date(2024, 5, 1), datetime(2024, 5, 3, 12, tzinfo=UTC), 9),
        ]
    )
    assert asof(t, ["k"], datetime(2024, 5, 2, tzinfo=UTC)).column("v").to_pylist() == [1]
    assert asof(t, ["k"], datetime(2024, 5, 3, 12, tzinfo=UTC)).column("v").to_pylist() == [9]
    assert asof(t, ["k"], datetime(2024, 5, 1, 11, tzinfo=UTC)).num_rows == 0


def test_latest_ignores_bounds_and_is_per_key():
    t = _table(
        [
            ("a", date(2024, 5, 1), datetime(2024, 5, 1, tzinfo=UTC), 1),
            ("a", date(2024, 5, 1), datetime(2024, 5, 3, tzinfo=UTC), 9),
            ("b", date(2030, 1, 1), datetime(2024, 5, 2, tzinfo=UTC), 5),
        ]
    )
    out = latest(t, ["k"])
    got = dict(zip(out.column("k").to_pylist(), out.column("v").to_pylist(), strict=True))
    assert got == {"a": 9, "b": 5}


def test_naive_as_of_rejected():
    t = _table([("a", date(2024, 5, 1), datetime(2024, 5, 1, tzinfo=UTC), 1)])
    with pytest.raises(ValueError):
        asof(t, ["k"], datetime(2024, 5, 2))


def test_known_at_tie_rejected():
    ts = datetime(2024, 5, 1, tzinfo=UTC)
    t = _table([("a", date(2024, 5, 1), ts, 1), ("a", date(2024, 5, 1), ts, 2)])
    with pytest.raises(ValueError):
        latest(t, ["k"])
    with pytest.raises(ValueError):
        asof(t, ["k"], datetime(2024, 6, 1, tzinfo=UTC))


def test_early_close_utc():
    s = sessions("XNYS", date(2024, 11, 29), date(2024, 11, 29))
    assert s.schema.field("close_utc").type == pa.timestamp("us", tz="UTC")
    assert s.schema.field("session_date").type == pa.date32()
    assert s.column("close_utc").to_pylist() == [datetime(2024, 11, 29, 18, tzinfo=UTC)]


def test_dst_close_utc():
    s = sessions("XNYS", date(2025, 3, 7), date(2025, 3, 10))
    assert s.column("session_date").to_pylist() == [date(2025, 3, 7), date(2025, 3, 10)]
    assert s.column("close_utc").to_pylist() == [
        datetime(2025, 3, 7, 21, tzinfo=UTC),
        datetime(2025, 3, 10, 20, tzinfo=UTC),
    ]
