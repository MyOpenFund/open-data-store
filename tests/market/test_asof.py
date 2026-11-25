from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from open_data_store.market.asof import asof, latest, sessions


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


def test_null_key_or_known_at_rejected():
    t = _table([("a", date(2024, 5, 1), datetime(2024, 5, 1, tzinfo=UTC), 1)])
    bad_key = t.set_column(0, "k", pa.array([None], pa.string()))
    bad_ka = t.set_column(2, "known_at", pa.array([None], pa.timestamp("us", tz="UTC")))
    big = datetime(2024, 6, 1, tzinfo=UTC)
    for fn in (lambda x: latest(x, ["k"]), lambda x: asof(x, ["k"], big)):
        with pytest.raises(ValueError, match="'k'"):
            fn(bad_key)
        with pytest.raises(ValueError, match="'known_at'"):
            fn(bad_ka)


def test_list_payload_column_supported():
    t = _table([("a", date(2024, 5, 1), datetime(2024, 5, 1, tzinfo=UTC), 1)])
    t = t.append_column("p", pa.array([[1, 2]], pa.list_(pa.int64())))
    out = latest(t, ["k"])
    assert out.column("p").to_pylist() == [[1, 2]]


def test_reserved_column_name_rejected():
    t = _table([("a", date(2024, 5, 1), datetime(2024, 5, 1, tzinfo=UTC), 1)])
    t = t.append_column("__ods_row__", pa.array([0], pa.int64()))
    with pytest.raises(ValueError, match="reserved"):
        latest(t, ["k"])


def test_tie_error_names_key():
    ts = datetime(2024, 5, 1, tzinfo=UTC)
    t = _table([("zz", date(2024, 5, 1), ts, 1), ("zz", date(2024, 5, 1), ts, 2)])
    with pytest.raises(ValueError, match="zz"):
        latest(t, ["k"])


def test_non_utc_as_of_uses_utc_date():
    t = _table([("a", date(2024, 5, 2), datetime(2024, 5, 1, tzinfo=UTC), 1)])
    # 2024-05-01 22:00 at -05:00 is 2024-05-02 03:00 UTC: valid_date 05-02 is admitted.
    as_of = datetime(2024, 5, 1, 22, tzinfo=timezone(timedelta(hours=-5)))
    assert asof(t, ["k"], as_of).num_rows == 1
    as_of2 = datetime(2024, 5, 1, 18, tzinfo=timezone(timedelta(hours=-5)))
    assert asof(t, ["k"], as_of2).num_rows == 0


def test_sessions_range_crossing_holiday_from_non_session_day():
    # 2024-11-27 Wed .. 2024-12-01 Sun: Thanksgiving 11-28 closed, 11-30 and 12-01 weekend.
    s = sessions("XNYS", date(2024, 11, 27), date(2024, 12, 1))
    assert s.column("session_date").to_pylist() == [date(2024, 11, 27), date(2024, 11, 29)]
    # Starts on a Saturday (non-session day).
    s = sessions("XNYS", date(2024, 11, 30), date(2024, 12, 3))
    assert s.column("session_date").to_pylist() == [date(2024, 12, 2), date(2024, 12, 3)]
