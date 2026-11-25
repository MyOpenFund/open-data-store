"""As-of helpers for market recipes: pure functions over Arrow tables (market-data spec §7.4, §9.3).

The store ``Recipe`` reads all inputs at one date and bounds only ``known_at``; market recipes
therefore apply their own ``valid_date <= t`` bound (``asof``) and, for identifier mappings, take
the latest observation per key (``latest``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import UTC, date, datetime

import pyarrow as pa
import pyarrow.compute as pc

_ROW = "__ods_row__"
_MAX = "__ods_max_known_at__"


def _check_nulls(table: pa.Table, keys: Sequence[str]) -> None:
    for name in (*keys, "known_at"):
        if table[name].null_count:
            raise ValueError(f"column {name!r} contains null values")


def _latest_per_key(table: pa.Table, natural_key: Sequence[str]) -> pa.Table:
    keys = list(natural_key)
    _check_nulls(table, keys)
    for reserved in (_ROW, _MAX):
        if reserved in table.column_names:
            raise ValueError(f"column name {reserved!r} is reserved")
    if table.num_rows == 0:
        return table
    # Join only the key columns, known_at and a row index: payload columns (lists, structs)
    # never enter the join.
    slim = table.select([*keys, "known_at"]).append_column(
        _ROW, pa.array(range(table.num_rows), pa.int64())
    )
    top = slim.group_by(keys, use_threads=False).aggregate([("known_at", "max")])
    top = top.rename_columns([*keys, _MAX])
    joined = slim.join(top, keys=keys, join_type="inner")
    joined = joined.filter(pc.equal(joined["known_at"], joined[_MAX]))
    # A tie on (key, known_at) is malformed input: the publisher refuses duplicates.
    counts = joined.group_by(keys, use_threads=False).aggregate([(_ROW, "count")])
    tied = counts.filter(pc.greater(counts[f"{_ROW}_count"], 1))
    if tied.num_rows:
        example = {k: tied[k][0].as_py() for k in keys}
        raise ValueError(f"several rows share the latest known_at for key {example}")
    rows = pc.sort_indices(joined[_ROW])
    return table.take(joined[_ROW].take(rows))


def latest(table: pa.Table, natural_key: Sequence[str]) -> pa.Table:
    """Latest ``known_at`` row per natural key, with no time bound.

    Selection is by latest ``known_at`` only: for dated facts the natural key must include
    ``valid_date``. Nulls in a key column or ``known_at``, and ties, raise ``ValueError``."""
    return _latest_per_key(table, natural_key)


def asof(table: pa.Table, natural_key: Sequence[str], as_of: datetime) -> pa.Table:
    """Rows visible at ``as_of``: ``known_at <= as_of`` and ``valid_date <= as_of`` (UTC date),
    then the latest ``known_at`` per natural key. ``as_of`` must be timezone-aware.

    Selection is by latest ``known_at`` only: for dated facts the natural key must include
    ``valid_date``. Nulls in a key column or ``known_at``, and ties, raise ``ValueError``."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    _check_nulls(table, list(natural_key))
    as_of_utc = as_of.astimezone(UTC)
    ts_type = table.schema.field("known_at").type
    bound = pa.scalar(as_of_utc, type=ts_type)
    mask = pc.and_(
        pc.less_equal(table["known_at"], bound),
        pc.less_equal(table["valid_date"], pa.scalar(as_of_utc.date(), type=pa.date32())),
    )
    return _latest_per_key(table.filter(mask), natural_key)


def sessions(mic: str, start: date, end: date) -> pa.Table:
    """Trading sessions of exchange ``mic`` in ``[start, end]`` inclusive, with the close in UTC."""
    import exchange_calendars as xcals

    with warnings.catch_warnings():
        # Upstream: exchange_calendars 4.13.2 calls Series.values on tz-aware data while building
        # a calendar, which pandas 3.1 flags with Pandas4Warning.
        warnings.filterwarnings(
            "ignore", message="Series.values returning an ndarray", category=Warning
        )
        cal = xcals.get_calendar(mic)
    days = cal.sessions_in_range(start.isoformat(), end.isoformat())
    closes = [cal.session_close(d).to_pydatetime().astimezone(UTC) for d in days]
    return pa.table(
        {
            "session_date": pa.array([d.date() for d in days], pa.date32()),
            "close_utc": pa.array(closes, pa.timestamp("us", tz="UTC")),
        }
    )
