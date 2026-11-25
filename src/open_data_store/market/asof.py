"""As-of helpers for market recipes: pure functions over Arrow tables (market-data spec §7.4, §9.3).

The store ``Recipe`` reads all inputs at one date and bounds only ``known_at``; market recipes
therefore apply their own ``valid_date <= t`` bound (``asof``) and, for identifier mappings, take
the latest observation per key (``latest``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

import pyarrow as pa
import pyarrow.compute as pc


def _latest_per_key(table: pa.Table, natural_key: Sequence[str]) -> pa.Table:
    if table.num_rows == 0:
        return table
    keys = list(natural_key)
    top = table.group_by(keys, use_threads=False).aggregate([("known_at", "max")])
    top = top.rename_columns(keys + ["_max_known_at"])
    joined = table.append_column("_row", pa.array(range(table.num_rows), pa.int64())).join(
        top, keys=keys, join_type="inner"
    )
    joined = joined.filter(pc.equal(joined["known_at"], joined["_max_known_at"]))
    # A tie on (key, known_at) is malformed input: the publisher refuses duplicates.
    counts = joined.group_by(keys, use_threads=False).aggregate([("_row", "count")])
    if pc.max(counts["_row_count"]).as_py() > 1:
        raise ValueError(f"several rows share the latest known_at for one key {keys}")
    rows = pc.sort_indices(joined["_row"])
    return joined.take(rows).select(table.column_names)


def latest(table: pa.Table, natural_key: Sequence[str]) -> pa.Table:
    """Latest ``known_at`` row per natural key, with no time bound."""
    return _latest_per_key(table, natural_key)


def asof(table: pa.Table, natural_key: Sequence[str], as_of: datetime) -> pa.Table:
    """Rows visible at ``as_of``: ``known_at <= as_of`` and ``valid_date <= as_of`` (UTC date),
    then the latest ``known_at`` per natural key. ``as_of`` must be timezone-aware."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")
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

    cal = xcals.get_calendar(mic)
    days = cal.sessions_in_range(start.isoformat(), end.isoformat())
    closes = [cal.session_close(d).to_pydatetime().astimezone(UTC) for d in days]
    return pa.table(
        {
            "session_date": pa.array([d.date() for d in days], pa.date32()),
            "close_utc": pa.array(closes, pa.timestamp("us", tz="UTC")),
        }
    )
