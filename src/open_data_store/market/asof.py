"""As-of helpers for market recipes: pure functions over Arrow tables (market-data spec §7.4, §9.3).

The store ``Recipe`` reads all inputs at one date and bounds only ``known_at``; market recipes
therefore apply their own ``valid_date <= t`` bound (``asof``) and, for identifier mappings, take
the latest observation per key (``latest``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def _check_nulls(table: pa.Table, keys: Sequence[str]) -> None:
    for name in (*keys, "known_at"):
        if table[name].null_count:
            raise ValueError(f"column {name!r} contains null values")


def _latest_per_key(
    table: pa.Table, natural_key: Sequence[str], candidates: np.ndarray | None = None
) -> pa.Table:
    """Latest row per key among ``candidates`` (row indices of ``table``; all rows if None)."""
    keys = list(natural_key)
    _check_nulls(table, keys)
    # Sort only the key columns and known_at of the candidates (payload columns never move),
    # then keep the last row of each key run; the payload is taken once, at the end.
    slim = table.select([*keys, "known_at"])
    if candidates is not None:
        slim = slim.take(pa.array(candidates, pa.int64()))
    if slim.num_rows == 0:
        return table.slice(0, 0)
    order = pc.sort_indices(slim, sort_keys=[(k, "ascending") for k in (*keys, "known_at")])
    ordered = slim.take(order)
    n = ordered.num_rows
    same_key = pa.array(np.ones(n - 1, bool))
    for name in keys:
        col = ordered[name].combine_chunks()
        same_key = pc.and_(same_key, pc.equal(col.slice(0, n - 1), col.slice(1)))
    known = ordered["known_at"].combine_chunks()
    # A tie on (key, known_at) is malformed input: the publisher refuses duplicates.
    # Only a tie at the latest known_at of a key matters (the pair ends its key run).
    same = same_key.to_numpy(zero_copy_only=False)
    is_last = np.append(~same, True)
    tied = same & pc.equal(known.slice(0, n - 1), known.slice(1)).to_numpy(zero_copy_only=False)
    tied &= is_last[1:]
    if tied.any():
        first = int(np.flatnonzero(tied)[0])
        example = {k: ordered[k][first].as_py() for k in keys}
        raise ValueError(f"several rows share the latest known_at for key {example}")
    rows = np.sort(order.to_numpy()[is_last])
    if candidates is not None:
        rows = candidates[rows]
    return table.take(pa.array(rows, pa.int64()))


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
    visible = np.flatnonzero(pc.fill_null(mask, False).to_numpy(zero_copy_only=False))
    return _latest_per_key(table, natural_key, visible)


def sessions(mic: str, start: date, end: date) -> pa.Table:
    """Trading sessions of exchange ``mic`` in ``[start, end]`` inclusive, with the close in UTC.

    The calendar is built on the requested range, not the library's default window, so any
    range the calendar library supports works, including history before its default start
    (e.g. XNYS before 2006). Reliable range: XNYS yearly session counts were checked from 1990
    (exchange_calendars 4.13.2); earlier years build but are unverified. A range with no session
    returns an empty table; ``start > end`` or a range outside the library's bounds raises
    ``ValueError``."""
    import exchange_calendars as xcals
    from exchange_calendars import errors as xerrors
    from pandas.errors import OutOfBoundsDatetime

    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    try:
        with warnings.catch_warnings():
            # Upstream: exchange_calendars 4.13.2 calls Series.values on tz-aware data while
            # building a calendar, which pandas 3.1 flags with Pandas4Warning.
            warnings.filterwarnings(
                "ignore", message="Series.values returning an ndarray", category=Warning
            )
            # The library needs start < end: build one day past end, filter back below.
            cal = xcals.get_calendar(
                mic, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat()
            )
    except xerrors.NoSessionsError:
        days = []
    except (OutOfBoundsDatetime, xerrors.DateOutOfBounds) as exc:
        raise ValueError(
            f"sessions of {mic!r} over {start.isoformat()}..{end.isoformat()} are outside the "
            f"range the exchange calendar library supports: {exc}"
        ) from exc
    else:
        days = [d for d in cal.sessions if d.date() <= end]
    closes = [cal.session_close(d).to_pydatetime().astimezone(UTC) for d in days]
    return pa.table(
        {
            "session_date": pa.array([d.date() for d in days], pa.date32()),
            "close_utc": pa.array(closes, pa.timestamp("us", tz="UTC")),
        }
    )
