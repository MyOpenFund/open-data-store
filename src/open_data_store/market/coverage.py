"""Coverage of an index universe and delisting exits (spec §7.4, G4, D8).

An explained gap is still bias: ``coverage`` gives every expected (member, session) cell a reason,
and measures the missing share per session against ``max_missing_share``.

Inputs (pure Arrow tables):

- ``intervals``: a ``membership_intervals`` output, half-open ``[valid_from, valid_to)``,
  ``valid_to`` null = open. Every row is used: filter one ``index_code`` beforehand. Overlapping
  intervals of one instrument count once.
- ``prices_resolved``: a ``resolve_prices`` output; two rows for one (instrument, session) raise
  ``ValueError``.
- ``sessions``: a table with a ``session_date`` column (e.g. ``asof.sessions``).
- ``fetch_coverage``: optional ``FETCH_COVERAGE_SCHEMA`` rows, fetch-log ranges already mapped to
  instruments by the caller: ``[covered_from, covered_to]`` inclusive, ``status`` from
  ``FETCH_STATUSES`` (an unknown status raises ``ValueError``; a null bound covers nothing).

Expected cells: every (instrument, session) where the session lies in one of the instrument's
intervals (and so is <= the last session of ``sessions``). Reason per cell, first match wins:

1. ``ok``: a resolved price with a finite close > 0;
2. ``price_invalid``: a resolved price whose close is null, non-finite or <= 0 (a trailing
   invalid close never disappears);
3. ``session_not_published``: the session is after the latest ``session_date`` in
   ``prices_resolved`` overall (no price at all: never matches);
4. ``delisted``: the instrument has price rows (valid or not) and the session is after its last
   price date;
5. ``source_has_no_data``: a ``fetch_coverage`` row of the instrument covers the session with
   status ``ok``, ``empty`` or ``not_found``;
6. ``not_fetched``: anything else.

``session_not_published`` is checked before ``delisted`` (controller ruling 3 lists them the other
way): otherwise, when prices stop for every instrument before the last session (a feed one day
behind), every live member would read as delisted, be excused from the missing count, and be
exited by ``delisting_exits``. ``unmapped`` and ``recycled_at_source`` stay in ``REASONS`` for
producers that see source symbols; this function never emits them (an unmapped symbol never
reaches an instrument cell: see ``membership.unresolved_members``).

Per session: ``members`` = expected cells, ``delisted`` = cells ``delisted``, ``missing`` = cells
neither ``ok`` nor ``delisted``, ``missing_share`` = missing / members (0.0 with no member),
``over_threshold`` = missing_share > ``max_missing_share``. ``passed`` = no session over it.
``max_missing_share`` defaults to ``DEFAULT_MAX_MISSING_SHARE`` (0.02), a placeholder to re-set
from measurements on real data (spec §11, review B5).

``delisting_exits``: an instrument is delisted when it has price rows and its membership holds a
session after its last price date and on or before the horizon (the earlier of the last session
and the latest price date overall), i.e. exactly when ``coverage`` gives it a ``delisted`` cell.
Rules:

- ``last_close`` (default, D8): exit at the last known (finite, > 0) close, ``exit_date`` its
  date. With no valid close at all, ``exit_price`` is null and ``exit_date`` the last price date.
- ``fixed_return``: the ``last_close`` exit with ``exit_price = close * (1 + fixed_return)``;
  ``fixed_return`` must be finite and >= -1.
- ``post_delisting_prices``: the latest valid close of ``post_prices`` (same schema as
  ``prices_resolved``) dated on or after the ``last_close`` exit date, ``exit_date`` its date;
  instruments with none fall back to ``last_close`` (``rule`` then reads ``last_close``).

Every exit row is ``flagged``; rows are sorted by ``instrument_id``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from open_data_store.contracts.standard.market import FETCH_STATUSES
from open_data_store.market._mapping import MAX_LISTED

__all__ = [
    "COVERAGE_PER_MEMBER_SCHEMA",
    "COVERAGE_PER_SESSION_SCHEMA",
    "DEFAULT_MAX_MISSING_SHARE",
    "DELISTING_EXITS_SCHEMA",
    "EXIT_RULES",
    "FETCH_COVERAGE_SCHEMA",
    "NO_DATA_STATUSES",
    "REASONS",
    "CoverageResult",
    "coverage",
    "delisting_exits",
]

REASONS = (
    "ok",
    "price_invalid",
    "not_fetched",
    "unmapped",
    "source_has_no_data",
    "recycled_at_source",
    "delisted",
    "session_not_published",
)
EXIT_RULES = ("last_close", "post_delisting_prices", "fixed_return")
NO_DATA_STATUSES = ("ok", "empty", "not_found")
DEFAULT_MAX_MISSING_SHARE = 0.02

FETCH_COVERAGE_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("covered_from", pa.date32()),
        ("covered_to", pa.date32()),
        ("status", pa.string()),
    ]
)
COVERAGE_PER_MEMBER_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("session_date", pa.date32()),
        ("reason", pa.string()),
    ]
)
COVERAGE_PER_SESSION_SCHEMA = pa.schema(
    [
        ("session_date", pa.date32()),
        ("members", pa.int64()),
        ("delisted", pa.int64()),
        ("missing", pa.int64()),
        ("missing_share", pa.float64()),
        ("over_threshold", pa.bool_()),
    ]
)
DELISTING_EXITS_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("exit_date", pa.date32()),
        ("exit_price", pa.float64()),
        ("rule", pa.string()),
        ("flagged", pa.bool_()),
    ]
)

_EPOCH = date(1970, 1, 1)
_CODE = {name: np.int8(i) for i, name in enumerate(REASONS)}


@dataclass(frozen=True)
class CoverageResult:
    per_member: pa.Table
    per_session: pa.Table
    passed: bool


def _days(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """date32 → int64 day numbers; nulls → INT64 min (callers mask them)."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    out = arr.cast(pa.int32()).fill_null(np.iinfo(np.int32).min)
    return out.to_numpy(zero_copy_only=False).astype(np.int64)


def _codes(column: pa.ChunkedArray, names: pa.Array) -> np.ndarray:
    """Positions of ``column`` values in ``names`` (-1 when absent)."""
    idx = pc.index_in(column, value_set=names)
    idx = idx.combine_chunks() if isinstance(idx, pa.ChunkedArray) else idx
    return idx.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)


@dataclass(frozen=True)
class _Universe:
    names: pa.Array  # sorted unique instrument ids of the intervals
    session_days: np.ndarray  # sorted unique session day numbers
    lo: np.ndarray  # per interval: first session index in it
    hi: np.ndarray  # per interval: one past its last session index
    inst: np.ndarray  # per interval: instrument code


def _universe(intervals: pa.Table, sessions: pa.Table) -> _Universe:
    names = pc.unique(intervals["instrument_id"].combine_chunks()).drop_null()
    names = names.take(pc.sort_indices(names))
    session_days = np.unique(_days(sessions["session_date"].drop_null()))
    inst = _codes(intervals["instrument_id"], names)
    start = _days(intervals["valid_from"])
    end = intervals["valid_to"].combine_chunks()
    lo = np.searchsorted(session_days, start, "left")
    hi = np.where(
        end.is_null().to_numpy(zero_copy_only=False),
        session_days.size,
        np.searchsorted(session_days, _days(end), "left"),
    )
    keep = (inst >= 0) & (hi > lo)
    return _Universe(names, session_days, lo[keep], hi[keep], inst[keep])


@dataclass(frozen=True)
class _Prices:
    inst: np.ndarray  # sorted by (inst, day)
    day: np.ndarray
    close: np.ndarray
    valid: np.ndarray
    latest: int | None  # latest session_date overall (all instruments), None when empty


def _prices(prices: pa.Table, names: pa.Array, what: str) -> _Prices:
    days_all = _days(prices["session_date"])
    latest = int(days_all.max()) if days_all.size else None
    inst = _codes(prices["instrument_id"], names)
    rows = np.flatnonzero(inst >= 0)
    inst, day = inst[rows], days_all[rows]
    close = (
        prices["close"].combine_chunks().fill_null(math.nan).to_numpy(zero_copy_only=False)[rows]
    )
    order = np.lexsort((day, inst))
    inst, day, close = inst[order], day[order], close[order]
    dup = np.flatnonzero((inst[1:] == inst[:-1]) & (day[1:] == day[:-1]))
    if dup.size:
        listed = [
            f"{names[int(inst[i])].as_py()!r} on {_EPOCH + timedelta(days=int(day[i]))}"
            for i in dup[:MAX_LISTED]
        ]
        raise ValueError(
            f"several prices for one instrument and session in {what} ({dup.size} key(s)); "
            f"first {len(listed)}: " + "; ".join(listed)
        )
    valid = np.isfinite(close) & (close > 0)
    return _Prices(inst, day, close, valid, latest)


def _last_per_instrument(inst: np.ndarray, n: int) -> np.ndarray:
    """Index of each instrument's last row in ``inst`` (sorted), -1 when it has none."""
    out = np.full(n, -1, np.int64)
    if inst.size:
        ends = np.flatnonzero(np.r_[inst[1:] != inst[:-1], True])
        out[inst[ends]] = ends
    return out


def _check_share(value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"max_missing_share must be within [0, 1], got {value!r}")


def _no_data_cover(
    fetch_coverage: pa.Table, names: pa.Array, inst: np.ndarray, day: np.ndarray
) -> np.ndarray:
    """Cells (inst, day) inside a fetch range with a no-data status."""
    status = fetch_coverage["status"]
    unknown = pc.invert(pc.is_in(status, value_set=pa.array(FETCH_STATUSES))).fill_null(True)
    if pc.any(unknown).as_py():
        bad = sorted(set(status.filter(unknown).to_pylist()), key=str)
        raise ValueError(f"fetch_coverage status not in {FETCH_STATUSES}: {bad[:MAX_LISTED]}")
    f_inst = _codes(fetch_coverage["instrument_id"], names)
    f_from = _days(fetch_coverage["covered_from"])
    f_to = _days(fetch_coverage["covered_to"])
    usable = (
        (f_inst >= 0)
        & pc.is_in(status, value_set=pa.array(NO_DATA_STATUSES)).to_numpy(zero_copy_only=False)
        & pc.is_valid(fetch_coverage["covered_from"]).to_numpy(zero_copy_only=False)
        & pc.is_valid(fetch_coverage["covered_to"]).to_numpy(zero_copy_only=False)
        & (f_to >= f_from)
    )
    f_inst, f_from, f_to = f_inst[usable], f_from[usable], f_to[usable]
    covered = np.zeros(inst.size, bool)
    if not f_inst.size or not inst.size:
        return covered
    # Merge the ranges per instrument, then find each cell's merged range by binary search.
    order = np.lexsort((f_from, f_inst))
    f_inst, f_from, f_to = f_inst[order], f_from[order], f_to[order]
    shift = int(max(f_to.max(), day.max())) - int(min(f_from.min(), day.min())) + 2
    base = int(min(f_from.min(), day.min()))
    k_from = f_inst * shift + (f_from - base)
    k_to = np.maximum.accumulate(f_inst * shift + (f_to - base))
    new_run = np.r_[True, k_from[1:] > k_to[:-1]]
    run_end = np.r_[np.flatnonzero(new_run)[1:] - 1, f_inst.size - 1]
    starts, ends = k_from[new_run], k_to[run_end]
    key = inst * shift + (day - base)
    pos = np.searchsorted(starts, key, "right") - 1
    hit = pos >= 0
    covered[hit] = key[hit] <= ends[pos[hit]]
    return covered


def coverage(
    intervals: pa.Table,
    prices_resolved: pa.Table,
    sessions: pa.Table,
    fetch_coverage: pa.Table | None = None,
    *,
    max_missing_share: float = DEFAULT_MAX_MISSING_SHARE,
) -> CoverageResult:
    """Reason per expected (member, session) cell and missing share per session.

    ``per_member`` follows ``COVERAGE_PER_MEMBER_SCHEMA`` (every expected cell, ``ok`` included,
    sorted by instrument_id, session_date); ``per_session`` follows
    ``COVERAGE_PER_SESSION_SCHEMA`` (every session of ``sessions``, sorted). See the module
    docstring for the rules."""
    _check_share(max_missing_share)
    u = _universe(intervals, sessions)
    n_sessions = u.session_days.size

    # Expected cells: expand intervals to session indices, dedupe overlaps.
    lengths = u.hi - u.lo
    total = int(lengths.sum())
    first = np.repeat(u.lo - np.r_[0, np.cumsum(lengths)[:-1]], lengths)
    sidx = first + np.arange(total, dtype=np.int64)
    cell = np.repeat(u.inst, lengths) * n_sessions + sidx
    del first, sidx
    cell.sort()  # sort + adjacent-difference dedupe: np.unique's hash path is ~10x slower
    cell = cell[np.r_[True, cell[1:] != cell[:-1]]] if cell.size else cell
    inst, sidx = np.divmod(cell, n_sessions) if n_sessions else (cell, cell)
    del cell
    day = u.session_days[sidx]

    p = _prices(prices_resolved, u.names, "prices_resolved")
    # Price lookup per cell: binary search on (inst, day) keys.
    if p.day.size:
        base = int(min(p.day.min(), u.session_days.min(initial=p.day.min())))
        shift = int(max(p.day.max(), u.session_days.max(initial=p.day.max()))) - base + 1
        p_key = p.inst * shift + (p.day - base)
        key = inst * shift + (day - base)
        pos = np.minimum(np.searchsorted(p_key, key), p_key.size - 1)
        has = p_key[pos] == key
        valid = has & p.valid[pos]
        del p_key, key, pos
    else:
        has = valid = np.zeros(inst.size, bool)
    last_row = _last_per_instrument(p.inst, len(u.names))
    has_prices = last_row >= 0
    inst_last = np.zeros(len(u.names), np.int64)
    inst_last[has_prices] = p.day[last_row[has_prices]]
    after_last = has_prices[inst] & (day > inst_last[inst])
    not_published = day > p.latest if p.latest is not None else np.zeros(inst.size, bool)
    if fetch_coverage is not None:
        no_data = _no_data_cover(fetch_coverage, u.names, inst, day)
    else:
        no_data = np.zeros(inst.size, bool)

    reason = np.select(
        [valid, has, not_published, after_last, no_data],
        [
            _CODE["ok"],
            _CODE["price_invalid"],
            _CODE["session_not_published"],
            _CODE["delisted"],
            _CODE["source_has_no_data"],
        ],
        _CODE["not_fetched"],
    ).astype(np.int8)
    del has, valid, not_published, after_last, no_data

    members = np.bincount(sidx, minlength=n_sessions)
    delisted = np.bincount(sidx, weights=reason == _CODE["delisted"], minlength=n_sessions)
    missing = members - np.bincount(
        sidx, weights=(reason == _CODE["ok"]) | (reason == _CODE["delisted"]), minlength=n_sessions
    )
    delisted, missing = delisted.astype(np.int64), missing.astype(np.int64)
    share = np.divide(missing, members, out=np.zeros(n_sessions, np.float64), where=members > 0)
    over = share > max_missing_share
    session_dates = pa.array(u.session_days.astype(np.int32), pa.int32()).cast(pa.date32())
    per_session = pa.table(
        [session_dates, members.astype(np.int64), delisted, missing, share, over],
        schema=COVERAGE_PER_SESSION_SCHEMA,
    )
    per_member = pa.table(
        [
            u.names.take(pa.array(inst)),
            session_dates.take(pa.array(sidx)),
            pa.array(REASONS, pa.string()).take(pa.array(reason)),
        ],
        schema=COVERAGE_PER_MEMBER_SCHEMA,
    )
    return CoverageResult(per_member, per_session, not bool(over.any()))


def _check_exit_rule(rule: str, fixed_return: float | None, post_prices: pa.Table | None) -> None:
    if rule not in EXIT_RULES:
        raise ValueError(f"rule must be one of {EXIT_RULES}, got {rule!r}")
    if rule == "fixed_return" and (
        fixed_return is None or not math.isfinite(fixed_return) or fixed_return < -1
    ):
        raise ValueError(
            f"rule 'fixed_return' needs a finite fixed_return >= -1, got {fixed_return!r}"
        )
    if rule == "post_delisting_prices" and post_prices is None:
        raise ValueError("rule 'post_delisting_prices' needs a post_prices table")


def delisting_exits(
    intervals: pa.Table,
    prices_resolved: pa.Table,
    sessions: pa.Table,
    *,
    rule: Literal["last_close", "post_delisting_prices", "fixed_return"] = "last_close",
    fixed_return: float | None = None,
    post_prices: pa.Table | None = None,
) -> pa.Table:
    """One flagged exit per delisted member, ``DELISTING_EXITS_SCHEMA``, sorted by
    instrument_id. Default rule: last known close (D8). See the module docstring."""
    _check_exit_rule(rule, fixed_return, post_prices)
    u = _universe(intervals, sessions)
    p = _prices(prices_resolved, u.names, "prices_resolved")
    n = len(u.names)
    if not p.day.size or not u.session_days.size:
        return DELISTING_EXITS_SCHEMA.empty_table()

    last_row = _last_per_instrument(p.inst, n)
    has_prices = last_row >= 0
    inst_last = np.where(has_prices, p.day[np.maximum(last_row, 0)], 0)
    horizon = min(int(u.session_days[-1]), p.latest)
    # Session indices (inst_last, horizon] per interval; delisted when one interval holds one.
    a = np.searchsorted(u.session_days, inst_last[u.inst], "right")
    b = np.searchsorted(u.session_days, horizon, "right")
    holds = has_prices[u.inst] & (np.maximum(u.lo, a) < np.minimum(u.hi, b))
    delisted = np.zeros(n, bool)
    delisted[u.inst[holds]] = True
    codes = np.flatnonzero(delisted)
    if not codes.size:
        return DELISTING_EXITS_SCHEMA.empty_table()

    # Last valid close per instrument (rows sorted by (inst, day)).
    valid_rows = np.flatnonzero(p.valid)
    last_valid = np.full(n, -1, np.int64)
    lv = _last_per_instrument(p.inst[valid_rows], n)
    last_valid[lv >= 0] = valid_rows[lv[lv >= 0]]
    lv_rows = last_valid[codes]
    known = lv_rows >= 0
    exit_day = np.where(known, p.day[np.maximum(lv_rows, 0)], inst_last[codes])
    exit_price = np.where(known, p.close[np.maximum(lv_rows, 0)], math.nan)
    rules = np.full(codes.size, "last_close", object)

    if rule == "fixed_return":
        exit_price = exit_price * (1.0 + fixed_return)
        rules[:] = "fixed_return"
    elif rule == "post_delisting_prices":
        q = _prices(post_prices, u.names, "post_prices")
        exit_of = np.full(n, np.iinfo(np.int64).max, np.int64)
        exit_of[codes] = exit_day
        rows = np.flatnonzero(q.valid & (q.day >= exit_of[q.inst]))
        last_post = _last_per_instrument(q.inst[rows], n)[codes]
        found = last_post >= 0
        post_rows = rows[last_post[found]]
        exit_day[found] = q.day[post_rows]
        exit_price[found] = q.close[post_rows]
        rules[found] = "post_delisting_prices"

    return pa.table(
        [
            u.names.take(pa.array(codes)),
            pa.array(exit_day.astype(np.int32), pa.int32()).cast(pa.date32()),
            pa.array(exit_price, pa.float64(), from_pandas=True),
            pa.array(rules.tolist(), pa.string()),
            pa.array(np.ones(codes.size, bool)),
        ],
        schema=DELISTING_EXITS_SCHEMA,
    )
