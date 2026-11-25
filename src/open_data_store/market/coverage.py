"""Coverage of an index universe and delisting exits (spec §7.4, G4, D8).

An explained gap is still bias: ``coverage`` gives every expected (member, session) cell a reason,
and measures the missing share per session against ``max_missing_share``.

Inputs (pure Arrow tables):

- ``intervals``: a ``membership_intervals`` output, half-open ``[valid_from, valid_to)``,
  ``valid_to`` null = open; a null ``valid_from`` raises ``ValueError``. Every row is used: filter
  one ``index_code`` beforehand. Overlapping intervals of one instrument count once; overlapping
  or touching intervals form one membership *spell*.
- ``prices_resolved``: a ``resolve_prices`` output; two rows for one (instrument, session) or a
  null ``session_date`` raise ``ValueError``. A close is *valid* when finite and > 0.
- ``sessions``: a table with a ``session_date`` column (e.g. ``asof.sessions``); nulls raise.
- ``fetch_coverage``: optional ``FETCH_COVERAGE_SCHEMA`` rows, fetch-log ranges already mapped to
  instruments by the caller: ``[covered_from, covered_to]`` inclusive, ``status`` from
  ``FETCH_STATUSES`` (an unknown status raises ``ValueError``; a null bound covers nothing).

Expected cells: every (instrument, session) where the session lies in one of the instrument's
intervals (and so is <= the last session of ``sessions``). The *latest published session* is the
latest ``session_date`` of ``prices_resolved`` among instruments of ``intervals`` (ruling 23).
Reason per cell, first match wins:

1. ``ok``: a resolved price with a valid close;
2. ``price_invalid``: a resolved price whose close is not valid (null, non-finite or <= 0); a
   trailing invalid close never disappears;
3. ``session_not_published``: the session is after the latest published session (no member price
   at all: never matches);
4. delisting candidate (rulings 21, 22): the instrument has a valid close; the session is after
   its last price row (valid or not) and inside the merged membership spell that contains its
   last valid price date (a later spell, after a rejoin, is never a delisting). The candidate is
   ``delisted`` only when corroborated, else ``not_fetched``:

   - with ``fetch_coverage``: a row of the instrument covering the session has status ``ok``,
     ``empty`` or ``not_found``, and no covering row has status ``error`` or ``partial`` (a
     failed fetch vetoes, whatever its order);
   - without it: the tail gap, the number of sessions of ``sessions`` after the last valid price
     date and on or before the latest published session, is >= ``min_delisting_gap_sessions``
     (``DEFAULT_MIN_DELISTING_GAP_SESSIONS`` = 5, a placeholder to set from measurements).
5. ``source_has_no_data``: a ``fetch_coverage`` row of the instrument covers the session with
   status ``ok``, ``empty`` or ``not_found``;
6. ``not_fetched``: anything else.

``session_not_published`` is checked before the delisting step (controller ruling 3 lists them
the other way): otherwise a feed one day behind for everyone would read as delistings.
``unmapped`` and ``recycled_at_source`` stay in ``REASONS`` for producers that see source
symbols; ``per_member`` never carries them (it is keyed by instrument, and an unmapped symbol has
none).

Unmapped members: ``unresolved`` (optional) is a ``membership.unresolved_members`` output of the
same index and ``as_of``. Each of its symbols is an unmapped member on every session of
``[first_seen, valid_to)`` (``valid_to`` null = open), counted per session in ``unmapped``, in
``members`` and in ``missing``. **Without ``unresolved``, unmapped members are ignored**: a
universe whose members are mostly unmapped can pass.

Per session: ``members`` = expected cells + ``unmapped``, ``unmapped`` = unmapped members,
``delisted`` = cells ``delisted``, ``missing`` = cells neither ``ok`` nor ``delisted``, plus
``unmapped``, ``missing_share`` = missing / members (0.0 with no member),
``over_threshold`` = missing_share > ``max_missing_share``. ``passed`` = no session over it.
``max_missing_share`` defaults to ``DEFAULT_MAX_MISSING_SHARE`` (0.02), a placeholder to re-set
from measurements on real data (spec §11, review B5).

``delisting_exits`` takes the same inputs and parameters and exits exactly the instruments that
``coverage`` gives a ``delisted`` cell. Rules:

- ``last_close`` (default, D8): exit at the last valid close, ``exit_date`` its date (inside the
  spell, so inside a membership interval).
- ``fixed_return``: the ``last_close`` exit with ``exit_price = close * (1 + fixed_return)``;
  ``fixed_return`` must be finite and >= -1.
- ``post_delisting_prices``: the latest valid close of ``post_prices`` (same schema as
  ``prices_resolved``) dated on or after the ``last_close`` exit date and before the end of the
  spell, ``exit_date`` its date; instruments with none fall back to ``last_close`` (``rule`` then
  reads ``last_close``).

Every exit row is ``flagged``; rows are sorted by ``instrument_id``.
"""

from __future__ import annotations

import math
import operator
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
    "DEFAULT_MIN_DELISTING_GAP_SESSIONS",
    "DELISTING_EXITS_SCHEMA",
    "EXIT_RULES",
    "FAILED_FETCH_STATUSES",
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
FAILED_FETCH_STATUSES = ("error", "partial")
DEFAULT_MAX_MISSING_SHARE = 0.02
DEFAULT_MIN_DELISTING_GAP_SESSIONS = 5

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
        ("unmapped", pa.int64()),
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
# Composite (instrument, day) keys: instrument * _SHIFT + day + _OFFSET, day in int32 range.
_OFFSET = 1 << 32
_SHIFT = 1 << 34
_OPEN = 1 << 31  # day value of an open interval end, past any date32


@dataclass(frozen=True)
class CoverageResult:
    per_member: pa.Table
    per_session: pa.Table
    passed: bool


def _require_no_nulls(column: pa.ChunkedArray, name: str) -> None:
    if column.null_count:
        raise ValueError(f"{name} has {column.null_count} null value(s)")


def _days(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """date32 → int64 day numbers; nulls → INT32 min (callers mask them or refuse nulls)."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    out = arr.cast(pa.int32()).fill_null(np.iinfo(np.int32).min)
    return out.to_numpy(zero_copy_only=False).astype(np.int64)


def _codes(column: pa.ChunkedArray, names: pa.Array) -> np.ndarray:
    """Positions of ``column`` values in ``names`` (-1 when absent)."""
    idx = pc.index_in(column, value_set=names)
    idx = idx.combine_chunks() if isinstance(idx, pa.ChunkedArray) else idx
    return idx.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)


def _key(inst: np.ndarray, day: np.ndarray) -> np.ndarray:
    return inst * _SHIFT + day + _OFFSET


@dataclass(frozen=True)
class _Universe:
    names: pa.Array  # sorted unique instrument ids of the intervals
    session_days: np.ndarray  # sorted unique session day numbers
    lo: np.ndarray  # per interval with sessions: first session index in it
    hi: np.ndarray  # per interval with sessions: one past its last session index
    inst: np.ndarray  # per interval with sessions: instrument code
    spell_from: np.ndarray  # merged spells, sorted: composite start key
    spell_to: np.ndarray  # merged spells: composite end key (exclusive)


def _universe(intervals: pa.Table, sessions: pa.Table) -> _Universe:
    _require_no_nulls(intervals["valid_from"], "intervals.valid_from")
    _require_no_nulls(sessions["session_date"], "sessions.session_date")
    names = pc.unique(intervals["instrument_id"].combine_chunks()).drop_null()
    names = names.take(pc.sort_indices(names))
    session_days = np.unique(_days(sessions["session_date"]))
    inst = _codes(intervals["instrument_id"], names)
    start = _days(intervals["valid_from"])
    end_col = intervals["valid_to"].combine_chunks()
    is_open = end_col.is_null().to_numpy(zero_copy_only=False)
    end = np.where(is_open, _OPEN, _days(end_col))
    lo = np.searchsorted(session_days, start, "left")
    hi = np.searchsorted(session_days, end, "left")
    keep = (inst >= 0) & (hi > lo)

    # Spells: merge overlapping or touching intervals per instrument ([a, b) + [b, c) = [a, c)).
    real = (inst >= 0) & (end > start)
    k_from, k_to = _key(inst[real], start[real]), _key(inst[real], end[real])
    order = np.argsort(k_from, kind="stable")
    k_from, k_to = k_from[order], np.maximum.accumulate(k_to[order])
    new_run = np.r_[True, k_from[1:] > k_to[:-1]] if k_from.size else np.zeros(0, bool)
    run_end = np.r_[np.flatnonzero(new_run)[1:] - 1, k_from.size - 1].astype(np.int64)
    return _Universe(
        names,
        session_days,
        lo[keep],
        hi[keep],
        inst[keep],
        k_from[new_run],
        k_to[run_end] if k_from.size else k_to,
    )


@dataclass(frozen=True)
class _Prices:
    inst: np.ndarray  # sorted by (inst, day)
    day: np.ndarray
    close: np.ndarray
    valid: np.ndarray
    latest: int | None  # latest session_date of member instruments, None when none


def _prices(prices: pa.Table, names: pa.Array, what: str) -> _Prices:
    _require_no_nulls(prices["session_date"], f"{what}.session_date")
    inst = _codes(prices["instrument_id"], names)
    rows = np.flatnonzero(inst >= 0)
    inst, day = inst[rows], _days(prices["session_date"])[rows]
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
    latest = int(day.max()) if day.size else None
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


def _check_gap(value: int) -> int:
    """Any integral value >= 1 (``operator.index``: numpy integers too); bools are refused."""
    try:
        gap = None if isinstance(value, (bool, np.bool_)) else operator.index(value)
    except TypeError:
        gap = None
    if gap is None or gap < 1:
        raise ValueError(f"min_delisting_gap_sessions must be an integer >= 1, got {value!r}")
    return gap


def _check_statuses(fetch_coverage: pa.Table) -> None:
    status = fetch_coverage["status"]
    unknown = pc.invert(pc.is_in(status, value_set=pa.array(FETCH_STATUSES))).fill_null(True)
    if pc.any(unknown).as_py():
        bad = sorted(set(status.filter(unknown).to_pylist()), key=str)
        raise ValueError(f"fetch_coverage status not in {FETCH_STATUSES}: {bad[:MAX_LISTED]}")


def _cover(
    fetch_coverage: pa.Table,
    names: pa.Array,
    inst: np.ndarray,
    day: np.ndarray,
    statuses: tuple[str, ...],
) -> np.ndarray:
    """Cells (inst, day) inside a fetch range whose status is in ``statuses``."""
    f_inst = _codes(fetch_coverage["instrument_id"], names)
    f_from = _days(fetch_coverage["covered_from"])
    f_to = _days(fetch_coverage["covered_to"])
    usable = (
        (f_inst >= 0)
        & pc.is_in(fetch_coverage["status"], value_set=pa.array(statuses)).to_numpy(
            zero_copy_only=False
        )
        & pc.is_valid(fetch_coverage["covered_from"]).to_numpy(zero_copy_only=False)
        & pc.is_valid(fetch_coverage["covered_to"]).to_numpy(zero_copy_only=False)
        & (f_to >= f_from)
    )
    covered = np.zeros(inst.size, bool)
    if not usable.any() or not inst.size:
        return covered
    # Merge the inclusive ranges per instrument, then binary-search each cell's merged range.
    k_from = _key(f_inst[usable], f_from[usable])
    k_to = _key(f_inst[usable], f_to[usable])
    order = np.argsort(k_from, kind="stable")
    k_from, k_to = k_from[order], np.maximum.accumulate(k_to[order])
    new_run = np.r_[True, k_from[1:] > k_to[:-1]]
    run_end = np.r_[np.flatnonzero(new_run)[1:] - 1, k_from.size - 1]
    starts, ends = k_from[new_run], k_to[run_end]
    key = _key(inst, day)
    pos = np.searchsorted(starts, key, "right") - 1
    hit = pos >= 0
    covered[hit] = key[hit] <= ends[pos[hit]]
    return covered


@dataclass(frozen=True)
class _Cells:
    u: _Universe
    p: _Prices
    inst: np.ndarray  # per cell, sorted by (inst, session)
    sidx: np.ndarray
    reason: np.ndarray
    last_valid: np.ndarray  # per instrument: row of its last valid close in p, -1 when none
    spell_end: np.ndarray  # per instrument: end day (exclusive) of the last-valid-price spell


def _classify(
    intervals: pa.Table,
    prices_resolved: pa.Table,
    sessions: pa.Table,
    fetch_coverage: pa.Table | None,
    min_delisting_gap_sessions: int,
) -> _Cells:
    min_delisting_gap_sessions = _check_gap(min_delisting_gap_sessions)
    if fetch_coverage is not None:
        _check_statuses(fetch_coverage)
    u = _universe(intervals, sessions)
    n_inst, n_sessions = len(u.names), u.session_days.size

    # Expected cells: expand intervals to session indices, dedupe overlaps.
    lengths = u.hi - u.lo
    total = int(lengths.sum())
    first = np.repeat(u.lo - np.r_[0, np.cumsum(lengths)[:-1]], lengths)
    cell = np.repeat(u.inst, lengths) * n_sessions + first + np.arange(total, dtype=np.int64)
    del first
    cell.sort()  # sort + adjacent-difference dedupe: np.unique's hash path is ~10x slower
    cell = cell[np.r_[True, cell[1:] != cell[:-1]]] if cell.size else cell
    inst, sidx = np.divmod(cell, n_sessions) if n_sessions else (cell, cell)
    del cell
    day = u.session_days[sidx]

    p = _prices(prices_resolved, u.names, "prices_resolved")
    if p.day.size:
        p_key = _key(p.inst, p.day)
        key = _key(inst, day)
        pos = np.minimum(np.searchsorted(p_key, key), p_key.size - 1)
        has = p_key[pos] == key
        valid = has & p.valid[pos]
        del p_key, key, pos
    else:
        has = valid = np.zeros(inst.size, bool)
    not_published = day > p.latest if p.latest is not None else np.zeros(inst.size, bool)

    # Per instrument: last price row (any), last valid close, its spell, its tail gap.
    last_any = _last_per_instrument(p.inst, n_inst)
    last_day = np.full(n_inst, np.iinfo(np.int64).max, np.int64)
    last_day[last_any >= 0] = p.day[last_any[last_any >= 0]]
    valid_rows = np.flatnonzero(p.valid)
    lv = _last_per_instrument(p.inst[valid_rows], n_inst)
    last_valid = np.where(lv >= 0, valid_rows[np.maximum(lv, 0)] if valid_rows.size else -1, -1)
    has_valid = last_valid >= 0
    valid_day = np.where(has_valid, p.day[np.maximum(last_valid, 0)] if p.day.size else 0, 0)
    codes = np.arange(n_inst, dtype=np.int64)
    v_key = _key(codes, valid_day)
    spos = np.searchsorted(u.spell_from, v_key, "right") - 1
    in_spell = has_valid & (spos >= 0)
    in_spell[in_spell] = v_key[in_spell] < u.spell_to[spos[in_spell]]
    spell_end = np.full(n_inst, np.iinfo(np.int64).min, np.int64)
    spell_end[in_spell] = u.spell_to[spos[in_spell]] - codes[in_spell] * _SHIFT - _OFFSET

    candidate = in_spell[inst] & (day > last_day[inst]) & (day < spell_end[inst])
    if fetch_coverage is not None:
        no_data = _cover(fetch_coverage, u.names, inst, day, NO_DATA_STATUSES)
        failed = _cover(fetch_coverage, u.names, inst, day, FAILED_FETCH_STATUSES)
        corroborated = no_data & ~failed
        del failed
    else:
        no_data = np.zeros(inst.size, bool)
        # Sessions up to the latest published one (no member price: no candidate anyway).
        published = 0 if p.latest is None else np.searchsorted(u.session_days, p.latest, "right")
        gap = published - np.searchsorted(u.session_days, valid_day, "right")
        corroborated = (gap >= min_delisting_gap_sessions)[inst]

    reason = np.select(
        [valid, has, not_published, candidate & corroborated, candidate, no_data],
        [
            _CODE["ok"],
            _CODE["price_invalid"],
            _CODE["session_not_published"],
            _CODE["delisted"],
            _CODE["not_fetched"],
            _CODE["source_has_no_data"],
        ],
        _CODE["not_fetched"],
    ).astype(np.int8)
    return _Cells(u, p, inst, sidx, reason, last_valid, spell_end)


def coverage(
    intervals: pa.Table,
    prices_resolved: pa.Table,
    sessions: pa.Table,
    fetch_coverage: pa.Table | None = None,
    *,
    unresolved: pa.Table | None = None,
    max_missing_share: float = DEFAULT_MAX_MISSING_SHARE,
    min_delisting_gap_sessions: int = DEFAULT_MIN_DELISTING_GAP_SESSIONS,
) -> CoverageResult:
    """Reason per expected (member, session) cell and missing share per session.

    ``per_member`` follows ``COVERAGE_PER_MEMBER_SCHEMA`` (every expected cell, ``ok`` included,
    sorted by instrument_id, session_date); ``per_session`` follows
    ``COVERAGE_PER_SESSION_SCHEMA`` (every session of ``sessions``, sorted). ``unresolved`` is an
    ``unresolved_members`` output (same index, ``as_of`` and parameters as ``intervals``); without
    it, unmapped members are ignored and coverage can pass on a universe with holes. See the
    module docstring for the rules."""
    _check_share(max_missing_share)
    c = _classify(intervals, prices_resolved, sessions, fetch_coverage, min_delisting_gap_sessions)
    n_sessions = c.u.session_days.size
    sidx, reason = c.sidx, c.reason
    unmapped = (
        _unmapped(unresolved, c.u.session_days)
        if unresolved is not None
        else np.zeros(n_sessions, np.int64)
    )
    members = np.bincount(sidx, minlength=n_sessions) + unmapped
    delisted = np.bincount(sidx, weights=reason == _CODE["delisted"], minlength=n_sessions)
    missing = members - np.bincount(
        sidx, weights=(reason == _CODE["ok"]) | (reason == _CODE["delisted"]), minlength=n_sessions
    )
    delisted, missing = delisted.astype(np.int64), missing.astype(np.int64)
    share = np.divide(missing, members, out=np.zeros(n_sessions, np.float64), where=members > 0)
    over = share > max_missing_share
    session_dates = pa.array(c.u.session_days.astype(np.int32), pa.int32()).cast(pa.date32())
    per_session = pa.table(
        [session_dates, members.astype(np.int64), unmapped, delisted, missing, share, over],
        schema=COVERAGE_PER_SESSION_SCHEMA,
    )
    per_member = pa.table(
        [
            c.u.names.take(pa.array(c.inst)),
            session_dates.take(pa.array(sidx)),
            pa.array(REASONS, pa.string()).take(pa.array(reason)),
        ],
        schema=COVERAGE_PER_MEMBER_SCHEMA,
    )
    return CoverageResult(per_member, per_session, not bool(over.any()))


def _unmapped(unresolved: pa.Table, session_days: np.ndarray) -> np.ndarray:
    """Per session: unresolved symbols whose ``[first_seen, valid_to)`` covers it."""
    _require_no_nulls(unresolved["first_seen"], "unresolved.first_seen")
    start = _days(unresolved["first_seen"])
    end_col = unresolved["valid_to"].combine_chunks()
    end = np.where(end_col.is_null().to_numpy(zero_copy_only=False), _OPEN, _days(end_col))
    lo = np.searchsorted(session_days, start, "left")
    hi = np.searchsorted(session_days, end, "left")
    real = hi > lo
    diff = np.zeros(session_days.size + 1, np.int64)
    np.add.at(diff, lo[real], 1)
    np.add.at(diff, hi[real], -1)
    return np.cumsum(diff[:-1])


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
    fetch_coverage: pa.Table | None = None,
    min_delisting_gap_sessions: int = DEFAULT_MIN_DELISTING_GAP_SESSIONS,
) -> pa.Table:
    """One flagged exit per instrument with a ``delisted`` coverage cell (same inputs and
    parameters as ``coverage``), ``DELISTING_EXITS_SCHEMA``, sorted by instrument_id. Default
    rule: last valid close (D8). See the module docstring."""
    _check_exit_rule(rule, fixed_return, post_prices)
    c = _classify(intervals, prices_resolved, sessions, fetch_coverage, min_delisting_gap_sessions)
    n = len(c.u.names)
    delisted = np.zeros(n, bool)
    delisted[c.inst[c.reason == _CODE["delisted"]]] = True
    codes = np.flatnonzero(delisted)  # names are sorted: codes order = instrument_id order
    if not codes.size:
        return DELISTING_EXITS_SCHEMA.empty_table()
    rows = c.last_valid[codes]  # >= 0: a delisted cell needs a valid close
    exit_day = c.p.day[rows].copy()
    exit_price = c.p.close[rows].copy()
    rules = np.full(codes.size, "last_close", object)

    if rule == "fixed_return":
        exit_price = exit_price * (1.0 + fixed_return)
        rules[:] = "fixed_return"
    elif rule == "post_delisting_prices":
        q = _prices(post_prices, c.u.names, "post_prices")
        lo = np.full(n, np.iinfo(np.int64).max, np.int64)
        lo[codes] = exit_day
        hi = np.full(n, np.iinfo(np.int64).min, np.int64)
        hi[codes] = c.spell_end[codes]
        keep = np.flatnonzero(q.valid & (q.day >= lo[q.inst]) & (q.day < hi[q.inst]))
        last_post = _last_per_instrument(q.inst[keep], n)[codes]
        found = last_post >= 0
        post_rows = keep[last_post[found]]
        exit_day[found] = q.day[post_rows]
        exit_price[found] = q.close[post_rows]
        rules[found] = "post_delisting_prices"

    return pa.table(
        [
            c.u.names.take(pa.array(codes)),
            pa.array(exit_day.astype(np.int32), pa.int32()).cast(pa.date32()),
            pa.array(exit_price, pa.float64()),
            pa.array(rules.tolist(), pa.string()),
            pa.array(np.ones(codes.size, bool)),
        ],
        schema=DELISTING_EXITS_SCHEMA,
    )
