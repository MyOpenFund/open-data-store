"""Resolved prices and corporate actions per instrument, and the total-return recipe (spec §7.4).

``resolve_prices`` and ``resolve_actions`` read their table as-of (``known_at <= as_of`` and
``valid_date <= as_of``) and map each provider series to an instrument on the row's date through
the ``source_symbol`` identifiers, read latest (see ``_mapping``). Unmapped rows are dropped (the
coverage recipe reports them). Per (instrument, date) the first source of ``precedence`` that has
a row wins; sources absent from a non-empty ``precedence`` are ignored. With an empty
``precedence``, two series for one (instrument, date) raise ``ValueError``; so do two series of
the winning source (two symbols of one source mapped to one instrument on one date). Those are
mapping errors, fixed upstream in the identifiers; the message lists up to ``MAX_LISTED`` keys.

``total_return`` forward-cumulates a total-return index per instrument from the resolved rows:
base 1.0 on the instrument's first valid price session, ``tr_index(t) = tr_index(t-1) * (1 +
r(t))``. Levels are never back-adjusted: an action dated after T never changes a value at or
before T.

Return of a price session t against the previous valid price session p of the instrument:

- Without actions, ``r(t) = close(t) / close(p) - 1``.
- Placement: an action whose ex-date is not a price session of the instrument moves to the next
  valid price session of that instrument; with no later price it is ignored. An action placed on
  the base row has no return to enter and is ignored, unflagged.
- The actions placed on t are applied in ex-date order to one share held at p's close: on each
  ex-date, splits and stock dividends first (shares ``n *= ratio_new / ratio_old``), then cash
  (regular and special cash dividends, return of capital, a spin-off with an amount) credited
  per share then held, ``cash += n * amount``. Cash is reinvested at t's close (the ex-date
  close, or the next price when the ex-date had none): ``r(t) = (n * close(t) + cash) /
  close(p) - 1``. On a single ex-date this is ``(close(t) + D) / (close(p) * F) - 1`` with
  ``F = ratio_old / ratio_new``. An amount with ``per_share_basis = "old_share"`` is converted to
  new shares of a same-day split (``amount * F``); ``"new_share"`` or null is used as is. A
  dividend dated before a later split within one gap is credited on the pre-split share count,
  never rescaled by that later split.
- ``amount`` is used as filed, gross or net (``amount_basis`` is not converted).
- A session of ``sessions`` strictly between p and t (a session with no price) flags ``gap``;
  the return is still taken against close(p). Gap detection is only as complete as ``sessions``.

Flags (column ``flags``: comma-joined, sorted, null when none) — never a guess, and a malformed
row never aborts the universe:

- ``currency_mismatch``: the action's currency differs from the price currency of t (``GBP`` vs
  ``GBp`` included); that action is not applied. A null action currency is a mismatch too:
  adapters must fill the currency of every cash action, or the cash is dropped (flagged).
- ``currency_change``: the price currency of t differs from that of p; r(t) = 0, never converted.
- ``spin_off_unvalued``: a spin-off with no amount; not applied (manual source precedence, §7.4).
- ``rights_unvalued``: a rights issue; never applied.
- ``action_invalid``: a split or stock dividend without positive finite ratios, or a cash action
  (spin-off with an amount included) whose amount is not finite and positive; not applied.
- ``price_invalid``: a close that is not finite and positive is treated as a missing price; the
  instrument's next valid row carries ``price_invalid`` and ``gap`` (``price_invalid`` alone when
  it is the base row).
- ``gap``: see above.

Two price rows for one (instrument, session) in ``total_return``'s input raise ``ValueError``
(the input is not a ``resolve_prices`` output).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from itertools import groupby
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from open_data_store.contracts.standard.market import CORPORATE_ACTIONS, PRICES
from open_data_store.market._mapping import MAX_LISTED, MappedRows, map_rows
from open_data_store.market.asof import asof

__all__ = [
    "MAX_LISTED",
    "REINVEST_AT",
    "RESOLVED_ACTIONS_SCHEMA",
    "RESOLVED_PRICES_SCHEMA",
    "TOTAL_RETURN_SCHEMA",
    "resolve_actions",
    "resolve_prices",
    "total_return",
]

REINVEST_AT = ("ex_date_close",)

RESOLVED_PRICES_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("session_date", pa.date32()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("currency", pa.string()),
        ("source_code", pa.string()),
        ("source_symbol", pa.string()),
    ]
)
RESOLVED_ACTIONS_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("ex_date", pa.date32()),
        ("action_type", pa.string()),
        ("action_seq", pa.int16()),
        ("amount", pa.float64()),
        ("currency", pa.string()),
        ("amount_basis", pa.string()),
        ("per_share_basis", pa.string()),
        ("ratio_old", pa.float64()),
        ("ratio_new", pa.float64()),
        ("spinoff_symbol", pa.string()),
        ("source_code", pa.string()),
        ("source_symbol", pa.string()),
    ]
)
TOTAL_RETURN_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("session_date", pa.date32()),
        ("tr_index", pa.float64()),
        ("flags", pa.string()),
    ]
)

_RATIO_TYPES = frozenset({"split", "stock_dividend"})
_CASH_TYPES = frozenset({"cash_dividend_regular", "cash_dividend_special", "return_of_capital"})
_EPOCH = date(1970, 1, 1)


def _check_precedence(precedence: Sequence[str]) -> None:
    if isinstance(precedence, str):
        raise TypeError("precedence must be a sequence of source codes, not a string")


def _resolve(
    table: pa.Table,
    identifiers: pa.Table,
    natural_key: Sequence[str],
    as_of: datetime,
    precedence: Sequence[str],
    columns: Sequence[str],
) -> tuple[pa.Table, pa.Array]:
    """As-of rows kept by the mapping and precedence rules (only ``columns``), and their
    instrument ids; sorted by (instrument_id, valid_date, source rank)."""
    _check_precedence(precedence)
    needed = list(dict.fromkeys([*natural_key, "valid_date", "known_at", *columns]))
    visible = asof(table.select(needed), natural_key, as_of)
    tokens = pc.binary_join_element_wise(visible["source_code"], visible["source_symbol"], ":")
    mapped = map_rows(identifiers, tokens, visible["valid_date"])
    del tokens
    inst = mapped.instrument_codes
    days = visible["valid_date"].combine_chunks().cast(pa.int32()).to_numpy(zero_copy_only=False)
    if precedence:
        order = pa.array(list(dict.fromkeys(precedence)), pa.string())
        ranks = pc.index_in(visible["source_code"], value_set=order).combine_chunks()
        rank = ranks.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int32)
    else:
        rank = np.zeros(len(inst), np.int32)
    rows = np.flatnonzero((inst >= 0) & (rank >= 0))
    rows = rows[np.lexsort((rank[rows], days[rows], inst[rows]))]
    inst, days, rank = inst[rows], days[rows], rank[rows]
    tok = mapped.token_codes[rows]

    # Runs of one (instrument, date), best rank first: keep the run's best-rank rows.
    starts = np.ones(len(rows), bool)
    starts[1:] = (inst[1:] != inst[:-1]) | (days[1:] != days[:-1])
    head = np.maximum.accumulate(np.where(starts, np.arange(len(rows)), 0))
    keep = rank == rank[head]
    conflict = keep & (tok != tok[head])
    if conflict.any():
        _raise_conflict(mapped, inst, days, tok, head, keep, conflict, bool(precedence))
    rows, inst = rows[keep], inst[keep]
    names = pa.array(mapped.instruments, pa.string())
    return visible.select(list(columns)).take(pa.array(rows, pa.int64())), names.take(
        pa.array(inst)
    )


def _raise_conflict(
    mapped: MappedRows,
    inst: np.ndarray,
    days: np.ndarray,
    tok: np.ndarray,
    head: np.ndarray,
    keep: np.ndarray,
    conflict: np.ndarray,
    has_precedence: bool,
) -> None:
    heads = np.unique(head[conflict])
    listed = []
    for h in heads[:MAX_LISTED]:
        run = (head == h) & keep
        series = sorted({mapped.tokens[t] for t in tok[run]})
        on = _EPOCH + timedelta(days=int(days[h]))
        listed.append(f"{mapped.instruments[inst[h]]!r} on {on.isoformat()}: {series}")
    hint = "" if has_precedence else "; pass a source precedence"
    raise ValueError(
        f"several series for one instrument and date ({heads.size} key(s){hint}); "
        f"first {len(listed)}: " + "; ".join(listed)
    )


def _output(
    rows: pa.Table,
    instruments: pa.Array,
    schema: pa.Schema,
    date_name: str,
    sort: list[str] | None,
) -> pa.Table:
    data: dict[str, Any] = {"instrument_id": instruments}
    for field in schema:
        if field.name == "instrument_id":
            continue
        source = "valid_date" if field.name == date_name else field.name
        data[field.name] = rows[source].cast(field.type)
    out = pa.table(data, schema=schema)
    return out.sort_by([(name, "ascending") for name in sort]) if sort else out


def resolve_prices(
    prices: pa.Table, identifiers: pa.Table, *, as_of: datetime, precedence: Sequence[str]
) -> pa.Table:
    """One price row per instrument and session as known at ``as_of`` (tz-aware).

    ``prices`` follows ``PRICES``, ``identifiers`` follows ``INSTRUMENT_IDENTIFIERS``. Returns
    ``RESOLVED_PRICES_SCHEMA`` rows sorted by (instrument_id, session_date). See the module
    docstring for mapping and precedence."""
    columns = ["valid_date", "close", "volume", "currency", "source_code", "source_symbol"]
    rows, instruments = _resolve(
        prices, identifiers, PRICES.natural_key, as_of, precedence, columns
    )
    # One row per (instrument, session), already in (instrument_id, session_date) order.
    return _output(rows, instruments, RESOLVED_PRICES_SCHEMA, "session_date", None)


def resolve_actions(
    actions: pa.Table, identifiers: pa.Table, *, as_of: datetime, precedence: Sequence[str]
) -> pa.Table:
    """Corporate actions per instrument and ex-date as known at ``as_of`` (tz-aware).

    ``actions`` follows ``CORPORATE_ACTIONS``. Per (instrument, ex-date) every action of the
    winning series is kept. Returns ``RESOLVED_ACTIONS_SCHEMA`` rows sorted by (instrument_id,
    ex_date, action_type, action_seq)."""
    columns = [n for n in RESOLVED_ACTIONS_SCHEMA.names if n not in ("instrument_id", "ex_date")]
    rows, instruments = _resolve(
        actions,
        identifiers,
        CORPORATE_ACTIONS.natural_key,
        as_of,
        precedence,
        ["valid_date", *columns],
    )
    return _output(
        rows,
        instruments,
        RESOLVED_ACTIONS_SCHEMA,
        "ex_date",
        ["instrument_id", "ex_date", "action_type", "action_seq"],
    )


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _action_return(
    prev_close: float, close: float, currency: str, placed: list[dict[str, Any]]
) -> tuple[float, set[str]]:
    """Return of a session carrying ``placed`` actions, and the flags they raise."""
    flags: set[str] = set()
    shares, cash = 1.0, 0.0
    ordered = sorted(placed, key=lambda a: a["ex_date"])
    for _, group in groupby(ordered, key=lambda a: a["ex_date"]):
        acts = list(group)
        day_factor = 1.0  # ratio_old / ratio_new of this ex-date's splits
        for act in acts:
            if act["action_type"] not in _RATIO_TYPES:
                continue
            old, new = act["ratio_old"], act["ratio_new"]
            if not (_finite_positive(old) and _finite_positive(new)):
                flags.add("action_invalid")
                continue
            shares *= new / old
            day_factor *= old / new
        for act in acts:
            kind, amount = act["action_type"], act["amount"]
            if kind == "rights":
                flags.add("rights_unvalued")
                continue
            if kind == "spin_off" and amount is None:
                flags.add("spin_off_unvalued")
                continue
            if kind not in _CASH_TYPES and kind != "spin_off":
                continue
            if not _finite_positive(amount):
                flags.add("action_invalid")
                continue
            if act["currency"] != currency:
                flags.add("currency_mismatch")
                continue
            per_share = amount * day_factor if act["per_share_basis"] == "old_share" else amount
            cash += shares * per_share
    return (shares * close + cash) / prev_close - 1.0, flags


def _empty_total_return() -> pa.Table:
    return TOTAL_RETURN_SCHEMA.empty_table()


def total_return(
    prices: pa.Table,
    actions: pa.Table,
    sessions: pa.Table,
    *,
    reinvest_at: Literal["ex_date_close"] = "ex_date_close",
) -> pa.Table:
    """Forward-cumulated total-return index per instrument (never back-adjusted levels).

    ``prices`` is a ``resolve_prices`` output, ``actions`` a ``resolve_actions`` output and
    ``sessions`` a table with a ``session_date`` column (e.g. ``asof.sessions``), used to detect
    gaps. Returns ``TOTAL_RETURN_SCHEMA`` rows sorted by (instrument_id, session_date): one per
    valid price row. See the module docstring for the rules and flags."""
    if reinvest_at not in REINVEST_AT:
        raise ValueError(f"reinvest_at must be one of {REINVEST_AT}, got {reinvest_at!r}")
    p = prices.select(["instrument_id", "session_date", "close", "currency"])
    p = p.sort_by([("instrument_id", "ascending"), ("session_date", "ascending")])
    p = p.combine_chunks()
    if p.num_rows == 0:
        return _empty_total_return()

    inst = pc.dictionary_encode(p["instrument_id"]).chunk(0)
    inst_codes = inst.indices.to_numpy()
    days = p["session_date"].chunk(0).cast(pa.int32()).to_numpy()
    close = p["close"].chunk(0).fill_null(math.nan).to_numpy()
    ccy = pc.dictionary_encode(p["currency"]).chunk(0)
    ccy_codes = ccy.indices.fill_null(-1).to_numpy()

    same = np.zeros(len(days), bool)
    same[1:] = inst_codes[1:] == inst_codes[:-1]
    dup = np.flatnonzero(same & (np.r_[-1, days[:-1]] == days))
    if dup.size:
        listed = [
            f"{inst.dictionary[inst_codes[i]].as_py()!r} on {p['session_date'][i].as_py()}"
            for i in dup[:MAX_LISTED]
        ]
        raise ValueError(
            f"several prices for one instrument and session ({dup.size} key(s)); "
            f"first {len(listed)}: " + "; ".join(listed)
        )

    # price_invalid: drop the row, flag the instrument's next valid row.
    valid = np.isfinite(close) & (close > 0)
    valid_idx = np.flatnonzero(valid)
    bad_idx = np.flatnonzero(~valid)
    nxt = np.searchsorted(valid_idx, bad_idx)
    keep = nxt < valid_idx.size
    nxt, bad_idx = nxt[keep], bad_idx[keep]
    nxt = nxt[inst_codes[valid_idx[nxt]] == inst_codes[bad_idx]]
    price_invalid = np.zeros(valid_idx.size, bool)
    price_invalid[nxt] = True

    inst_codes, days, close, ccy_codes = (
        a[valid_idx] for a in (inst_codes, days, close, ccy_codes)
    )
    n = days.size
    if n == 0:
        return _empty_total_return()
    same = np.zeros(n, bool)
    same[1:] = inst_codes[1:] == inst_codes[:-1]
    prev_days = np.r_[days[0], days[:-1]]
    prev_close = np.r_[close[0], close[:-1]]

    session_days = np.unique(
        sessions["session_date"].combine_chunks().cast(pa.int32()).to_numpy(zero_copy_only=False)
    )
    between = np.searchsorted(session_days, days, "left") - np.searchsorted(
        session_days, prev_days, "right"
    )
    gap = same & ((between > 0) | price_invalid)
    ccy_change = same & (np.r_[ccy_codes[0], ccy_codes[:-1]] != ccy_codes)

    r = np.where(same, close / prev_close - 1.0, 0.0)
    extra: dict[int, set[str]] = {}

    starts = np.flatnonzero(~same)
    ends = np.r_[starts[1:], n]
    segment = {
        inst.dictionary[inst_codes[s]].as_py(): (int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    }
    placed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for act in actions.to_pylist():
        span = segment.get(act["instrument_id"])
        if span is None:
            continue
        s, e = span
        ex = (act["ex_date"] - _EPOCH).days
        j = s + int(np.searchsorted(days[s:e], ex, "left"))
        if s < j < e:  # past the last price: ignored; on the base row: no return to enter
            placed[j].append(act)
    ccy_values = ccy.dictionary.to_pylist()
    for j, acts in placed.items():
        currency = ccy_values[ccy_codes[j]] if ccy_codes[j] >= 0 else None
        r[j], extra[j] = _action_return(prev_close[j], close[j], currency, acts)
    r[ccy_change] = 0.0

    level = np.empty(n)
    for s, e in zip(starts, ends, strict=True):
        level[s:e] = np.cumprod(1.0 + r[s:e])

    flags: list[str | None] = [None] * n
    marks = {
        "gap": gap,
        "currency_change": ccy_change,
        "price_invalid": price_invalid,
    }
    flagged = set(np.flatnonzero(gap | ccy_change | price_invalid).tolist()) | set(extra)
    for j in flagged:
        names = {name for name, mask in marks.items() if mask[j]} | extra.get(j, set())
        flags[j] = ",".join(sorted(names)) or None

    return pa.table(
        {
            "instrument_id": p["instrument_id"].take(pa.array(valid_idx)),
            "session_date": p["session_date"].take(pa.array(valid_idx)),
            "tr_index": pa.array(level, pa.float64()),
            "flags": pa.array(flags, pa.string()),
        },
        schema=TOTAL_RETURN_SCHEMA,
    )
