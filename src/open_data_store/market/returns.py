"""Resolved prices and corporate actions per instrument, and the total-return recipe (spec §7.4).

``resolve_prices`` and ``resolve_actions`` read their table as-of (``known_at <= as_of`` and
``valid_date <= as_of``) and map each provider series to an instrument on the row's date through
the ``source_symbol`` identifiers, read latest (see ``_mapping``). Unmapped rows are dropped (the
coverage recipe reports them). Per (instrument, date) the first source of ``precedence`` that has
a row wins; sources absent from a non-empty ``precedence`` are ignored. With an empty
``precedence``, two series for one (instrument, date) raise ``ValueError``; so do two series of
the winning source (two symbols of one source mapped to one instrument on one date).

``total_return`` forward-cumulates a total-return index per instrument from the resolved rows:
base 1.0 on the instrument's first price session, ``tr_index(t) = tr_index(t-1) * (1 + r(t))``.
Levels are never back-adjusted: an action dated after T never changes a value at or before T.

Return of a price session t against the previous price session p of the instrument:

- ``r(t) = (close(t) + D) / (close(p) * F) - 1`` where F is the product of ``ratio_old /
  ratio_new`` over the splits and stock dividends placed on t, and D the sum of the cash amounts
  placed on t (regular and special cash dividends, return of capital, spin-off with an amount),
  reinvested at the ex-date close. ``amount`` is used as filed, gross or net (``amount_basis`` is
  not converted).
- Same-day split and dividend: the split applies first; a dividend with ``per_share_basis =
  "old_share"`` is converted to new shares (``amount * F``); ``"new_share"`` or null is used as is.
- Placement: an action whose ex-date is not a price session of the instrument moves to the next
  price session of that instrument; with no later price it is ignored. An action placed on the
  base row has no return to enter.
- A session of ``sessions`` strictly between p and t (a session with no price) flags ``gap``;
  the return is still taken against close(p). Gap detection is only as complete as ``sessions``.

Flags (column ``flags``: comma-joined, sorted, null when none) — never a guess:

- ``currency_mismatch``: the action's currency differs from the price currency of t (``GBP`` vs
  ``GBp`` included, null included); that action is not applied.
- ``currency_change``: the price currency of t differs from that of p; r(t) = 0, never converted.
- ``spin_off_unvalued``: a spin-off with no amount; not applied (manual source precedence, §7.4).
- ``rights_unvalued``: a rights issue; never applied.
- ``gap``: see above.

Malformed inputs raise ``ValueError``: a split or stock dividend without a positive finite
``ratio_old`` and ``ratio_new``; a cash dividend or return of capital without a finite amount;
a non-positive or non-finite close; two price rows for one (instrument, session).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal

import pyarrow as pa

from open_data_store.contracts.standard.market import CORPORATE_ACTIONS, PRICES
from open_data_store.market._mapping import resolve, symbol_mappings, symbol_token
from open_data_store.market.asof import asof

__all__ = [
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


def _check_precedence(precedence: Sequence[str]) -> None:
    if isinstance(precedence, str):
        raise TypeError("precedence must be a sequence of source codes, not a string")


def _resolve(
    table: pa.Table,
    identifiers: pa.Table,
    natural_key: Sequence[str],
    as_of: datetime,
    precedence: Sequence[str],
) -> tuple[pa.Table, list[str]]:
    """As-of rows kept by the mapping and precedence rules, with their instrument ids."""
    _check_precedence(precedence)
    visible = asof(table, natural_key, as_of)
    mappings = symbol_mappings(identifiers)
    cols = visible.select(["source_code", "source_symbol", "valid_date"]).to_pydict()
    rows = list(zip(cols["source_code"], cols["source_symbol"], cols["valid_date"], strict=True))
    instruments = [resolve(mappings, symbol_token(s, sym), d) for s, sym, d in rows]

    rank: dict[str, int] = {}
    for i, src in enumerate(precedence):
        rank.setdefault(src, i)

    def rank_of(src: str) -> int | None:
        return rank.get(src) if precedence else 0

    best: dict[tuple[str, date], int] = {}
    for (src, _, d), instrument in zip(rows, instruments, strict=True):
        r = rank_of(src)
        if instrument is None or r is None:
            continue
        key = (instrument, d)
        best[key] = min(best.get(key, r), r)

    keep: list[int] = []
    series: dict[tuple[str, date], set[str]] = defaultdict(set)
    for i, ((src, sym, d), instrument) in enumerate(zip(rows, instruments, strict=True)):
        r = rank_of(src)
        if instrument is None or r is None or r != best[(instrument, d)]:
            continue
        keep.append(i)
        series[(instrument, d)].add(symbol_token(src, sym))
    for (instrument, d), tokens in sorted(series.items()):
        if len(tokens) > 1:
            hint = "" if precedence else "; pass a source precedence"
            raise ValueError(
                f"several series for instrument {instrument!r} on {d.isoformat()}: "
                f"{sorted(tokens)}{hint}"
            )
    return visible.take(pa.array(keep, pa.int64())), [instruments[i] for i in keep]


def _output(
    rows: pa.Table, instruments: list[str], schema: pa.Schema, date_name: str, sort: list[str]
) -> pa.Table:
    data: dict[str, Any] = {"instrument_id": pa.array(instruments, pa.string())}
    for field in schema:
        if field.name == "instrument_id":
            continue
        source = "valid_date" if field.name == date_name else field.name
        data[field.name] = rows[source].cast(field.type)
    out = pa.table(data, schema=schema)
    return out.sort_by([(name, "ascending") for name in sort])


def resolve_prices(
    prices: pa.Table, identifiers: pa.Table, *, as_of: datetime, precedence: Sequence[str]
) -> pa.Table:
    """One price row per instrument and session as known at ``as_of`` (tz-aware).

    ``prices`` follows ``PRICES``, ``identifiers`` follows ``INSTRUMENT_IDENTIFIERS``. Returns
    ``RESOLVED_PRICES_SCHEMA`` rows sorted by (instrument_id, session_date). See the module
    docstring for mapping and precedence."""
    rows, instruments = _resolve(prices, identifiers, PRICES.natural_key, as_of, precedence)
    return _output(
        rows,
        instruments,
        RESOLVED_PRICES_SCHEMA,
        "session_date",
        ["instrument_id", "session_date"],
    )


def resolve_actions(
    actions: pa.Table, identifiers: pa.Table, *, as_of: datetime, precedence: Sequence[str]
) -> pa.Table:
    """Corporate actions per instrument and ex-date as known at ``as_of`` (tz-aware).

    ``actions`` follows ``CORPORATE_ACTIONS``. Per (instrument, ex-date) every action of the
    winning series is kept. Returns ``RESOLVED_ACTIONS_SCHEMA`` rows sorted by (instrument_id,
    ex_date, action_type, action_seq)."""
    rows, instruments = _resolve(
        actions, identifiers, CORPORATE_ACTIONS.natural_key, as_of, precedence
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


def _session_return(
    prev: tuple[date, float, str],
    cur: tuple[date, float, str],
    placed: list[dict[str, Any]],
    session_days: list[date],
) -> tuple[float, set[str]]:
    """Return of ``cur`` against ``prev`` with the actions placed on ``cur``, and its flags."""
    (prev_day, prev_close, prev_ccy), (day, close, ccy) = prev, cur
    flags: set[str] = set()
    if bisect_right(session_days, prev_day) < bisect_left(session_days, day):
        flags.add("gap")

    factor = 1.0
    for act in placed:
        if act["action_type"] in _RATIO_TYPES:
            if not (_finite_positive(act["ratio_old"]) and _finite_positive(act["ratio_new"])):
                raise ValueError(
                    f"{act['action_type']} of {act['instrument_id']!r} on "
                    f"{act['ex_date'].isoformat()} needs ratio_old and ratio_new > 0"
                )
            factor *= act["ratio_old"] / act["ratio_new"]

    cash = 0.0
    for act in placed:
        kind, amount = act["action_type"], act["amount"]
        if kind == "rights":
            flags.add("rights_unvalued")
            continue
        if kind == "spin_off" and amount is None:
            flags.add("spin_off_unvalued")
            continue
        if kind not in _CASH_TYPES and kind != "spin_off":
            continue
        if amount is None or not math.isfinite(amount):
            raise ValueError(
                f"{kind} of {act['instrument_id']!r} on {act['ex_date'].isoformat()} "
                "needs a finite amount"
            )
        if act["currency"] != ccy:
            flags.add("currency_mismatch")
            continue
        cash += amount * factor if act["per_share_basis"] == "old_share" else amount

    if ccy != prev_ccy:
        flags.add("currency_change")
        return 0.0, flags
    return (close + cash) / (prev_close * factor) - 1.0, flags


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
    price row. See the module docstring for the rules and flags."""
    if reinvest_at not in REINVEST_AT:
        raise ValueError(f"reinvest_at must be one of {REINVEST_AT}, got {reinvest_at!r}")
    session_days = sorted(set(sessions["session_date"].to_pylist()))

    by_instrument: dict[str, list[tuple[date, float, str]]] = defaultdict(list)
    cols = prices.select(["instrument_id", "session_date", "close", "currency"]).to_pydict()
    for instrument, day, close, ccy in zip(
        cols["instrument_id"], cols["session_date"], cols["close"], cols["currency"], strict=True
    ):
        if not _finite_positive(close):
            raise ValueError(f"close of {instrument!r} on {day.isoformat()} must be > 0")
        by_instrument[instrument].append((day, close, ccy))

    actions_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for act in actions.to_pylist():
        actions_by[act["instrument_id"]].append(act)

    out: list[tuple[str, date, float, str | None]] = []
    for instrument in sorted(by_instrument):
        series = sorted(by_instrument[instrument])
        days = [s[0] for s in series]
        for a, b in zip(days, days[1:], strict=False):
            if a == b:
                raise ValueError(f"several prices for {instrument!r} on {a.isoformat()}")
        placed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for act in actions_by.get(instrument, ()):
            i = bisect_left(days, act["ex_date"])
            if i < len(days):
                placed[i].append(act)

        level = 1.0
        out.append((instrument, days[0], level, None))
        for i in range(1, len(series)):
            r, flags = _session_return(series[i - 1], series[i], placed[i], session_days)
            level *= 1.0 + r
            out.append((instrument, days[i], level, ",".join(sorted(flags)) or None))

    return pa.table(
        {
            "instrument_id": pa.array([r[0] for r in out], pa.string()),
            "session_date": pa.array([r[1] for r in out], pa.date32()),
            "tr_index": pa.array([r[2] for r in out], pa.float64()),
            "flags": pa.array([r[3] for r in out], pa.string()),
        },
        schema=TOTAL_RETURN_SCHEMA,
    )
