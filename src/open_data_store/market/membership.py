"""Point-in-time index membership: who was in an index at date t, as known at t (spec §7.4).

``membership_intervals`` turns index snapshots into half-open instrument intervals
``[valid_from, valid_to)``. Snapshots are read as-of (``known_at <= as_of`` and
``valid_date <= as_of``); symbol mappings are read latest, as the per-input policy says.

Rules:

- A snapshot dated D (as filed, weekend allowed) is valid from D until the next snapshot date of
  the chosen series. A gap longer than ``staleness_days`` closes the interval at
  D + ``staleness_days``. The last snapshot stays open (``valid_to`` null) only while
  (UTC date of ``as_of`` - D) <= ``staleness_days``; otherwise it closes at D + ``staleness_days``.
- Source choice per snapshot date: the first source of ``precedence`` that has a snapshot that
  date (a source "has a snapshot" when it filed any row for the index that date, even rows with
  ``is_member = false``). Sources absent from a non-empty ``precedence`` are ignored. With an
  empty ``precedence``, more than one source for the index is ambiguous and raises.
- Members: rows with ``is_member = false`` are removed. A symbol maps to an instrument through
  ``instrument_identifiers`` rows with ``id_scheme = "source_symbol"`` and
  ``id_value = "<source_code>:<source_symbol>"`` covering D (``valid_date <= D`` and
  ``valid_to`` null or D < ``valid_to``), so a recycled symbol maps to different instruments on
  disjoint ranges. No covering row: the member is unresolved (``unresolved_members``), never in
  the intervals. Covering rows naming different instruments raise ``ValueError``.
- Only the chosen source of a date is resolved: when the preferred source's symbol is unmapped on
  a date, the instrument drops out for that date even if a lower-precedence source has it mapped,
  and the symbol appears in ``unresolved_members``. That is a universe hole the caller must check;
  symbols of a losing source are never reported.
- Several symbols of one snapshot mapping to one instrument count once. Adjacent intervals of one
  instrument from the same source are merged; a change of source splits the interval at the
  switch date (no gap), so each row keeps the source it came from.

Precondition: ``asof`` selects per natural key (source, index, symbol, date), so a restated
snapshot that merely omits a symbol leaves the old row for that symbol visible. Ingestors must
emit an ``is_member = false`` row for every symbol a restatement removes.

``staleness_days`` defaults to ``DEFAULT_STALENESS_DAYS`` (45): a documented parameter, to be set
from measurements on real data (spec §11).
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pyarrow as pa
import pyarrow.compute as pc

from open_data_store.contracts.standard.market import INDEX_SNAPSHOTS
from open_data_store.market._mapping import resolve, symbol_mappings, symbol_token
from open_data_store.market.asof import asof

__all__ = [
    "DEFAULT_STALENESS_DAYS",
    "INTERVALS_SCHEMA",
    "UNRESOLVED_SCHEMA",
    "membership_intervals",
    "unresolved_members",
]

DEFAULT_STALENESS_DAYS = 45

INTERVALS_SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("index_code", pa.string()),
        ("valid_from", pa.date32()),
        ("valid_to", pa.date32()),
        ("source_code", pa.string()),
    ]
)
UNRESOLVED_SCHEMA = pa.schema(
    [
        ("source_code", pa.string()),
        ("source_symbol", pa.string()),
        ("first_seen", pa.date32()),
        ("last_seen", pa.date32()),
    ]
)


@dataclass(frozen=True)
class _Snapshot:
    valid_date: date
    source_code: str
    instruments: frozenset[str]
    unresolved: frozenset[str]  # source symbols


def _validate(as_of: datetime, staleness_days: int, precedence: Sequence[str]) -> int:
    """Check the parameters; return ``staleness_days`` as a plain int."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    if isinstance(staleness_days, bool):
        raise TypeError("staleness_days must be an integral number of days, not a bool")
    try:
        days = operator.index(staleness_days)
    except TypeError:
        raise TypeError(
            f"staleness_days must be an integral number of days, got {staleness_days!r}"
        ) from None
    if days <= 0:
        raise ValueError(f"staleness_days must be > 0, got {days}")
    if isinstance(precedence, str):
        raise TypeError("precedence must be a sequence of source codes, not a string")
    return days


def _chosen_snapshots(
    snapshots: pa.Table,
    identifiers: pa.Table,
    index_code: str,
    as_of: datetime,
    precedence: Sequence[str],
) -> list[_Snapshot]:
    visible = asof(snapshots, INDEX_SNAPSHOTS.natural_key, as_of)
    visible = visible.filter(pc.equal(visible["index_code"], index_code))
    cols = visible.select(["valid_date", "source_code", "source_symbol", "is_member"]).to_pydict()

    by_date: dict[date, dict[str, list[tuple[str, bool]]]] = defaultdict(lambda: defaultdict(list))
    for d, src, sym, member in zip(
        cols["valid_date"],
        cols["source_code"],
        cols["source_symbol"],
        cols["is_member"],
        strict=True,
    ):
        by_date[d][src].append((sym, member))

    sources = {src for per_source in by_date.values() for src in per_source}
    order = list(precedence)
    if not order:
        if len(sources) > 1:
            raise ValueError(
                f"index {index_code!r} has snapshots from several sources {sorted(sources)}; "
                "pass a source precedence"
            )
        order = sorted(sources)

    mappings = symbol_mappings(identifiers)
    chosen: list[_Snapshot] = []
    for d in sorted(by_date):
        src = next((s for s in order if s in by_date[d]), None)
        if src is None:
            continue
        instruments: set[str] = set()
        unresolved: set[str] = set()
        for sym, member in by_date[d][src]:
            if not member:
                continue
            instrument = resolve(mappings, symbol_token(src, sym), d)
            if instrument is None:
                unresolved.add(sym)
            else:
                instruments.add(instrument)
        chosen.append(_Snapshot(d, src, frozenset(instruments), frozenset(unresolved)))
    return chosen


def membership_intervals(
    snapshots: pa.Table,
    identifiers: pa.Table,
    *,
    index_code: str,
    as_of: datetime,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
    precedence: Sequence[str] = (),
) -> pa.Table:
    """Half-open membership intervals of ``index_code`` as known at ``as_of`` (tz-aware).

    ``snapshots`` follows ``INDEX_SNAPSHOTS``, ``identifiers`` follows ``INSTRUMENT_IDENTIFIERS``.
    Returns ``INTERVALS_SCHEMA`` rows sorted by (instrument_id, valid_from); ``valid_to`` null
    means still a member at the last snapshot. See the module docstring for the rules.

    ``staleness_days`` accepts any integral value (``operator.index``) > 0. With several sources
    in ``precedence``, rows are not one per membership spell: a source switch splits a spell into
    contiguous rows. Coalesce contiguous rows of an instrument before deriving entries, exits or
    turnover."""
    days = _validate(as_of, staleness_days, precedence)
    chosen = _chosen_snapshots(snapshots, identifiers, index_code, as_of, precedence)
    cap = timedelta(days=days)
    as_of_date = as_of.astimezone(UTC).date()

    # instrument -> [(valid_from, valid_to, source_code)] in date order, merged on the fly.
    spans: dict[str, list[tuple[date, date | None, str]]] = defaultdict(list)
    for i, snap in enumerate(chosen):
        d = snap.valid_date
        if i + 1 < len(chosen):
            end: date | None = min(chosen[i + 1].valid_date, d + cap)
        else:
            end = None if (as_of_date - d) <= cap else d + cap
        for instrument in snap.instruments:
            runs = spans[instrument]
            if runs and runs[-1][1] == d and runs[-1][2] == snap.source_code:
                runs[-1] = (runs[-1][0], end, snap.source_code)
            else:
                runs.append((d, end, snap.source_code))

    rows = [
        (instrument, start, end, src)
        for instrument in sorted(spans)
        for start, end, src in spans[instrument]
    ]
    return pa.table(
        {
            "instrument_id": pa.array([r[0] for r in rows], pa.string()),
            "index_code": pa.array([index_code] * len(rows), pa.string()),
            "valid_from": pa.array([r[1] for r in rows], pa.date32()),
            "valid_to": pa.array([r[2] for r in rows], pa.date32()),
            "source_code": pa.array([r[3] for r in rows], pa.string()),
        },
        schema=INTERVALS_SCHEMA,
    )


def unresolved_members(
    snapshots: pa.Table,
    identifiers: pa.Table,
    *,
    index_code: str,
    as_of: datetime,
    precedence: Sequence[str] = (),
) -> pa.Table:
    """Members of the chosen snapshots whose symbol maps to no instrument on the snapshot date.

    Same snapshot selection as ``membership_intervals``. Returns ``UNRESOLVED_SCHEMA`` rows: the
    first and last snapshot dates where the symbol was an unresolved member, sorted by
    (source_code, source_symbol)."""
    _validate(as_of, DEFAULT_STALENESS_DAYS, precedence)
    seen: dict[tuple[str, str], tuple[date, date]] = {}
    for snap in _chosen_snapshots(snapshots, identifiers, index_code, as_of, precedence):
        for sym in snap.unresolved:
            key = (snap.source_code, sym)
            first = seen[key][0] if key in seen else snap.valid_date
            seen[key] = (first, snap.valid_date)
    keys = sorted(seen)
    return pa.table(
        {
            "source_code": pa.array([k[0] for k in keys], pa.string()),
            "source_symbol": pa.array([k[1] for k in keys], pa.string()),
            "first_seen": pa.array([seen[k][0] for k in keys], pa.date32()),
            "last_seen": pa.array([seen[k][1] for k in keys], pa.date32()),
        },
        schema=UNRESOLVED_SCHEMA,
    )
