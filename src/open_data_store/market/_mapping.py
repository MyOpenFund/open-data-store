"""Symbol → instrument resolution shared by the market recipes (private).

A provider series (source_code, source_symbol) maps to an instrument through
``instrument_identifiers`` rows with ``id_scheme = "source_symbol"`` and
``id_value = "<source_code>:<source_symbol>"``, read latest (the per-input policy for identifier
mappings), applied on a date D only when ``[valid_date, valid_to)`` covers D. Covering rows that
name different instruments raise ``ValueError``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from open_data_store.contracts.standard.market import INSTRUMENT_IDENTIFIERS
from open_data_store.market.asof import latest

SYMBOL_SCHEME = "source_symbol"

Mappings = dict[str, list[tuple[date, date | None, str]]]


def symbol_mappings(identifiers: pa.Table) -> Mappings:
    """id_value -> [(valid_date, valid_to, instrument_id)] for source_symbol rows, read latest."""
    rows = latest(identifiers, INSTRUMENT_IDENTIFIERS.natural_key)
    rows = rows.filter(pc.equal(rows["id_scheme"], SYMBOL_SCHEME))
    out: Mappings = defaultdict(list)
    cols = rows.select(["id_value", "valid_date", "valid_to", "instrument_id"]).to_pydict()
    for value, start, end, instrument in zip(
        cols["id_value"], cols["valid_date"], cols["valid_to"], cols["instrument_id"], strict=True
    ):
        out[value].append((start, end, instrument))
    return out


def symbol_token(source_code: str, source_symbol: str) -> str:
    """The ``id_value`` of a provider series: ``"<source_code>:<source_symbol>"``."""
    return f"{source_code}:{source_symbol}"


MAX_LISTED = 10  # offending keys quoted in an error message


@dataclass(frozen=True)
class MappedRows:
    """Per-row result of ``map_rows``, aligned with its input.

    ``instrument_codes[i]`` indexes ``instruments`` (sorted, so code order is id order), -1 when
    row i is unmapped. ``token_codes[i]`` indexes ``tokens``, the distinct input tokens."""

    instrument_codes: np.ndarray
    instruments: list[str]
    token_codes: np.ndarray
    tokens: list[str]


def _flat(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    return values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values


def map_rows(
    identifiers: pa.Table, tokens: pa.Array | pa.ChunkedArray, dates: pa.Array | pa.ChunkedArray
) -> MappedRows:
    """Vectorised ``resolve`` over aligned ``tokens`` and ``dates``.

    Same rules and same ambiguity ``ValueError`` as ``resolve``; the message lists up to
    ``MAX_LISTED`` ambiguous (symbol, date) keys. Work is one numpy pass per mapping interval over
    the rows of its token: no join, no per-row Python."""
    mappings = symbol_mappings(identifiers)
    encoded = pc.dictionary_encode(_flat(tokens))
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    days = _flat(dates).cast(pa.int32()).to_numpy(zero_copy_only=False)
    distinct = encoded.dictionary.to_pylist()
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(distinct) + 1))

    instruments = sorted({i for token in distinct for *_, i in mappings.get(token, ())})
    numbering = {instrument: k for k, instrument in enumerate(instruments)}
    found = np.full(len(codes), -1, np.int32)
    ambiguous = np.zeros(len(codes), bool)
    for code, token in enumerate(distinct):
        rows = order[bounds[code] : bounds[code + 1]]
        row_days = days[rows]
        for start, end, instrument in mappings.get(token, ()):
            k = numbering[instrument]
            cover = row_days >= _days(start)
            if end is not None:
                cover &= row_days < _days(end)
            hit = rows[cover]
            prior = found[hit]
            ambiguous[hit[(prior >= 0) & (prior != k)]] = True
            found[hit] = k
    if ambiguous.any():
        _raise_ambiguous(mappings, distinct, codes, days, np.flatnonzero(ambiguous))
    return MappedRows(found, instruments, codes, distinct)


_EPOCH = date(1970, 1, 1)


def _days(d: date) -> int:
    return (d - _EPOCH).days


def _raise_ambiguous(
    mappings: Mappings, tokens: list[str], codes: np.ndarray, days: np.ndarray, rows: np.ndarray
) -> None:
    keys = sorted({(tokens[codes[i]], _EPOCH + timedelta(days=int(days[i]))) for i in rows})
    listed = []
    for token, on in keys[:MAX_LISTED]:
        hits = {i for s, e, i in mappings[token] if s <= on and (e is None or on < e)}
        listed.append(f"symbol {token!r} on {on.isoformat()}: several instruments {sorted(hits)}")
    raise ValueError(
        f"ambiguous mapping for {len(keys)} (symbol, date) key(s), fix the mapping "
        f"upstream; first {len(listed)}: " + "; ".join(listed)
    )


def resolve(mappings: Mappings, token: str, on: date) -> str | None:
    """The instrument ``token`` maps to on ``on``; None when unmapped, ValueError if ambiguous."""
    found = {
        instrument
        for start, end, instrument in mappings.get(token, ())
        if start <= on and (end is None or on < end)
    }
    if len(found) > 1:
        raise ValueError(
            f"ambiguous mapping for symbol {token!r} on {on.isoformat()}: "
            f"several instruments {sorted(found)}; fix the mapping upstream"
        )
    return next(iter(found), None)
