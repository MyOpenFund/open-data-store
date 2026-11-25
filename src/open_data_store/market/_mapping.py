"""Symbol → instrument resolution shared by the market recipes (private).

A provider series (source_code, source_symbol) maps to an instrument through
``instrument_identifiers`` rows with ``id_scheme = "source_symbol"`` and
``id_value = "<source_code>:<source_symbol>"``, read latest (the per-input policy for identifier
mappings), applied on a date D only when ``[valid_date, valid_to)`` covers D. Covering rows that
name different instruments raise ``ValueError``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

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
