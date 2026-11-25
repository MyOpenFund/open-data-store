"""Source registry: licence terms and access class per data source (market-data spec §13).

Sources are declared through the "open_data_store.sources" entry-point group; each entry point is a
callable returning a list of SourceInfo. By default a ``source_code`` that is not registered is
treated as open, so existing producers keep working; ``filter_sources(..., strict=True)`` denies it
instead (fail closed), and the market-data binding always reads in strict mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from open_data_store.contracts.base import ContractError

GROUP = "open_data_store.sources"
SOURCE_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,62}")
SYMBOL_SCHEME = "source_symbol"


@dataclass(frozen=True)
class SourceInfo:
    code: str
    licence: str
    terms_url: str
    attribution: str
    access: Literal["open", "restricted"]
    no_public_display: bool = False
    no_ml_training: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not SOURCE_CODE_PATTERN.fullmatch(self.code):
            raise ContractError(
                f"source code must match {SOURCE_CODE_PATTERN.pattern!r} (lowercase, no "
                f"surrounding spaces, at most 63 characters), got {self.code!r}"
            )
        if self.access not in ("open", "restricted"):
            raise ContractError(
                f"source {self.code!r}: access must be 'open' or 'restricted', got {self.access!r}"
            )


def discover_sources(extra: list[SourceInfo] | None = None) -> dict[str, SourceInfo]:
    """Return {code: SourceInfo} for every installed source.

    The same code declared twice with different content is a ContractError.
    """
    found: list[SourceInfo] = []
    for ep in entry_points(group=GROUP):
        found.extend(ep.load()())
    found.extend(extra or [])
    out: dict[str, SourceInfo] = {}
    for s in found:
        if s.code in out and out[s.code] != s:
            raise ContractError(f"source {s.code!r} declared twice with different definitions")
        out[s.code] = s
    return out


def restricted_codes(sources: dict[str, SourceInfo] | None = None) -> frozenset[str]:
    sources = discover_sources() if sources is None else sources
    return frozenset(c for c, s in sources.items() if s.access == "restricted")


def ml_forbidden_codes(sources: dict[str, SourceInfo] | None = None) -> frozenset[str]:
    """Codes whose rows a training pipeline must refuse (Massive §5(g))."""
    sources = discover_sources() if sources is None else sources
    return frozenset(c for c, s in sources.items() if s.no_ml_training)


def filter_sources(
    table: pa.Table,
    *,
    allowed_restricted: frozenset[str],
    restricted: frozenset[str],
    strict: bool = False,
    registered: frozenset[str] | None = None,
) -> pa.Table:
    """Drop rows whose source is denied: restricted and not in ``allowed_restricted``, or, with
    ``strict=True``, not in ``registered`` (the registry's codes; required in strict mode).

    The source of a row is its ``source_code``; a row with ``id_scheme = "source_symbol"`` (an
    identifier mapping) also names the source of its ``id_value`` prefix (``"<source>:"``), and is
    dropped when either source is denied. Fail closed: a null ``source_code``, or a
    ``source_symbol`` row whose ``id_value`` has no prefix, is dropped too (contracts make them
    well formed, so only malformed tables are affected). A table without a ``source_code`` column
    is returned unchanged.

    Filter before any column projection: a projection that drops ``source_code`` (or
    ``id_scheme``/``id_value``) also drops what the filter needs.
    """
    if strict and registered is None:
        raise ValueError("filter_sources(strict=True) needs the registered source codes")
    if "source_code" not in table.column_names:
        return table
    denied = pa.array(sorted(restricted - allowed_restricted), pa.string())
    kept = pa.array(sorted(registered or ()), pa.string())

    def ok(codes: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
        """True where the code is a source the caller may read; null codes are refused."""
        good = pc.invert(pc.is_in(codes, value_set=denied))
        if strict:
            good = pc.and_(good, pc.is_in(codes, value_set=kept))
        return pc.fill_null(pc.and_(good, pc.is_valid(codes)), False)

    keep = ok(table.column("source_code"))
    if {"id_scheme", "id_value"} <= set(table.column_names):
        prefix = pc.struct_field(
            pc.extract_regex(table.column("id_value"), r"^(?P<source>[^:]+):"), [0]
        )
        is_symbol = pc.fill_null(pc.equal(table.column("id_scheme"), SYMBOL_SCHEME), False)
        keep = pc.and_(keep, pc.or_(pc.invert(is_symbol), ok(prefix)))
    return table.filter(keep)
