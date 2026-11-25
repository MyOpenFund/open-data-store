"""Source registry: licence terms and access class per data source (market-data spec §13).

Sources are declared through the "open_data_store.sources" entry-point group; each entry point is a
callable returning a list of SourceInfo. A ``source_code`` that is not registered is treated as
open, so existing producers keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from open_data_store.contracts.base import ContractError

GROUP = "open_data_store.sources"


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
    table: pa.Table, allowed_restricted: frozenset[str], restricted: frozenset[str]
) -> pa.Table:
    """Drop rows whose source_code is restricted and not allowed.

    A table without a source_code column is returned unchanged. Fail closed: when the column
    exists, rows with a null source_code are dropped too (contracts make it non-null, so only
    malformed tables are affected).
    """
    if "source_code" not in table.column_names:
        return table
    col = table.column("source_code")
    denied = pa.array(sorted(restricted - allowed_restricted), pa.string())
    hit = pc.fill_null(pc.is_in(col, value_set=denied), False)
    return table.filter(pc.invert(pc.or_(hit, pc.is_null(col))))
