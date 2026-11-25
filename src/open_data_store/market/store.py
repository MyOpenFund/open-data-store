"""Store binding: the market inputs the recipes need, read from a store at one date (spec §7.4).

Facts (prices, actions, snapshots, fetch log) are read as-of; identifier mappings are read latest,
by design. Each contract is read at its own major (``<name>_v<major>``). The licence gate runs in
strict mode (every market contract's licence is "per-source; see the source registry"): rows of a
restricted source are dropped unless the caller allows it, rows of a source missing from the
registry are dropped always, and identifier mappings naming such a source in their ``id_value``
are dropped too. A dataset that is not published on the store yields an empty table with its
contract schema, so the binding works on a store that holds only part of the market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from open_data_store import sources
from open_data_store.contracts.base import Contract
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    FETCH_LOG,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    PRICES,
)
from open_data_store.reader.api import Store, UnknownDataset

__all__ = ["MarketInputs", "load_market_inputs"]

_PUBLISHER_COLUMNS = ("entity_id", "batch_id")


@dataclass(frozen=True)
class MarketInputs:
    prices: pa.Table
    actions: pa.Table
    snapshots: pa.Table
    identifiers: pa.Table
    fetch_log: pa.Table


def _read(
    store: Store,
    contract: Contract,
    as_of: datetime,
    allowed_restricted: frozenset[str],
    registry: dict[str, sources.SourceInfo],
) -> pa.Table:
    dataset = contract.table_name
    try:
        published = store.is_published(dataset)
    except UnknownDataset:
        published = False
    if not published:
        return contract.schema.empty_table()
    table = store.read_asof(dataset, as_of)
    # The licence gate runs before any projection: it needs source_code, id_scheme, id_value.
    table = sources.filter_sources(
        table,
        allowed_restricted=allowed_restricted,
        restricted=sources.restricted_codes(registry),
        strict=True,
        registered=frozenset(registry),
    )
    table = table.drop_columns([c for c in _PUBLISHER_COLUMNS if c in table.column_names])
    return table.select(contract.schema.names).cast(contract.schema)


def load_market_inputs(
    store: Store,
    *,
    as_of: datetime,
    allowed_restricted: frozenset[str] = frozenset(),
    identifiers_as_of: datetime | None = None,
) -> MarketInputs:
    """Read the market inputs of ``store`` as known at ``as_of`` (tz-aware).

    Prices, actions, snapshots and the fetch log are read as-of; identifiers are read latest:
    at ``identifiers_as_of`` (tz-aware), default now (UTC); pass it to reproduce a read. Each
    contract is read at its own major. Strict licence gate: rows of restricted sources are dropped
    unless their code is in ``allowed_restricted``; rows of sources missing from the registry are
    always dropped; a ``source_symbol`` identifier row is dropped when the source of its
    ``id_value`` prefix is denied.
    ``entity_id`` and ``batch_id`` are dropped, so every table follows its contract schema
    (empty and non-empty tables have the same schema; a dataset approved but not
    yet published, or not approved, gives an empty table).

    Only the latest vintage knowable at ``as_of`` is kept per key, so call the recipes with the same
    ``as_of`` as this load."""
    for name, value in (("as_of", as_of), ("identifiers_as_of", identifiers_as_of)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{name} must be a timezone-aware datetime")
    registry = sources.discover_sources()
    now = identifiers_as_of or datetime.now(UTC)
    return MarketInputs(
        prices=_read(store, PRICES, as_of, allowed_restricted, registry),
        actions=_read(store, CORPORATE_ACTIONS, as_of, allowed_restricted, registry),
        snapshots=_read(store, INDEX_SNAPSHOTS, as_of, allowed_restricted, registry),
        identifiers=_read(store, INSTRUMENT_IDENTIFIERS, now, allowed_restricted, registry),
        fetch_log=_read(store, FETCH_LOG, as_of, allowed_restricted, registry),
    )
