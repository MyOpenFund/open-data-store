"""Store binding: the market inputs the recipes need, read from a store at one date (spec §7.4).

Facts (prices, actions, snapshots, fetch log) are read as-of; identifier mappings are read latest,
by design. Restricted sources are dropped unless the caller allows them. A dataset that is not
published on the store yields an empty table with its contract schema, so the binding works on a
store that holds only part of the market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from open_data_store.contracts.base import Contract
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    FETCH_LOG,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    PRICES,
)
from open_data_store.reader.api import Store, UnknownDataset
from open_data_store.sources import filter_sources, restricted_codes

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
    restricted: frozenset[str],
) -> pa.Table:
    try:
        published = store.is_published(contract.name)
    except UnknownDataset:
        published = False
    if not published:
        return contract.schema.empty_table()
    table = store.read_asof(contract.name, as_of)
    table = table.drop_columns([c for c in _PUBLISHER_COLUMNS if c in table.column_names])
    table = table.select(contract.schema.names).cast(contract.schema)
    return filter_sources(table, allowed_restricted, restricted)


def load_market_inputs(
    store: Store,
    *,
    as_of: datetime,
    allowed_restricted: frozenset[str] = frozenset(),
    identifiers_as_of: datetime | None = None,
) -> MarketInputs:
    """Read the market inputs of ``store`` as known at ``as_of`` (tz-aware).

    Prices, actions, snapshots and the fetch log are read as-of; identifiers are read latest:
    at ``identifiers_as_of`` (tz-aware), default now (UTC); pass it to reproduce a read. Rows of
    restricted sources are dropped unless their code is in ``allowed_restricted``.
    ``entity_id`` and ``batch_id`` are dropped, so every table follows its contract schema
    (empty and non-empty tables have the same schema; a dataset approved but not
    yet published, or not approved, gives an empty table).

    Only the latest vintage knowable at ``as_of`` is kept per key, so call the recipes with the same
    ``as_of`` as this load."""
    for name, value in (("as_of", as_of), ("identifiers_as_of", identifiers_as_of)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{name} must be a timezone-aware datetime")
    restricted = restricted_codes()
    now = identifiers_as_of or datetime.now(UTC)
    return MarketInputs(
        prices=_read(store, PRICES, as_of, allowed_restricted, restricted),
        actions=_read(store, CORPORATE_ACTIONS, as_of, allowed_restricted, restricted),
        snapshots=_read(store, INDEX_SNAPSHOTS, as_of, allowed_restricted, restricted),
        identifiers=_read(store, INSTRUMENT_IDENTIFIERS, now, allowed_restricted, restricted),
        fetch_log=_read(store, FETCH_LOG, as_of, allowed_restricted, restricted),
    )
