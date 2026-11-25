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
        table = store.read_asof(contract.name, as_of)
    except UnknownDataset:
        return contract.schema.empty_table()
    table = table.drop_columns([c for c in _PUBLISHER_COLUMNS if c in table.column_names])
    return filter_sources(table, allowed_restricted, restricted)


def load_market_inputs(
    store: Store, *, as_of: datetime, allowed_restricted: frozenset[str] = frozenset()
) -> MarketInputs:
    """Read the market inputs of ``store`` as known at ``as_of`` (tz-aware).

    Prices, actions, snapshots and the fetch log are read as-of; identifiers are read latest
    (``now``, UTC). Rows of restricted sources are dropped unless their code is in
    ``allowed_restricted``. ``entity_id`` and ``batch_id`` are dropped, so every table follows its
    contract schema."""
    restricted = restricted_codes()
    now = datetime.now(UTC)
    return MarketInputs(
        prices=_read(store, PRICES, as_of, allowed_restricted, restricted),
        actions=_read(store, CORPORATE_ACTIONS, as_of, allowed_restricted, restricted),
        snapshots=_read(store, INDEX_SNAPSHOTS, as_of, allowed_restricted, restricted),
        identifiers=_read(store, INSTRUMENT_IDENTIFIERS, now, allowed_restricted, restricted),
        fetch_log=_read(store, FETCH_LOG, as_of, allowed_restricted, restricted),
    )
