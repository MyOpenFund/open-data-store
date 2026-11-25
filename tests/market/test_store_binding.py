import dataclasses

import pytest
from conftest import NOW, identifier_row, price_row, snapshot_row, ts
from test_publisher import land

from open_data_store import sources
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    PRICES,
)
from open_data_store.market.store import MarketInputs, load_market_inputs
from open_data_store.publisher import approve_contract, publish_pending
from open_data_store.reader import Store
from open_data_store.sources import SourceInfo

LATE = ts("2026-06-01T00:00:00")


@pytest.fixture
def restricted_registry(monkeypatch):
    info = SourceInfo("vendr", "vendor terms", "https://example.test/t", "Vendor", "restricted")
    real = sources.discover_sources
    monkeypatch.setattr(
        sources, "discover_sources", lambda extra=None: real([*(extra or []), info])
    )


@pytest.fixture
def store(approved, restricted_registry):
    for c in (PRICES, INDEX_SNAPSHOTS, INSTRUMENT_IDENTIFIERS):
        approve_contract(approved, c, "test")
    land(
        approved,
        PRICES,
        [price_row(), price_row(source_code="vendr", source_symbol="VND", close=7.0)],
    )
    land(approved, INSTRUMENT_IDENTIFIERS, [identifier_row()], run_id="r2")
    land(approved, INDEX_SNAPSHOTS, [snapshot_row()], run_id="r3")
    assert publish_pending(approved, now=NOW).exit_code == 0
    s = Store.open(approved)
    yield s
    s.close()


def test_binding_reads_published_rows_and_filters_restricted(store):
    got = load_market_inputs(store, as_of=LATE)
    assert isinstance(got, MarketInputs)
    assert got.prices.column("source_code").to_pylist() == ["massive"]
    allowed = load_market_inputs(store, as_of=LATE, allowed_restricted=frozenset({"vendr"}))
    assert sorted(allowed.prices.column("source_code").to_pylist()) == ["massive", "vendr"]
    assert got.snapshots.num_rows == 1
    assert got.identifiers.column("id_value").to_pylist() == ["srcx:TOKA"]


def test_facts_respect_as_of_and_identifiers_are_latest(store):
    before = load_market_inputs(store, as_of=ts("2025-06-01T00:00:00"))
    assert before.prices.num_rows == 0
    assert before.identifiers.num_rows == 1  # mapping read latest, whatever as_of


def test_unpublished_dataset_gives_an_empty_table_with_the_contract_schema(store):
    got = load_market_inputs(store, as_of=LATE)
    assert got.actions.num_rows == 0
    assert got.actions.schema.equals(CORPORATE_ACTIONS.schema)
    assert got.fetch_log.num_rows == 0


def test_publisher_columns_are_dropped(store):
    got = load_market_inputs(store, as_of=LATE)
    for t in (got.prices, got.snapshots, got.identifiers):
        assert "entity_id" not in t.column_names
        assert "batch_id" not in t.column_names
    for t, c in ((got.prices, PRICES), (got.snapshots, INDEX_SNAPSHOTS)):
        assert t.schema.names == c.schema.names
        assert t.schema.types == c.schema.types


def test_inputs_are_frozen(store):
    got = load_market_inputs(store, as_of=LATE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.prices = got.actions  # type: ignore[misc]
