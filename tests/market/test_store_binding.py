import dataclasses
from datetime import datetime

import pytest
from conftest import NOW, identifier_row, price_row, snapshot_row, ts
from test_publisher import land

from open_data_store import sources
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    FETCH_LOG,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    MARKET_CONTRACTS,
    PRICES,
)
from open_data_store.market.store import MarketInputs, load_market_inputs
from open_data_store.publisher import approve_contract, publish_pending
from open_data_store.reader import Store
from open_data_store.sources import SourceInfo

LATE = ts("2026-06-01T00:00:00")


def _open(code):
    return SourceInfo(code, "open terms", "https://example.test/o", code, "open")


# Market data is read in strict mode: every source of the fixtures must be registered.
OPEN_SOURCES = [_open(c) for c in ("massive", "srcx", "mapper")]


@pytest.fixture
def restricted_registry(monkeypatch):
    info = SourceInfo("vendr", "vendor terms", "https://example.test/t", "Vendor", "restricted")
    real = sources.discover_sources
    monkeypatch.setattr(
        sources,
        "discover_sources",
        lambda extra=None: real([*(extra or []), *OPEN_SOURCES, info]),
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


def test_approved_but_unpublished_datasets_give_empty_tables(approved):
    for c in MARKET_CONTRACTS:
        approve_contract(approved, c, "test")
    with Store.open(approved) as s:
        got = load_market_inputs(s, as_of=LATE)
    for t, c in (
        (got.prices, PRICES),
        (got.actions, CORPORATE_ACTIONS),
        (got.snapshots, INDEX_SNAPSHOTS),
        (got.identifiers, INSTRUMENT_IDENTIFIERS),
        (got.fetch_log, FETCH_LOG),
    ):
        assert t.num_rows == 0
        assert t.schema.equals(c.schema)


def test_empty_and_non_empty_tables_share_the_contract_schema(store):
    got = load_market_inputs(store, as_of=LATE)
    assert got.prices.num_rows == 1
    assert got.prices.schema.equals(PRICES.schema)
    assert got.snapshots.schema.equals(INDEX_SNAPSHOTS.schema)
    assert load_market_inputs(store, as_of=ts("2025-06-01T00:00:00")).prices.schema.equals(
        PRICES.schema
    )


def test_naive_as_of_is_rejected(store):
    with pytest.raises(ValueError, match="as_of"):
        load_market_inputs(store, as_of=datetime(2026, 6, 1))
    with pytest.raises(ValueError, match="identifiers_as_of"):
        load_market_inputs(store, as_of=LATE, identifiers_as_of=datetime(2026, 6, 1))


def test_identifiers_as_of_reproduces_a_read(store):
    assert load_market_inputs(store, as_of=LATE, identifiers_as_of=NOW).identifiers.num_rows == 1
    early = ts("2020-01-01T00:00:00")
    assert load_market_inputs(store, as_of=LATE, identifiers_as_of=early).identifiers.num_rows == 0


def test_restricted_sources_are_filtered_in_snapshots_and_identifiers(
    approved, restricted_registry
):
    for c in (INDEX_SNAPSHOTS, INSTRUMENT_IDENTIFIERS):
        approve_contract(approved, c, "test")
    land(approved, INDEX_SNAPSHOTS, [snapshot_row(), snapshot_row(source_code="vendr")])
    land(
        approved,
        INSTRUMENT_IDENTIFIERS,
        [identifier_row(), identifier_row(source_code="vendr", id_value="vendr:TOKA")],
        run_id="r2",
    )
    assert publish_pending(approved, now=NOW).exit_code == 0
    with Store.open(approved) as s:
        got = load_market_inputs(s, as_of=LATE)
        allowed = load_market_inputs(s, as_of=LATE, allowed_restricted=frozenset({"vendr"}))
    assert got.snapshots.column("source_code").to_pylist() == ["srcx"]
    assert got.identifiers.column("source_code").to_pylist() == ["mapper"]
    assert allowed.snapshots.num_rows == 2
    assert allowed.identifiers.num_rows == 2


def test_unregistered_sources_are_denied(approved, restricted_registry):
    approve_contract(approved, PRICES, "test")
    land(approved, PRICES, [price_row(), price_row(source_code="ghost", source_symbol="GHO")])
    assert publish_pending(approved, now=NOW).exit_code == 0
    with Store.open(approved) as s:
        got = load_market_inputs(s, as_of=LATE)
        allowed = load_market_inputs(s, as_of=LATE, allowed_restricted=frozenset({"ghost"}))
    assert got.prices.column("source_code").to_pylist() == ["massive"]
    assert allowed.prices.column("source_code").to_pylist() == ["massive"]


def test_identifier_rows_naming_a_restricted_source_are_filtered(approved, restricted_registry):
    approve_contract(approved, INSTRUMENT_IDENTIFIERS, "test")
    # Both rows are stamped by the open mapper; the second maps a symbol of the restricted vendor.
    land(
        approved,
        INSTRUMENT_IDENTIFIERS,
        [identifier_row(), identifier_row(id_value="vendr:VND")],
    )
    assert publish_pending(approved, now=NOW).exit_code == 0
    with Store.open(approved) as s:
        got = load_market_inputs(s, as_of=LATE)
        allowed = load_market_inputs(s, as_of=LATE, allowed_restricted=frozenset({"vendr"}))
    assert got.identifiers.column("id_value").to_pylist() == ["srcx:TOKA"]
    assert sorted(allowed.identifiers.column("id_value").to_pylist()) == ["srcx:TOKA", "vendr:VND"]


def test_each_contract_is_read_at_its_own_major(approved, restricted_registry):
    approve_contract(approved, PRICES, "test")
    land(approved, PRICES, [price_row()])
    assert publish_pending(approved, now=NOW).exit_code == 0
    # An approved, unpublished later major must not hide the published v1 rows.
    approve_contract(approved, dataclasses.replace(PRICES, version="2.0.0"), "test")
    with Store.open(approved) as s:
        got = load_market_inputs(s, as_of=LATE)
    assert got.prices.column("source_code").to_pylist() == ["massive"]
