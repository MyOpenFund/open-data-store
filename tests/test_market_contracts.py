import uuid
from datetime import date

import pyarrow as pa
import pytest
from conftest import NOW, action_row, price_row, table, ts

from open_data_store.contracts.registry import discover_contracts
from open_data_store.contracts.rules import BatchInvalid, conform, validate_contract
from open_data_store.contracts.standard import (
    ENTITY_OBSERVATIONS,
    FILING_STATEMENTS,
    FILINGS,
    FUNDAMENTALS,
)
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    FETCH_LOG,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    LINK_REASONS,
    MARKET_CONTRACTS,
    MARKET_ID_SCHEMES,
    PRICES,
    VENDOR_TOTAL_RETURN,
)

# Computed on the code before the market contracts were added.
FINGERPRINTS_BEFORE = {
    "entity_observations": "8b7260f697934ad13546c611d0ba71d33cd3161c3de4b8ef6e9d32f57b5465e3",
    "filing_statements": "78714daa856dcbd3fded9bd43b2d255fc433b8fbe6d9365f557c8a391fd61665",
    "filings": "82164e8519eb3affae2a6381832cecf2aa14fdf24005ad1db6649aaa6ef424bd",
    "fundamentals": "1a828cfc77c20239f858a9582d8ea6990991782f39d84cc3ea8ec1f4ee35b951",
}

# The six market contracts at 1.0.0 (fetch_log with a nullable requested range). A change here is
# a new MAJOR, or a MINOR the publisher accepts: never an edit of this dict alone.
MARKET_FINGERPRINTS = {
    "prices": "2ac2c813d515a3b3c3fdbba8ee6b5b700078acbb25fcadf6a8c9b67e740559c8",
    "corporate_actions": "1d5dbb8cdcb23e913d4756a7fda325eedd5c769c5073fba7b011f6cda0ad72f4",
    "vendor_total_return": "e0fd9d4f6c84553829d9f7f9896ff6f288cb850492e522f0ac0e04065856b7c0",
    "index_snapshots": "56a538f32d9b9eada0571ace47caf14eb8e46cc24678a25b31d0723239be7bb9",
    "instrument_identifiers": "034169f616fdee7bb42d8c9441b2ce8d6c3bbd2c97edf25aed694fb8850a721d",
    "fetch_log": "0cdbfee56fa30b40e6e6017ab836128cebf8eb1bff7f069e2d7edc585521445d",
}


def test_market_fingerprints_pinned():
    assert {c.name: c.fingerprint() for c in MARKET_CONTRACTS} == MARKET_FINGERPRINTS


def test_market_contracts_validate():
    assert len(MARKET_CONTRACTS) == 6
    for c in MARKET_CONTRACTS:
        validate_contract(c)


def test_common_shape():
    for c in MARKET_CONTRACTS:
        assert c.kind == "fact"
        assert c.version == "1.0.0"
        assert c.entity_column_pair is None
        assert c.owner == "open-data-store"
        assert c.licence == "per-source; see the source registry"
        assert c.partition_by == ("source_code",)
        assert not c.column("known_at_basis").nullable
        assert c.column("known_at_basis").allowed is not None


def test_existing_fingerprints_unchanged():
    got = {
        c.name: c.fingerprint()
        for c in (ENTITY_OBSERVATIONS, FILINGS, FUNDAMENTALS, FILING_STATEMENTS)
    }
    assert got == FINGERPRINTS_BEFORE


def test_discovered():
    names = {k[0] for k in discover_contracts()}
    assert {
        "prices",
        "corporate_actions",
        "vendor_total_return",
        "index_snapshots",
        "instrument_identifiers",
        "fetch_log",
    } <= names


def test_natural_keys():
    assert PRICES.natural_key == ("source_code", "source_symbol", "valid_date")
    assert CORPORATE_ACTIONS.natural_key == (
        "source_code",
        "source_symbol",
        "valid_date",
        "action_type",
        "action_seq",
    )
    assert VENDOR_TOTAL_RETURN.natural_key == ("source_code", "source_symbol", "valid_date")
    assert INDEX_SNAPSHOTS.natural_key == (
        "source_code",
        "index_code",
        "source_symbol",
        "valid_date",
    )
    assert INSTRUMENT_IDENTIFIERS.natural_key == (
        "source_code",
        "instrument_id",
        "id_scheme",
        "id_value",
        "valid_date",
    )
    assert FETCH_LOG.natural_key == ("source_code", "source_symbol", "fetch_id", "valid_date")


def test_id_schemes_untouched_and_own_vocabulary():
    assert INSTRUMENT_IDENTIFIERS.column("id_scheme").allowed == MARKET_ID_SCHEMES


def test_prices_accepts_a_raw_row_with_only_close():
    row = price_row(open=None, high=None, low=None, volume=None)
    assert conform(PRICES, table(PRICES, [row]), now=NOW).num_rows == 1


def test_prices_rejects_adjusted_basis():
    with pytest.raises(BatchInvalid, match="value_not_allowed:price_basis"):
        conform(PRICES, table(PRICES, [price_row(price_basis="adjusted")]), now=NOW)


def test_prices_require_close_and_currency():
    for col in ("close", "currency"):
        with pytest.raises(BatchInvalid, match=f"nulls_in_required:{col}"):
            conform(PRICES, table(PRICES, [price_row(**{col: None})]), now=NOW)


def test_prices_refuse_unknown_known_at_basis():
    with pytest.raises(BatchInvalid, match="value_not_allowed:known_at_basis"):
        conform(PRICES, table(PRICES, [price_row(known_at_basis="midnight")]), now=NOW)


def test_two_actions_same_day_need_distinct_seq():
    rows = [action_row(action_type="split", action_seq=0)] * 2
    with pytest.raises(BatchInvalid, match="duplicate_natural_key"):
        conform(CORPORATE_ACTIONS, table(CORPORATE_ACTIONS, rows), now=NOW)
    rows = [action_row(action_seq=0), action_row(action_seq=1)]
    assert conform(CORPORATE_ACTIONS, table(CORPORATE_ACTIONS, rows), now=NOW).num_rows == 2


def test_actions_refuse_unknown_type():
    with pytest.raises(BatchInvalid, match="value_not_allowed:action_type"):
        conform(
            CORPORATE_ACTIONS,
            table(CORPORATE_ACTIONS, [action_row(action_type="merger")]),
            now=NOW,
        )


def test_nullable_vocabulary_columns_accept_null_but_refuse_other_values():
    ok = action_row(amount_basis=None, per_share_basis=None)
    assert conform(CORPORATE_ACTIONS, table(CORPORATE_ACTIONS, [ok]), now=NOW).num_rows == 1
    with pytest.raises(BatchInvalid, match="value_not_allowed:amount_basis"):
        conform(
            CORPORATE_ACTIONS,
            table(CORPORATE_ACTIONS, [action_row(amount_basis="net_ish")]),
            now=NOW,
        )


_COMMON = dict(
    valid_date=date(2025, 6, 30),
    known_at=ts("2025-06-30T20:00:00"),
    known_at_kind="published",
    source_code="massive",
    source_doc_id=None,
)


def _rows():
    return {
        VENDOR_TOTAL_RETURN: dict(
            source_symbol="AAPL", tr_index=1234.5, known_at_basis="fetch_time", **_COMMON
        ),
        INDEX_SNAPSHOTS: dict(
            index_code="sp500",
            source_symbol="AAPL",
            snapshot_size=503,
            weight=None,
            is_member=True,
            known_at_basis="snapshot_date",
            **_COMMON,
        ),
        INSTRUMENT_IDENTIFIERS: dict(
            instrument_id="mof:" + uuid.uuid4().hex,
            id_scheme="source_symbol",
            id_value="massive:AAPL",
            valid_to=None,
            link_reason=None,
            known_at_basis="load_time",
            **_COMMON,
        ),
        FETCH_LOG: dict(
            source_symbol="AAPL",
            fetch_id="f-1",
            requested_from=date(2025, 1, 1),
            requested_to=date(2025, 6, 30),
            returned_from=date(2025, 1, 2),
            returned_to=date(2025, 6, 30),
            status="ok",
            row_count=124,
            archive_ref=None,
            known_at_basis="fetch_time",
            **_COMMON,
        ),
    }


@pytest.mark.parametrize(
    "contract", [VENDOR_TOTAL_RETURN, INDEX_SNAPSHOTS, INSTRUMENT_IDENTIFIERS, FETCH_LOG]
)
def test_valid_synthetic_row_conforms(contract):
    row = _rows()[contract]
    assert conform(contract, table(contract, [row]), now=NOW).num_rows == 1


def test_link_reason_is_a_closed_nullable_vocabulary():
    assert INSTRUMENT_IDENTIFIERS.column("link_reason").allowed == LINK_REASONS
    row = _rows()[INSTRUMENT_IDENTIFIERS]
    row["link_reason"] = "successor"
    assert conform(INSTRUMENT_IDENTIFIERS, table(INSTRUMENT_IDENTIFIERS, [row]), now=NOW)
    row["link_reason"] = "because"
    with pytest.raises(BatchInvalid, match="value_not_allowed:link_reason"):
        conform(INSTRUMENT_IDENTIFIERS, table(INSTRUMENT_IDENTIFIERS, [row]), now=NOW)


def test_column_types_are_pinned():
    assert PRICES.column("close").type == pa.float64()
    assert PRICES.column("volume").type == pa.int64()
    assert CORPORATE_ACTIONS.column("action_seq").type == pa.int16()


def test_fetch_log_requested_range_is_nullable():
    # A fetch of "everything the source has" requests no range.
    for col in ("requested_from", "requested_to"):
        assert FETCH_LOG.column(col).nullable
    row = dict(_rows()[FETCH_LOG], requested_from=None, requested_to=None)
    assert conform(FETCH_LOG, table(FETCH_LOG, [row]), now=NOW).num_rows == 1
