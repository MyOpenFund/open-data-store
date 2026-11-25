import pytest
from conftest import NOW, action_row, price_row, table

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
    from open_data_store.contracts.rules import ID_SCHEMES

    assert ID_SCHEMES == ("cik", "lei", "isin", "figi", "ticker", "cb_code", "national_id")
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
