from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest
from conftest import action_row, identifier_row, price_row, table

from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    INSTRUMENT_IDENTIFIERS,
    PRICES,
)
from open_data_store.market.asof import sessions
from open_data_store.market.returns import resolve_actions, resolve_prices, total_return

M1 = "mof:" + "1" * 32
M2 = "mof:" + "2" * 32
AS_OF = datetime(2026, 11, 1, 12, tzinfo=UTC)


def _at(d: date, hour: int = 21) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


def _prices(*rows):
    """rows: (valid_date, close[, overrides]). Source srcx, symbol TOKA, currency USD."""
    out = []
    for r in rows:
        row = dict(
            valid_date=r[0],
            close=r[1],
            known_at=_at(r[0]),
            source_code="srcx",
            source_symbol="TOKA",
        )
        row.update(r[2] if len(r) > 2 else {})
        out.append(price_row(**row))
    return table(PRICES, out)


def _actions(*rows):
    """rows: dicts of overrides; defaults to a srcx/TOKA cash dividend known before its ex-date."""
    out = []
    for over in rows:
        row = dict(
            source_code="srcx",
            source_symbol="TOKA",
            action_type="cash_dividend_regular",
            source_action_type="dividend",
            amount=None,
            currency=None,
            ratio_old=None,
            ratio_new=None,
            known_at=datetime(1999, 1, 1, tzinfo=UTC),
        )
        row.update(over)
        out.append(action_row(**row))
    return table(CORPORATE_ACTIONS, out)


def _cash(ex: date, amount: float, currency: str = "USD", **over):
    return dict(valid_date=ex, amount=amount, currency=currency, amount_basis="gross", **over)


def _split(ex: date, old: float, new: float, **over):
    return dict(
        valid_date=ex, action_type="split", source_action_type="split", ratio_old=old,
        ratio_new=new, **over,
    )  # fmt: skip


def _ids(*maps):
    """maps: (id_value, instrument_id[, overrides])."""
    rows = []
    for m in maps:
        row = dict(id_value=m[0], instrument_id=m[1])
        row.update(m[2] if len(m) > 2 else {})
        rows.append(identifier_row(**row))
    return table(INSTRUMENT_IDENTIFIERS, rows)


IDS = _ids(("srcx:TOKA", M1))


def _sessions(start: date, end: date) -> pa.Table:
    return sessions("XNYS", start, end)


def _tr(prices, actions=None, *, ids=IDS, start=None, end=None):
    rp = resolve_prices(prices, ids, as_of=AS_OF, precedence=())
    actions = actions if actions is not None else _actions()
    ra = resolve_actions(actions, ids, as_of=AS_OF, precedence=())
    dates = prices["valid_date"].to_pylist()
    s = _sessions(start or min(dates), end or max(dates))
    return total_return(rp, ra, s)


def _rows(out: pa.Table):
    return [
        (r["session_date"], pytest.approx(r["tr_index"], abs=1e-12), r["flags"])
        for r in out.to_pylist()
    ]


# --- brief tests -------------------------------------------------------------------------------


def test_dividend_reinvested_at_ex_close():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 99.0))
    out = _tr(prices, _actions(_cash(date(2020, 3, 3), 1.0)))
    assert _rows(out) == [(date(2020, 3, 2), 1.0, None), (date(2020, 3, 3), 1.0, None)]


def test_split_two_for_one():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 50.0))
    out = _tr(prices, _actions(_split(date(2020, 3, 3), 1.0, 2.0)))
    assert _rows(out) == [(date(2020, 3, 2), 1.0, None), (date(2020, 3, 3), 1.0, None)]


def test_split_and_dividend_same_day_new_share_basis():
    # 2:1 split and a dividend of 0.5 per new share: (49.5 + 0.5) / (100 / 2) - 1 = 0.
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 49.5))
    actions = _actions(
        _split(date(2020, 3, 3), 1.0, 2.0),
        _cash(date(2020, 3, 3), 0.5, per_share_basis="new_share"),
    )
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_ex_date_on_holiday_moves_to_next_session():
    # 2020-07-03 is an NYSE holiday: the dividend lands on 2020-07-06.
    prices = _prices((date(2020, 7, 1), 100.0), (date(2020, 7, 2), 100.0), (date(2020, 7, 6), 98.0))
    out = _tr(prices, _actions(_cash(date(2020, 7, 3), 2.0)))
    assert _rows(out) == [
        (date(2020, 7, 1), 1.0, None),
        (date(2020, 7, 2), 1.0, None),
        (date(2020, 7, 6), 1.0, None),
    ]


def test_gbp_dividend_on_gbp_pence_price_flagged():
    prices = _prices(
        (date(2020, 3, 2), 1000.0, dict(currency="GBp")),
        (date(2020, 3, 3), 990.0, dict(currency="GBp")),
    )
    out = _tr(prices, _actions(_cash(date(2020, 3, 3), 0.10, currency="GBP")))
    # Not reinvested, never converted: the raw price return stands.
    assert _rows(out)[-1] == (date(2020, 3, 3), 0.99, "currency_mismatch")


def test_precedence_picks_first_source():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0, dict(source_code="srca"))),
            _prices((date(2020, 3, 2), 101.0, dict(source_code="srcb"))),
        ]
    )
    ids = _ids(("srca:TOKA", M1), ("srcb:TOKA", M1))
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=("srcb", "srca"))
    assert out.column_names == [
        "instrument_id",
        "session_date",
        "close",
        "volume",
        "currency",
        "source_code",
        "source_symbol",
    ]
    assert out.to_pylist() == [
        dict(
            instrument_id=M1,
            session_date=date(2020, 3, 2),
            close=101.0,
            volume=1_000_000,
            currency="USD",
            source_code="srcb",
            source_symbol="TOKA",
        )
    ]
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=("srca", "srcb"))
    assert out.column("source_code").to_pylist() == ["srca"]


def test_forward_index_not_rewritten_by_future_action():
    prices = _prices(
        (date(2020, 3, 2), 100.0),
        (date(2020, 3, 3), 99.0),
        (date(2026, 3, 2), 120.0),
        (date(2026, 3, 3), 60.0),
    )
    early = _tr(prices, _actions(_cash(date(2020, 3, 3), 1.0)))
    later = _tr(prices, _actions(_cash(date(2020, 3, 3), 1.0), _split(date(2026, 3, 3), 1.0, 2.0)))
    cut = date(2020, 12, 31)
    keep = [r for r in early.to_pylist() if r["session_date"] <= cut]
    assert keep == [r for r in later.to_pylist() if r["session_date"] <= cut]
    # 2020: dividend reinvested (1.0); 2026-03-02: 120 / 99; the split day is flat only with it.
    assert later.column("tr_index").to_pylist()[-1] == pytest.approx(120 / 99)
    assert early.column("tr_index").to_pylist()[-1] == pytest.approx(120 / 99 * 0.5)


# --- ruling tests ------------------------------------------------------------------------------


def test_missing_price_session_flags_gap():
    # 2020-07-07 is a session with no price: the 07-08 return is against the 07-06 close.
    prices = _prices((date(2020, 7, 6), 100.0), (date(2020, 7, 8), 110.0))
    out = _tr(prices)
    assert _rows(out) == [(date(2020, 7, 6), 1.0, None), (date(2020, 7, 8), 1.1, "gap")]


def test_action_on_session_without_price_moves_to_next_price_session():
    prices = _prices((date(2020, 7, 6), 100.0), (date(2020, 7, 8), 98.0))
    out = _tr(prices, _actions(_cash(date(2020, 7, 7), 2.0)))
    assert _rows(out)[-1] == (date(2020, 7, 8), 1.0, "gap")


def test_action_after_last_price_ignored():
    prices = _prices((date(2020, 7, 6), 100.0), (date(2020, 7, 7), 100.0))
    out = _tr(prices, _actions(_cash(date(2020, 7, 8), 2.0)))
    assert _rows(out) == [(date(2020, 7, 6), 1.0, None), (date(2020, 7, 7), 1.0, None)]


def test_base_is_first_price_session_per_instrument():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 110.0)),
            _prices(
                (date(2020, 3, 3), 50.0, dict(source_symbol="TOKB")),
                (date(2020, 3, 4), 55.0, dict(source_symbol="TOKB")),
            ),
        ]
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    out = _tr(prices, ids=ids)
    assert [(r["instrument_id"], r["session_date"]) for r in out.to_pylist()] == [
        (M1, date(2020, 3, 2)),
        (M1, date(2020, 3, 3)),
        (M2, date(2020, 3, 3)),
        (M2, date(2020, 3, 4)),
    ]
    assert out.column("tr_index").to_pylist() == pytest.approx([1.0, 1.1, 1.0, 1.1])
    assert out.schema.field("tr_index").type == pa.float64()


def test_old_share_dividend_with_same_day_split_converted():
    # 2:1 split and 1.0 per OLD share = 0.5 per new share: (49.5 + 0.5) / 50 - 1 = 0.
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 49.5))
    actions = _actions(
        _split(date(2020, 3, 3), 1.0, 2.0),
        _cash(date(2020, 3, 3), 1.0, per_share_basis="old_share"),
    )
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_null_basis_dividend_with_same_day_split_used_as_is():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 49.5))
    actions = _actions(
        _split(date(2020, 3, 3), 1.0, 2.0), _cash(date(2020, 3, 3), 0.5, per_share_basis=None)
    )
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_stock_dividend_uses_ratio():
    # 10 % stock dividend (10 old : 11 new): 100 -> 90.909...
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 100.0 / 1.1))
    actions = _actions(dict(_split(date(2020, 3, 3), 10.0, 11.0), action_type="stock_dividend"))
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_return_of_capital_reinvested():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 95.0))
    actions = _actions(dict(_cash(date(2020, 3, 3), 5.0), action_type="return_of_capital"))
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_special_dividend_reinvested():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 95.0))
    actions = _actions(
        _cash(date(2020, 3, 3), 1.0),
        dict(_cash(date(2020, 3, 3), 4.0, action_seq=0), action_type="cash_dividend_special"),
    )
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_spin_off_without_value_flagged_not_applied():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 80.0))
    actions = _actions(
        dict(valid_date=date(2020, 3, 3), action_type="spin_off", spinoff_symbol="TOKS")
    )
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 0.8, "spin_off_unvalued")


def test_spin_off_with_value_reinvested():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 80.0))
    actions = _actions(
        dict(_cash(date(2020, 3, 3), 20.0), action_type="spin_off", spinoff_symbol="TOKS")
    )
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)


def test_rights_flagged_not_applied():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 97.0))
    actions = _actions(dict(_cash(date(2020, 3, 3), 3.0), action_type="rights"))
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 0.97, "rights_unvalued")


def test_currency_change_zero_return_never_converted():
    prices = _prices(
        (date(2020, 3, 2), 100.0, dict(currency="USD")),
        (date(2020, 3, 3), 9000.0, dict(currency="GBp")),
        (date(2020, 3, 4), 9900.0, dict(currency="GBp")),
    )
    assert _rows(_tr(prices)) == [
        (date(2020, 3, 2), 1.0, None),
        (date(2020, 3, 3), 1.0, "currency_change"),
        (date(2020, 3, 4), 1.1, None),
    ]


def test_flags_sorted_and_comma_joined():
    prices = _prices((date(2020, 7, 6), 100.0), (date(2020, 7, 8), 90.0))
    actions = _actions(
        _cash(date(2020, 7, 8), 1.0, currency="EUR"),
        dict(valid_date=date(2020, 7, 8), action_type="rights"),
    )
    assert _rows(_tr(prices, actions))[-1] == (
        date(2020, 7, 8),
        0.9,
        "currency_mismatch,gap,rights_unvalued",
    )


def test_empty_precedence_conflict_raises_naming_sources():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0, dict(source_code="srca"))),
            _prices((date(2020, 3, 2), 101.0, dict(source_code="srcb"))),
        ]
    )
    ids = _ids(("srca:TOKA", M1), ("srcb:TOKA", M1))
    with pytest.raises(ValueError, match=r"srca.*srcb"):
        resolve_prices(prices, ids, as_of=AS_OF, precedence=())
    actions = _actions(
        _cash(date(2020, 3, 2), 1.0, source_code="srca"),
        _cash(date(2020, 3, 2), 1.0, source_code="srcb"),
    )
    with pytest.raises(ValueError, match=r"srca.*srcb"):
        resolve_actions(actions, ids, as_of=AS_OF, precedence=())


def test_empty_precedence_one_source_per_session_is_fine():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0, dict(source_code="srca"))),
            _prices((date(2020, 3, 3), 101.0, dict(source_code="srcb"))),
        ]
    )
    ids = _ids(("srca:TOKA", M1), ("srcb:TOKA", M1))
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=())
    assert out.column("source_code").to_pylist() == ["srca", "srcb"]


def test_unlisted_source_ignored():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0, dict(source_code="srca"))),
            _prices((date(2020, 3, 3), 101.0, dict(source_code="srcb"))),
        ]
    )
    ids = _ids(("srca:TOKA", M1), ("srcb:TOKA", M1))
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=("srca",))
    assert out.column("session_date").to_pylist() == [date(2020, 3, 2)]


def test_unmapped_rows_dropped():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0)),
            _prices((date(2020, 3, 2), 50.0, dict(source_symbol="TOKZ"))),
            # Mapping starts on 2020-03-03: the 03-02 row of TOKB is unmapped.
            _prices(
                (date(2020, 3, 2), 70.0, dict(source_symbol="TOKB")),
                (date(2020, 3, 3), 71.0, dict(source_symbol="TOKB")),
            ),
        ]
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2, dict(valid_date=date(2020, 3, 3))))
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=())
    assert [(r["instrument_id"], r["session_date"]) for r in out.to_pylist()] == [
        (M1, date(2020, 3, 2)),
        (M2, date(2020, 3, 3)),
    ]
    actions = _actions(_cash(date(2020, 3, 2), 1.0, source_symbol="TOKZ"))
    assert resolve_actions(actions, ids, as_of=AS_OF, precedence=()).num_rows == 0


def test_recycled_symbol_maps_by_date():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 50.0))
    ids = _ids(
        ("srcx:TOKA", M1, dict(valid_to=date(2020, 3, 3))),
        ("srcx:TOKA", M2, dict(valid_date=date(2020, 3, 3))),
    )
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=())
    assert [(r["instrument_id"], r["close"]) for r in out.to_pylist()] == [
        (M1, 100.0),
        (M2, 50.0),
    ]


def test_ambiguous_mapping_raises():
    prices = _prices((date(2020, 3, 2), 100.0))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKA", M2))
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_prices(prices, ids, as_of=AS_OF, precedence=())


def test_resolve_reads_as_of():
    prices = _prices(
        (date(2020, 3, 2), 100.0),
        # Corrected on 2020-03-10.
        (date(2020, 3, 2), 100.5, dict(known_at=_at(date(2020, 3, 10)))),
        (date(2020, 3, 3), 101.0),
    )
    out = resolve_prices(prices, IDS, as_of=_at(date(2020, 3, 2), 22), precedence=())
    assert out.column("close").to_pylist() == [100.0]
    out = resolve_prices(prices, IDS, as_of=_at(date(2020, 3, 11)), precedence=())
    assert out.column("close").to_pylist() == [100.5, 101.0]
    # An action announced after as_of is invisible.
    actions = _actions(_cash(date(2020, 3, 3), 1.0, known_at=_at(date(2020, 3, 3), 1)))
    assert resolve_actions(actions, IDS, as_of=_at(date(2020, 3, 2)), precedence=()).num_rows == 0
    out = resolve_actions(actions, IDS, as_of=_at(date(2020, 3, 3)), precedence=())
    assert out.to_pylist()[0]["instrument_id"] == M1
    assert out.to_pylist()[0]["ex_date"] == date(2020, 3, 3)


def test_resolve_actions_keeps_every_action_of_the_chosen_source():
    actions = _actions(
        _split(date(2020, 3, 3), 1.0, 2.0, source_code="srca"),
        _cash(date(2020, 3, 3), 0.5, source_code="srca"),
        _cash(date(2020, 3, 3), 9.0, source_code="srcb"),
    )
    ids = _ids(("srca:TOKA", M1), ("srcb:TOKA", M1))
    out = resolve_actions(actions, ids, as_of=AS_OF, precedence=("srca", "srcb"))
    assert sorted(out.column("action_type").to_pylist()) == ["cash_dividend_regular", "split"]
    assert set(out.column("source_code").to_pylist()) == {"srca"}


def test_split_without_ratio_raises():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 50.0))
    actions = _actions(dict(valid_date=date(2020, 3, 3), action_type="split"))
    with pytest.raises(ValueError, match="ratio"):
        _tr(prices, actions)


def test_cash_without_amount_raises():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 50.0))
    actions = _actions(dict(valid_date=date(2020, 3, 3)))
    with pytest.raises(ValueError, match="amount"):
        _tr(prices, actions)


def test_bad_parameters():
    prices = _prices((date(2020, 3, 2), 100.0))
    with pytest.raises(TypeError):
        resolve_prices(prices, IDS, as_of=AS_OF, precedence="srcx")
    with pytest.raises(ValueError):
        resolve_prices(prices, IDS, as_of=datetime(2020, 3, 3), precedence=())
    rp = resolve_prices(prices, IDS, as_of=AS_OF, precedence=())
    ra = resolve_actions(_actions(), IDS, as_of=AS_OF, precedence=())
    with pytest.raises(ValueError, match="reinvest_at"):
        total_return(rp, ra, _sessions(date(2020, 3, 2), date(2020, 3, 2)), reinvest_at="pay")


def test_empty_inputs():
    rp = resolve_prices(_prices(), IDS, as_of=AS_OF, precedence=())
    ra = resolve_actions(_actions(), IDS, as_of=AS_OF, precedence=())
    out = total_return(rp, ra, _sessions(date(2020, 3, 2), date(2020, 3, 3)))
    assert out.num_rows == 0
    assert out.column_names == ["instrument_id", "session_date", "tr_index", "flags"]


def test_lower_precedence_source_fills_missing_session():
    prices = pa.concat_tables(
        [
            _prices((date(2020, 3, 2), 100.0, dict(source_code="srca"))),
            _prices(
                (date(2020, 3, 2), 99.0, dict(source_code="srcb")),
                (date(2020, 3, 3), 101.0, dict(source_code="srcb")),
            ),
        ]
    )
    ids = _ids(("srca:TOKA", M1), ("srcb:TOKA", M1))
    out = resolve_prices(prices, ids, as_of=AS_OF, precedence=("srca", "srcb"))
    assert [(r["source_code"], r["close"]) for r in out.to_pylist()] == [
        ("srca", 100.0),
        ("srcb", 101.0),
    ]


def test_old_share_dividend_without_split_used_as_is():
    prices = _prices((date(2020, 3, 2), 100.0), (date(2020, 3, 3), 99.0))
    actions = _actions(_cash(date(2020, 3, 3), 1.0, per_share_basis="old_share"))
    assert _rows(_tr(prices, actions))[-1] == (date(2020, 3, 3), 1.0, None)
