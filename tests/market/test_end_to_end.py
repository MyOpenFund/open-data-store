"""End to end: publish synthetic market batches, load them through the binding, run every recipe."""

from __future__ import annotations

from datetime import date

import pyarrow.compute as pc
import pytest
from conftest import NOW, action_row, identifier_row, price_row, snapshot_row, ts
from test_publisher import land

from open_data_store import sources
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    MARKET_CONTRACTS,
    PRICES,
)
from open_data_store.market.asof import sessions
from open_data_store.market.coverage import coverage, delisting_exits
from open_data_store.market.membership import membership_intervals, unresolved_members
from open_data_store.market.returns import resolve_actions, resolve_prices, total_return
from open_data_store.market.store import load_market_inputs
from open_data_store.publisher import approve_contract, publish_pending
from open_data_store.reader import Store
from open_data_store.sources import SourceInfo

I1 = "mof:" + "1" * 32
I2 = "mof:" + "2" * 32
AS_OF = ts("2025-07-15T00:00:00")
SESS = sessions("XNYS", date(2025, 6, 2), date(2025, 6, 30))
DAYS = SESS["session_date"].to_pylist()
CLOSES = SESS["close_utc"].to_pylist()
LAST_B = date(2025, 6, 13)  # TOKB's last price: it dies mid-month
CORRECTED = date(2025, 6, 10)  # TOKA's close of that day is corrected on 2025-06-20
EX_DATE = date(2025, 6, 16)  # TOKA's cash dividend


@pytest.fixture
def registry(monkeypatch):
    declared = [
        SourceInfo("srcx", "open", "https://example.test/x", "X", "open"),
        SourceInfo("mapper", "open", "https://example.test/m", "M", "open"),
        SourceInfo("vendr", "vendor terms", "https://example.test/v", "V", "restricted"),
    ]
    real = sources.discover_sources
    monkeypatch.setattr(
        sources, "discover_sources", lambda extra=None: real([*(extra or []), *declared])
    )


@pytest.fixture
def store(approved, registry):
    for c in MARKET_CONTRACTS:
        approve_contract(approved, c, "test")
    prices = []
    for i, (d, k) in enumerate(zip(DAYS, CLOSES, strict=True)):
        base = dict(source_code="srcx", valid_date=d, known_at=k)
        prices.append(price_row(source_symbol="TOKA", close=100.0 + i, **base))
        if d <= LAST_B:
            prices.append(price_row(source_symbol="TOKB", close=50.0 - i, **base))
    # Rows the licence gate must drop: a restricted source and an unregistered one.
    prices.append(price_row(source_code="vendr", source_symbol="TOKA", valid_date=DAYS[0]))
    prices.append(price_row(source_code="ghost", source_symbol="TOKA", valid_date=DAYS[0]))
    land(approved, PRICES, prices, run_id="p1")
    correction = price_row(
        source_code="srcx",
        source_symbol="TOKA",
        valid_date=CORRECTED,
        known_at=ts("2025-06-20T12:00:00"),
        known_at_kind="collected",
        known_at_basis="fetch_time",
        close=999.0,
    )
    land(approved, PRICES, [correction], run_id="p2")
    dividend = action_row(
        source_code="srcx",
        source_symbol="TOKA",
        action_type="cash_dividend_regular",
        source_action_type="dividend",
        valid_date=EX_DATE,
        known_at=ts("2025-06-05T23:59:59"),
        amount=1.0,
        currency="USD",
        per_share_basis="new_share",
        ratio_old=None,
        ratio_new=None,
    )
    land(approved, CORPORATE_ACTIONS, [dividend], run_id="a1")
    may = [
        snapshot_row(
            valid_date=date(2025, 5, 31),
            known_at=ts("2025-05-31T23:59:59"),
            source_symbol=s,
            snapshot_size=3,
        )
        for s in ("TOKA", "TOKB", "DEAD")  # DEAD maps to no instrument
    ]
    land(approved, INDEX_SNAPSHOTS, may, run_id="s1")
    june = snapshot_row(
        valid_date=date(2025, 6, 30),
        known_at=ts("2025-06-30T23:59:59"),
        source_symbol="TOKA",
        snapshot_size=1,
    )
    land(approved, INDEX_SNAPSHOTS, [june], run_id="s2")
    ids = [
        identifier_row(instrument_id=I1, id_value="srcx:TOKA"),
        identifier_row(instrument_id=I2, id_value="srcx:TOKB"),
        identifier_row(instrument_id=I1, id_scheme="ticker", id_value="AAA"),
    ]
    land(approved, INSTRUMENT_IDENTIFIERS, ids, run_id="i1")
    assert publish_pending(approved, now=NOW).exit_code == 0
    with Store.open(approved) as s:
        yield s


def _close(table, instrument, day):
    rows = table.filter(pc.equal(table["instrument_id"], instrument)).to_pylist()
    return next(r["close"] for r in rows if r["session_date"] == day)


def test_publish_load_and_run_every_recipe(store):
    m = load_market_inputs(store, as_of=AS_OF)
    assert set(m.prices["source_code"].to_pylist()) == {"srcx"}  # vendr and ghost dropped

    kw = dict(index_code="idx1", as_of=AS_OF)
    intervals = membership_intervals(m.snapshots, m.identifiers, **kw)
    assert [
        (r["instrument_id"], r["valid_from"], r["valid_to"]) for r in intervals.to_pylist()
    ] == [
        (I1, date(2025, 5, 31), None),
        (I2, date(2025, 5, 31), date(2025, 6, 30)),
    ]
    unresolved = unresolved_members(m.snapshots, m.identifiers, **kw)
    assert unresolved.to_pylist() == [
        dict(
            source_code="srcx",
            source_symbol="DEAD",
            first_seen=date(2025, 5, 31),
            last_seen=date(2025, 5, 31),
            valid_to=date(2025, 6, 30),
        )
    ]

    rp = resolve_prices(m.prices, m.identifiers, as_of=AS_OF, precedence=())
    ra = resolve_actions(m.actions, m.identifiers, as_of=AS_OF, precedence=())
    assert "source_symbol" not in rp.column_names
    assert _close(rp, I1, CORRECTED) == 999.0  # the correction is known at AS_OF
    assert ra["instrument_id"].to_pylist() == [I1]

    tr = total_return(rp, ra, SESS)
    a = {r["session_date"]: r for r in tr.to_pylist() if r["instrument_id"] == I1}
    assert all(r["flags"] is None for r in a.values())
    prev = DAYS[DAYS.index(EX_DATE) - 1]
    expected = (_close(rp, I1, EX_DATE) + 1.0) / _close(rp, I1, prev)
    assert a[EX_DATE]["tr_index"] / a[prev]["tr_index"] == pytest.approx(expected)

    # Without the unresolved members, coverage is blind to DEAD and passes.
    assert coverage(intervals, rp, SESS).passed is True
    result = coverage(intervals, rp, SESS, unresolved=unresolved)
    assert result.passed is False
    first = result.per_session.to_pylist()[0]
    assert (first["members"], first["unmapped"], first["missing"]) == (3, 1, 1)
    reasons = {
        r["session_date"]: r["reason"]
        for r in result.per_member.to_pylist()
        if r["instrument_id"] == I2
    }
    assert reasons[LAST_B] == "ok"
    # I2's interval ends on 2025-06-30 (exclusive): every later session it was a member of.
    assert {reason for d, reason in reasons.items() if d > LAST_B} == {"delisted"}
    assert max(reasons) == date(2025, 6, 27)

    exits = delisting_exits(intervals, rp, SESS)
    assert exits.to_pylist() == [
        dict(
            instrument_id=I2,
            exit_date=LAST_B,
            exit_price=_close(rp, I2, LAST_B),
            rule="last_close",
            flagged=True,
        )
    ]


def test_load_at_an_earlier_date_hides_the_later_correction(store):
    t = ts("2025-06-12T23:00:00")
    m = load_market_inputs(store, as_of=t)
    rp = resolve_prices(m.prices, m.identifiers, as_of=t, precedence=())
    assert _close(rp, I1, CORRECTED) == 100.0 + DAYS.index(CORRECTED)
    assert max(rp["session_date"].to_pylist()) == date(2025, 6, 12)
