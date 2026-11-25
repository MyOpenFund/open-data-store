from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from open_data_store.market.coverage import (
    COVERAGE_PER_MEMBER_SCHEMA,
    COVERAGE_PER_SESSION_SCHEMA,
    DELISTING_EXITS_SCHEMA,
    FETCH_COVERAGE_SCHEMA,
    REASONS,
    coverage,
    delisting_exits,
)
from open_data_store.market.membership import INTERVALS_SCHEMA
from open_data_store.market.returns import RESOLVED_PRICES_SCHEMA

M1 = "mof:" + "1" * 32
M2 = "mof:" + "2" * 32
M3 = "mof:" + "3" * 32

# Ten synthetic weekday sessions, 2026-01-05 (Mon) .. 2026-01-16 (Fri).
DAYS = [date(2026, 1, 5) + timedelta(days=i) for i in range(12) if (5 + i) % 7 not in (3, 4)]
assert len(DAYS) == 10 and DAYS[-1] == date(2026, 1, 16)


def _sessions(days=DAYS):
    return pa.table(
        {
            "session_date": pa.array(days, pa.date32()),
            "close_utc": pa.array(
                [datetime(d.year, d.month, d.day, 21, tzinfo=UTC) for d in days],
                pa.timestamp("us", tz="UTC"),
            ),
        }
    )


def _intervals(*rows):
    """rows: (instrument_id, valid_from, valid_to)."""
    return pa.table(
        {
            "instrument_id": [r[0] for r in rows],
            "index_code": ["idx1"] * len(rows),
            "valid_from": [r[1] for r in rows],
            "valid_to": [r[2] for r in rows],
            "source_code": ["srcx"] * len(rows),
        },
        schema=INTERVALS_SCHEMA,
    )


def _prices(*rows):
    """rows: (instrument_id, session_date, close)."""
    return pa.table(
        {
            "instrument_id": [r[0] for r in rows],
            "session_date": [r[1] for r in rows],
            "close": [r[2] for r in rows],
            "volume": [100] * len(rows),
            "currency": ["USD"] * len(rows),
            "source_code": ["srcx"] * len(rows),
            "source_symbol": ["TOK"] * len(rows),
        },
        schema=RESOLVED_PRICES_SCHEMA,
    )


def _fetch(*rows):
    """rows: (instrument_id, covered_from, covered_to, status)."""
    return pa.table(
        {
            "instrument_id": [r[0] for r in rows],
            "covered_from": [r[1] for r in rows],
            "covered_to": [r[2] for r in rows],
            "status": [r[3] for r in rows],
        },
        schema=FETCH_COVERAGE_SCHEMA,
    )


def _full(inst, close=10.0, days=DAYS):
    return [(inst, d, close) for d in days]


def _reasons(result, inst):
    rows = result.per_member.filter(pc.equal(result.per_member["instrument_id"], inst))
    return dict(zip(rows["session_date"].to_pylist(), rows["reason"].to_pylist(), strict=True))


def _session(result, d):
    rows = result.per_session.to_pylist()
    return next(r for r in rows if r["session_date"] == d)


# --- brief tests --------------------------------------------------------------------------


def test_unexplained_gap_fails():
    intervals = _intervals((M1, DAYS[0], None))
    prices = _prices(*[r for r in _full(M1) if r[1] != DAYS[4]])
    result = coverage(intervals, prices, _sessions(), None, max_missing_share=0.0)
    assert _reasons(result, M1)[DAYS[4]] == "not_fetched"
    s = _session(result, DAYS[4])
    assert (s["members"], s["missing"], s["missing_share"], s["over_threshold"]) == (
        1,
        1,
        1.0,
        True,
    )
    assert result.passed is False


def test_missing_share_over_threshold_fails_even_if_explained():
    # 1 of 2 members on DAYS[4] has an explained gap (source has no data): still bias.
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    prices = _prices(*_full(M1), *[r for r in _full(M2) if r[1] != DAYS[4]])
    fetch = _fetch((M2, DAYS[0], DAYS[-1], "empty"))
    result = coverage(intervals, prices, _sessions(), fetch, max_missing_share=0.4)
    assert _reasons(result, M2)[DAYS[4]] == "source_has_no_data"
    s = _session(result, DAYS[4])
    assert (s["members"], s["missing"], s["missing_share"], s["over_threshold"]) == (
        2,
        1,
        0.5,
        True,
    )
    assert result.passed is False


def test_delisted_member_exits_at_last_close_and_is_flagged():
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    prices = _prices(*_full(M1), *[(M2, d, 7.0 + i) for i, d in enumerate(DAYS[:6])])
    exits = delisting_exits(intervals, prices, _sessions())
    assert exits.schema == DELISTING_EXITS_SCHEMA
    assert exits.to_pylist() == [
        {
            "instrument_id": M2,
            "exit_date": DAYS[5],
            "exit_price": 12.0,
            "rule": "last_close",
            "flagged": True,
        }
    ]
    result = coverage(intervals, prices, _sessions(), None, max_missing_share=0.0)
    assert set(_reasons(result, M2)[d] for d in DAYS[6:]) == {"delisted"}
    assert result.passed is True


def test_fixed_return_rule_requires_value():
    intervals = _intervals((M1, DAYS[0], None))
    prices = _prices(*_full(M1, days=DAYS[:3]), ("mof:" + "9" * 32, DAYS[-1], 1.0))
    with pytest.raises(ValueError, match="fixed_return"):
        delisting_exits(intervals, prices, _sessions(), rule="fixed_return")


# --- reasons, in priority order -----------------------------------------------------------


def test_reasons_vocabulary():
    assert REASONS == (
        "ok",
        "price_invalid",
        "not_fetched",
        "unmapped",
        "source_has_no_data",
        "recycled_at_source",
        "delisted",
        "session_not_published",
    )


def test_ok_cells_and_schemas():
    result = coverage(_intervals((M1, DAYS[0], None)), _prices(*_full(M1)), _sessions(), None)
    assert result.per_member.schema == COVERAGE_PER_MEMBER_SCHEMA
    assert result.per_session.schema == COVERAGE_PER_SESSION_SCHEMA
    assert set(_reasons(result, M1).values()) == {"ok"}
    assert len(_reasons(result, M1)) == 10
    assert result.per_session["missing"].to_pylist() == [0] * 10
    assert result.passed is True


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, None])
def test_trailing_invalid_close_is_price_invalid_never_dropped(bad):
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    prices = _prices(*_full(M1), *_full(M2, days=DAYS[:5]), (M2, DAYS[5], bad))
    result = coverage(intervals, prices, _sessions(), None, max_missing_share=0.5)
    reasons = _reasons(result, M2)
    assert reasons[DAYS[4]] == "ok"
    assert reasons[DAYS[5]] == "price_invalid"
    assert {reasons[d] for d in DAYS[6:]} == {"delisted"}
    s = _session(result, DAYS[5])
    assert (s["missing"], s["delisted"]) == (1, 0)


def test_member_with_only_invalid_closes():
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    prices = _prices(*_full(M1), (M2, DAYS[1], 0.0), (M2, DAYS[2], -3.0))
    result = coverage(intervals, prices, _sessions(), None)
    reasons = _reasons(result, M2)
    assert reasons[DAYS[0]] == "not_fetched"
    assert reasons[DAYS[1]] == reasons[DAYS[2]] == "price_invalid"
    assert {reasons[d] for d in DAYS[3:]} == {"delisted"}
    assert result.passed is False
    # No known close: the exit carries no price, still flagged.
    exits = delisting_exits(intervals, prices, _sessions())
    assert exits.to_pylist() == [
        {
            "instrument_id": M2,
            "exit_date": DAYS[2],
            "exit_price": None,
            "rule": "last_close",
            "flagged": True,
        }
    ]


def test_session_not_published_beats_delisted():
    # Prices stop at DAYS[7] for everyone: the last two sessions are not published, never
    # "delisted" (a stale feed must not exit live members).
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None), (M3, DAYS[0], None))
    prices = _prices(*_full(M1, days=DAYS[:8]), *_full(M2, days=DAYS[:3]))
    result = coverage(intervals, prices, _sessions(), None)
    for inst in (M1, M2, M3):
        assert {_reasons(result, inst)[d] for d in DAYS[8:]} == {"session_not_published"}
    assert {_reasons(result, M2)[d] for d in DAYS[3:8]} == {"delisted"}
    assert _session(result, DAYS[8])["missing"] == 3
    exits = delisting_exits(intervals, prices, _sessions())
    assert exits["instrument_id"].to_pylist() == [M2]


def test_source_has_no_data_per_status():
    intervals = _intervals(*[(m, DAYS[0], None) for m in (M1, M2, M3)])
    others = "mof:" + "4" * 32
    intervals = pa.concat_tables([intervals, _intervals((others, DAYS[0], None))])
    prices = _prices(("mof:" + "9" * 32, DAYS[-1], 1.0))
    fetch = _fetch(
        (M1, DAYS[0], DAYS[-1], "ok"),
        (M2, DAYS[0], DAYS[-1], "not_found"),
        (M3, DAYS[0], DAYS[4], "empty"),
        (others, DAYS[0], DAYS[-1], "error"),
        (others, DAYS[0], DAYS[-1], "partial"),
    )
    result = coverage(intervals, prices, _sessions(), fetch)
    assert set(_reasons(result, M1).values()) == {"source_has_no_data"}
    assert set(_reasons(result, M2).values()) == {"source_has_no_data"}
    m3 = _reasons(result, M3)
    assert {m3[d] for d in DAYS[:5]} == {"source_has_no_data"}  # covered_to inclusive
    assert {m3[d] for d in DAYS[5:]} == {"not_fetched"}
    assert set(_reasons(result, others).values()) == {"not_fetched"}


def test_fetch_coverage_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        coverage(
            _intervals((M1, DAYS[0], None)),
            _prices(*_full(M1)),
            _sessions(),
            _fetch((M1, DAYS[0], DAYS[-1], "bogus")),
        )


def test_not_fetched_without_fetch_coverage():
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    result = coverage(intervals, _prices(*_full(M1)), _sessions(), None)
    assert set(_reasons(result, M2).values()) == {"not_fetched"}


# --- expected cells and per-session counts ------------------------------------------------


def test_expected_cells_follow_half_open_intervals_and_last_session():
    intervals = _intervals(
        (M1, DAYS[2], DAYS[5]),  # [DAYS[2], DAYS[5])
        (M1, DAYS[4], DAYS[7]),  # overlap counts once
        (M2, date(2026, 1, 10), None),  # weekend start
        (M3, date(2027, 1, 1), None),  # after the last session: no cell
    )
    prices = _prices(*_full(M1), *_full(M2))
    result = coverage(intervals, prices, _sessions(), None)
    assert sorted(_reasons(result, M1)) == DAYS[2:7]
    assert sorted(_reasons(result, M2)) == DAYS[5:]
    assert _reasons(result, M3) == {}
    members = dict(
        zip(
            result.per_session["session_date"].to_pylist(),
            result.per_session["members"].to_pylist(),
            strict=True,
        )
    )
    assert members == {d: (d >= DAYS[2] and d < DAYS[7]) + (d >= DAYS[5]) for d in DAYS}
    assert result.per_member["instrument_id"].to_pylist() == sorted(
        result.per_member["instrument_id"].to_pylist()
    )


def test_delisted_not_counted_missing():
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    prices = _prices(*_full(M1), *_full(M2, days=DAYS[:2]))
    result = coverage(intervals, prices, _sessions(), None, max_missing_share=0.0)
    s = _session(result, DAYS[5])
    assert (s["members"], s["delisted"], s["missing"], s["missing_share"]) == (2, 1, 0, 0.0)
    assert result.passed is True


def test_zero_members_share_is_zero():
    intervals = _intervals((M1, DAYS[5], None))
    result = coverage(intervals, _prices(*_full(M1)), _sessions(), None, max_missing_share=0.0)
    s = _session(result, DAYS[0])
    assert (s["members"], s["missing"], s["missing_share"], s["over_threshold"]) == (
        0,
        0,
        0.0,
        False,
    )
    assert result.per_session.num_rows == 10


def test_empty_universe_passes():
    result = coverage(_intervals(), _prices(), _sessions(), None)
    assert result.per_member.num_rows == 0
    assert result.per_session["members"].to_pylist() == [0] * 10
    assert result.passed is True


@pytest.mark.parametrize("share", [0.0, 1.0, 0.02])
def test_threshold_bounds_accepted(share):
    coverage(
        _intervals((M1, DAYS[0], None)),
        _prices(*_full(M1)),
        _sessions(),
        None,
        max_missing_share=share,
    )


@pytest.mark.parametrize("share", [-0.01, 1.01, math.nan])
def test_threshold_out_of_bounds_refused(share):
    with pytest.raises(ValueError, match="max_missing_share"):
        coverage(
            _intervals((M1, DAYS[0], None)),
            _prices(*_full(M1)),
            _sessions(),
            None,
            max_missing_share=share,
        )


def test_threshold_is_strict():
    # Share exactly at the threshold passes; above it fails.
    intervals = _intervals(*[(m, DAYS[0], None) for m in (M1, M2)])
    prices = _prices(*_full(M1), *[r for r in _full(M2) if r[1] != DAYS[3]])
    assert coverage(intervals, prices, _sessions(), None, max_missing_share=0.5).passed
    assert not coverage(intervals, prices, _sessions(), None, max_missing_share=0.49).passed


def test_duplicate_price_key_raises():
    prices = _prices(*_full(M1), (M1, DAYS[0], 3.0))
    with pytest.raises(ValueError, match="several prices"):
        coverage(_intervals((M1, DAYS[0], None)), prices, _sessions(), None)


# --- delisting exits ----------------------------------------------------------------------


def _delisted_case():
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None), (M3, DAYS[0], None))
    prices = _prices(
        *_full(M1), *_full(M2, close=20.0, days=DAYS[:4]), *_full(M3, close=5.0, days=DAYS[:6])
    )
    return intervals, prices


def test_fixed_return_rule():
    intervals, prices = _delisted_case()
    exits = delisting_exits(intervals, prices, _sessions(), rule="fixed_return", fixed_return=-0.3)
    assert exits["instrument_id"].to_pylist() == [M2, M3]
    assert exits["exit_date"].to_pylist() == [DAYS[3], DAYS[5]]
    assert exits["exit_price"].to_pylist() == pytest.approx([14.0, 3.5])
    assert exits["rule"].to_pylist() == ["fixed_return"] * 2
    assert exits["flagged"].to_pylist() == [True, True]


def test_fixed_return_minus_one_accepted():
    intervals, prices = _delisted_case()
    exits = delisting_exits(intervals, prices, _sessions(), rule="fixed_return", fixed_return=-1.0)
    assert exits["exit_price"].to_pylist() == [0.0, 0.0]


@pytest.mark.parametrize("value", [-1.0001, math.nan, -math.inf])
def test_fixed_return_below_minus_one_refused(value):
    intervals, prices = _delisted_case()
    with pytest.raises(ValueError, match="fixed_return"):
        delisting_exits(intervals, prices, _sessions(), rule="fixed_return", fixed_return=value)


def test_unknown_rule_refused():
    intervals, prices = _delisted_case()
    with pytest.raises(ValueError, match="rule"):
        delisting_exits(intervals, prices, _sessions(), rule="mean_reversion")


def test_post_delisting_prices_rule_and_fallback():
    intervals, prices = _delisted_case()
    post = _prices(
        (M2, DAYS[2], 99.0),  # before the exit date: ignored
        (M2, DAYS[6], 1.5),
        (M2, DAYS[8], 0.8),  # the last post-delisting price wins
        (M2, DAYS[9], math.nan),  # invalid: ignored
        (M1, DAYS[9], 7.0),  # not delisted: ignored
    )
    exits = delisting_exits(
        intervals, prices, _sessions(), rule="post_delisting_prices", post_prices=post
    )
    assert exits.to_pylist() == [
        {
            "instrument_id": M2,
            "exit_date": DAYS[8],
            "exit_price": 0.8,
            "rule": "post_delisting_prices",
            "flagged": True,
        },
        {  # no post-delisting price: falls back to the last close, still flagged
            "instrument_id": M3,
            "exit_date": DAYS[5],
            "exit_price": 5.0,
            "rule": "last_close",
            "flagged": True,
        },
    ]


def test_post_delisting_prices_requires_table():
    intervals, prices = _delisted_case()
    with pytest.raises(ValueError, match="post_prices"):
        delisting_exits(intervals, prices, _sessions(), rule="post_delisting_prices")


def test_not_delisted_when_membership_ends_with_the_prices():
    # M2's prices stop at DAYS[3] and its membership ends at DAYS[4] (exclusive): no exit.
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], DAYS[4]), (M3, DAYS[0], DAYS[6]))
    prices = _prices(*_full(M1), *_full(M2, days=DAYS[:4]), *_full(M3, days=DAYS[:4]))
    exits = delisting_exits(intervals, prices, _sessions())
    assert exits["instrument_id"].to_pylist() == [M3]
    result = coverage(intervals, prices, _sessions(), None)
    assert "delisted" not in _reasons(result, M2).values()


def test_no_exit_for_instrument_without_prices_or_trading_to_the_end():
    intervals = _intervals((M1, DAYS[0], None), (M2, DAYS[0], None))
    exits = delisting_exits(intervals, _prices(*_full(M1)), _sessions())
    assert exits.num_rows == 0
    assert exits.schema == DELISTING_EXITS_SCHEMA


def test_exits_match_delisted_cells_on_random_chunked_universe():
    rng = np.random.default_rng(7)
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(60)]
    insts = [f"mof:{i:032x}" for i in range(40)]
    iv, pr, fc = [], [], []
    for k, m in enumerate(insts):
        a, b = sorted(rng.integers(0, 70, 2))
        iv.append(
            (
                m,
                days[0] + timedelta(days=int(a)),
                None if k % 3 else days[0] + timedelta(days=int(b)),
            )
        )
        stop = int(rng.integers(0, 61))
        for d in days[:stop]:
            if rng.random() > 0.1:
                pr.append((m, d, float(rng.choice([10.0, 0.0, math.nan], p=[0.9, 0.05, 0.05]))))
        fc.append((m, days[int(rng.integers(0, 60))], days[int(rng.integers(0, 60))], "empty"))
    intervals = pa.concat_tables([_intervals(*iv[:20]), _intervals(*iv[20:])])
    prices = pa.concat_tables([_prices(*pr[: len(pr) // 2]), _prices(*pr[len(pr) // 2 :])])
    sess = _sessions(days)
    result = coverage(intervals, prices, sess, _fetch(*fc))
    pm = result.per_member
    delisted = set(pm.filter(pc.equal(pm["reason"], "delisted"))["instrument_id"].to_pylist())
    exits = delisting_exits(intervals, prices, sess)
    assert set(exits["instrument_id"].to_pylist()) == delisted
    assert exits["flagged"].to_pylist() == [True] * exits.num_rows
    # Per-session counts are consistent with the per-member table.
    for row in result.per_session.to_pylist():
        cells = [r for r in pm.to_pylist() if r["session_date"] == row["session_date"]]
        assert row["members"] == len(cells)
        assert row["missing"] == sum(r["reason"] not in ("ok", "delisted") for r in cells)
