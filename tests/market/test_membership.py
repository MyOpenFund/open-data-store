from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pytest
from conftest import identifier_row, snapshot_row, table

from open_data_store.contracts.standard.market import INDEX_SNAPSHOTS, INSTRUMENT_IDENTIFIERS
from open_data_store.market.membership import membership_intervals, unresolved_members

M1 = "mof:" + "1" * 32
M2 = "mof:" + "2" * 32
M3 = "mof:" + "3" * 32


def _at(d: date, hour: int = 23) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


def _snaps(*members, index_code="idx1"):
    """members: (valid_date, source_symbol[, overrides]). known_at = valid_date 23:00 UTC."""
    rows = []
    for m in members:
        d, sym = m[0], m[1]
        over = m[2] if len(m) > 2 else {}
        row = dict(valid_date=d, source_symbol=sym, known_at=_at(d), index_code=index_code)
        row.update(over)
        rows.append(snapshot_row(**row))
    return table(INDEX_SNAPSHOTS, rows)


def _ids(*maps):
    """maps: (id_value, instrument_id[, overrides])."""
    rows = []
    for m in maps:
        row = dict(id_value=m[0], instrument_id=m[1])
        row.update(m[2] if len(m) > 2 else {})
        rows.append(identifier_row(**row))
    return table(INSTRUMENT_IDENTIFIERS, rows)


def _rows(out: pa.Table):
    return [
        (r["instrument_id"], r["valid_from"], r["valid_to"], r["source_code"])
        for r in out.to_pylist()
    ]


def _run(snaps, ids, as_of: date, **kw):
    return membership_intervals(snaps, ids, index_code="idx1", as_of=_at(as_of, 12), **kw)


# --- core rules --------------------------------------------------------------------------------


def test_venue_change_no_gap():
    snaps = _snaps((date(2006, 5, 31), "TOKA"), (date(2006, 6, 30), "TOKB"))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M1))
    out = _run(snaps, ids, date(2006, 7, 15))
    assert _rows(out) == [(M1, date(2006, 5, 31), None, "srcx")]


def test_weekend_month_end_kept_as_filed():
    assert date(2026, 1, 31).weekday() == 5
    out = _run(_snaps((date(2026, 1, 31), "TOKA")), _ids(("srcx:TOKA", M1)), date(2026, 2, 10))
    assert _rows(out) == [(M1, date(2026, 1, 31), None, "srcx")]


def test_removed_member_interval_closes_at_next_snapshot():
    snaps = _snaps(
        (date(2020, 1, 31), "TOKA"), (date(2020, 1, 31), "TOKB"), (date(2020, 2, 29), "TOKA")
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    out = _run(snaps, ids, date(2020, 3, 10))
    assert _rows(out) == [
        (M1, date(2020, 1, 31), None, "srcx"),
        (M2, date(2020, 1, 31), date(2020, 2, 29), "srcx"),
    ]


def test_staleness_cap():
    snaps = _snaps((date(2001, 1, 31), "TOKA"), (date(2001, 6, 30), "TOKA"))
    out = _run(snaps, _ids(("srcx:TOKA", M1)), date(2001, 7, 1), staleness_days=45)
    assert _rows(out) == [
        (M1, date(2001, 1, 31), date(2001, 3, 17), "srcx"),
        (M1, date(2001, 6, 30), None, "srcx"),
    ]


def test_unmapped_reported_not_dropped():
    snaps = _snaps((date(2020, 1, 31), "TOKA"), (date(2020, 1, 31), "TOKZ"))
    ids = _ids(("srcx:TOKA", M1))
    out = _run(snaps, ids, date(2020, 2, 10))
    assert _rows(out) == [(M1, date(2020, 1, 31), None, "srcx")]
    un = unresolved_members(snaps, ids, index_code="idx1", as_of=_at(date(2020, 2, 10), 12))
    assert un.to_pylist() == [
        dict(
            source_code="srcx",
            source_symbol="TOKZ",
            first_seen=date(2020, 1, 31),
            last_seen=date(2020, 1, 31),
            valid_to=None,
        )
    ]


def test_asof_hides_later_snapshot():
    snaps = _snaps(
        (date(2020, 1, 31), "TOKA"),
        (date(2020, 1, 31), "TOKB"),
        # Filed for 2020-02-29 but only known on 2020-03-05.
        (date(2020, 2, 29), "TOKA", dict(known_at=_at(date(2020, 3, 5)))),
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    open_both = [(M1, date(2020, 1, 31), None, "srcx"), (M2, date(2020, 1, 31), None, "srcx")]
    assert _rows(_run(snaps, ids, date(2020, 2, 15))) == open_both  # valid_date in the future
    assert _rows(_run(snaps, ids, date(2020, 3, 1))) == open_both  # not yet known
    assert _rows(_run(snaps, ids, date(2020, 3, 6))) == [
        (M1, date(2020, 1, 31), None, "srcx"),
        (M2, date(2020, 1, 31), date(2020, 2, 29), "srcx"),
    ]


# --- edge cases --------------------------------------------------------------------------------


def test_recycled_symbol_maps_to_disjoint_instruments():
    snaps = _snaps(
        (date(2009, 11, 30), "TOKA"), (date(2009, 12, 31), "TOKA"), (date(2010, 1, 31), "TOKA")
    )
    ids = _ids(
        ("srcx:TOKA", M1, dict(valid_date=date(1990, 1, 1), valid_to=date(2010, 1, 31))),
        ("srcx:TOKA", M2, dict(valid_date=date(2010, 1, 31))),
    )
    out = _run(snaps, ids, date(2010, 2, 5))
    assert _rows(out) == [
        (M1, date(2009, 11, 30), date(2010, 1, 31), "srcx"),
        (M2, date(2010, 1, 31), None, "srcx"),
    ]


def test_mapping_not_yet_valid_is_unresolved():
    snaps = _snaps((date(2020, 1, 31), "TOKA"), (date(2020, 2, 29), "TOKA"))
    ids = _ids(("srcx:TOKA", M1, dict(valid_date=date(2020, 2, 1))))
    as_of = date(2020, 3, 5)
    assert _rows(_run(snaps, ids, as_of)) == [(M1, date(2020, 2, 29), None, "srcx")]
    un = unresolved_members(snaps, ids, index_code="idx1", as_of=_at(as_of, 12))
    assert [(r["source_symbol"], r["first_seen"]) for r in un.to_pylist()] == [
        ("TOKA", date(2020, 1, 31))
    ]


def test_mapping_correction_read_latest():
    snaps = _snaps((date(2020, 1, 31), "TOKA"))
    ids = _ids(
        ("srcx:TOKA", M1, dict(known_at=_at(date(2019, 1, 1)))),
        ("srcx:TOKA", M2, dict(known_at=_at(date(2019, 1, 1)), valid_date=date(1991, 1, 1))),
        # Correction of the M2 row: its validity is closed before any snapshot.
        (
            "srcx:TOKA",
            M2,
            dict(
                known_at=_at(date(2025, 1, 1)),
                valid_date=date(1991, 1, 1),
                valid_to=date(1992, 1, 1),
            ),
        ),
    )
    assert _rows(_run(snaps, ids, date(2020, 2, 5))) == [(M1, date(2020, 1, 31), None, "srcx")]


def test_ambiguous_mapping_raises():
    snaps = _snaps((date(2020, 1, 31), "TOKA"))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKA", M2, dict(valid_date=date(2000, 1, 1))))
    with pytest.raises(ValueError, match=r"srcx:TOKA.*2020-01-31|2020-01-31.*srcx:TOKA"):
        _run(snaps, ids, date(2020, 2, 5))
    with pytest.raises(ValueError, match="TOKA"):
        unresolved_members(snaps, ids, index_code="idx1", as_of=_at(date(2020, 2, 5)))


def test_two_covering_rows_same_instrument_is_not_ambiguous():
    snaps = _snaps((date(2020, 1, 31), "TOKA"))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKA", M1, dict(valid_date=date(2000, 1, 1))))
    assert _rows(_run(snaps, ids, date(2020, 2, 5))) == [(M1, date(2020, 1, 31), None, "srcx")]


def test_other_schemes_and_sources_ignored():
    snaps = _snaps((date(2020, 1, 31), "TOKA"))
    ids = _ids(
        ("srcx:TOKA", M1, dict(id_scheme="ticker")),  # wrong scheme
        ("srcy:TOKA", M2),  # same symbol, other source
    )
    assert _run(snaps, ids, date(2020, 2, 5)).num_rows == 0


def test_two_sources_without_precedence_raises():
    snaps = pa.concat_tables(
        [
            _snaps((date(2020, 1, 31), "TOKA")),
            _snaps((date(2020, 1, 31), "TOKA", dict(source_code="srcy"))),
        ]
    )
    ids = _ids(("srcx:TOKA", M1), ("srcy:TOKA", M1))
    with pytest.raises(ValueError, match="precedence"):
        _run(snaps, ids, date(2020, 2, 5))


def test_other_index_does_not_make_sources_ambiguous():
    snaps = pa.concat_tables(
        [
            _snaps((date(2020, 1, 31), "TOKA")),
            _snaps((date(2020, 1, 31), "TOKA", dict(source_code="srcy")), index_code="idx2"),
        ]
    )
    ids = _ids(("srcx:TOKA", M1))
    assert _rows(_run(snaps, ids, date(2020, 2, 5))) == [(M1, date(2020, 1, 31), None, "srcx")]


def test_precedence_chooses_source_per_date():
    y = dict(source_code="srcy")
    snaps = pa.concat_tables(
        [
            _snaps(
                (date(2020, 1, 31), "TOKA"),
                (date(2020, 1, 31), "TOKB"),
                (date(2020, 2, 29), "TOKA"),
                (date(2020, 2, 29), "TOKB"),
            ),
            # srcy wins 2020-01-31 (TOKA only) and has no 2020-02-29 snapshot.
            _snaps((date(2020, 1, 31), "TOKA", y)),
        ]
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2), ("srcy:TOKA", M1))
    out = _run(snaps, ids, date(2020, 3, 5), precedence=("srcy", "srcx"))
    assert _rows(out) == [
        (M1, date(2020, 1, 31), date(2020, 2, 29), "srcy"),
        (M1, date(2020, 2, 29), None, "srcx"),
        (M2, date(2020, 2, 29), None, "srcx"),
    ]


def test_source_not_in_precedence_ignored():
    snaps = pa.concat_tables(
        [
            _snaps((date(2020, 1, 31), "TOKA")),
            # srcy alone files a later snapshot: it is not listed, so it must not end the interval.
            _snaps((date(2020, 2, 29), "TOKB", dict(source_code="srcy"))),
        ]
    )
    ids = _ids(("srcx:TOKA", M1), ("srcy:TOKB", M2))
    out = _run(snaps, ids, date(2020, 3, 5), precedence=("srcx",))
    assert _rows(out) == [(M1, date(2020, 1, 31), None, "srcx")]


def test_last_snapshot_staleness():
    snaps = _snaps((date(2020, 1, 31), "TOKA"))
    ids = _ids(("srcx:TOKA", M1))
    # 2020-01-31 + 45 days = 2020-03-16.
    assert _rows(_run(snaps, ids, date(2020, 3, 16))) == [(M1, date(2020, 1, 31), None, "srcx")]
    assert _rows(_run(snaps, ids, date(2020, 3, 17))) == [
        (M1, date(2020, 1, 31), date(2020, 3, 16), "srcx")
    ]
    assert _rows(_run(snaps, ids, date(2020, 2, 5), staleness_days=3)) == [
        (M1, date(2020, 1, 31), date(2020, 2, 3), "srcx")
    ]


def test_gap_equal_to_staleness_is_not_capped():
    snaps = _snaps((date(2020, 1, 1), "TOKA"), (date(2020, 1, 11), "TOKB"))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    out = _run(snaps, ids, date(2020, 1, 12), staleness_days=10)
    assert _rows(out) == [
        (M1, date(2020, 1, 1), date(2020, 1, 11), "srcx"),
        (M2, date(2020, 1, 11), None, "srcx"),
    ]


def test_is_member_false_removes_member():
    snaps = _snaps(
        (date(2020, 1, 31), "TOKA"),
        (date(2020, 1, 31), "TOKB"),
        (date(2020, 2, 29), "TOKA"),
        (date(2020, 2, 29), "TOKB", dict(is_member=False)),
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    out = _run(snaps, ids, date(2020, 3, 5))
    assert _rows(out) == [
        (M1, date(2020, 1, 31), None, "srcx"),
        (M2, date(2020, 1, 31), date(2020, 2, 29), "srcx"),
    ]


def test_is_member_correction_is_bitemporal():
    d = date(2020, 1, 31)
    snaps = _snaps(
        (d, "TOKA"),
        (d, "TOKB"),
        (d, "TOKB", dict(is_member=False, known_at=_at(date(2020, 2, 10)))),
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    before = _run(snaps, ids, date(2020, 2, 5))
    assert [r[0] for r in _rows(before)] == [M1, M2]
    after = _run(snaps, ids, date(2020, 2, 15))
    assert [r[0] for r in _rows(after)] == [M1]
    un = unresolved_members(
        snaps, _ids(("srcx:TOKA", M1)), index_code="idx1", as_of=_at(date(2020, 2, 15))
    )
    assert un.num_rows == 0  # a non-member is never reported as unresolved


def test_duplicate_tokens_count_once():
    d = date(2020, 1, 31)
    snaps = _snaps((d, "TOKA"), (d, "TOKB"), (d + timedelta(days=29), "TOKA"))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M1))
    out = _run(snaps, ids, date(2020, 3, 5))
    assert _rows(out) == [(M1, d, None, "srcx")]


def test_reentry_after_absence_gives_two_intervals():
    snaps = _snaps(
        (date(2020, 1, 31), "TOKA"),
        (date(2020, 1, 31), "TOKC"),
        (date(2020, 2, 29), "TOKC"),
        (date(2020, 3, 31), "TOKA"),
        (date(2020, 3, 31), "TOKC"),
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKC", M3))
    out = _run(snaps, ids, date(2020, 4, 5))
    assert _rows(out) == [
        (M1, date(2020, 1, 31), date(2020, 2, 29), "srcx"),
        (M1, date(2020, 3, 31), None, "srcx"),
        (M3, date(2020, 1, 31), None, "srcx"),
    ]


def test_unresolved_first_last_seen_sorted():
    snaps = _snaps(
        (date(2020, 1, 31), "TOKZ"),
        (date(2020, 1, 31), "TOKY"),
        (date(2020, 2, 29), "TOKA"),
        (date(2020, 3, 31), "TOKZ"),
    )
    ids = _ids(("srcx:TOKA", M1))
    un = unresolved_members(snaps, ids, index_code="idx1", as_of=_at(date(2020, 4, 5)))
    assert un.schema == pa.schema(
        [
            ("source_code", pa.string()),
            ("source_symbol", pa.string()),
            ("first_seen", pa.date32()),
            ("last_seen", pa.date32()),
            ("valid_to", pa.date32()),
        ]
    )
    assert [tuple(r.values()) for r in un.to_pylist()] == [
        ("srcx", "TOKY", date(2020, 1, 31), date(2020, 1, 31), date(2020, 2, 29)),
        ("srcx", "TOKZ", date(2020, 1, 31), date(2020, 3, 31), None),
    ]


def test_output_schema_and_empty_index():
    out = _run(_snaps((date(2020, 1, 31), "TOKA")), _ids(("srcx:TOKA", M1)), date(2020, 2, 5))
    assert out.schema == pa.schema(
        [
            ("instrument_id", pa.string()),
            ("index_code", pa.string()),
            ("valid_from", pa.date32()),
            ("valid_to", pa.date32()),
            ("source_code", pa.string()),
        ]
    )
    assert out.column("index_code").to_pylist() == ["idx1"]
    empty = membership_intervals(
        _snaps((date(2020, 1, 31), "TOKA")),
        _ids(("srcx:TOKA", M1)),
        index_code="other",
        as_of=_at(date(2020, 2, 5)),
    )
    assert empty.num_rows == 0 and empty.schema == out.schema


def test_parameters_validated():
    snaps, ids = _snaps((date(2020, 1, 31), "TOKA")), _ids(("srcx:TOKA", M1))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="staleness_days"):
            _run(snaps, ids, date(2020, 2, 5), staleness_days=bad)
    with pytest.raises(ValueError, match="timezone"):
        membership_intervals(snaps, ids, index_code="idx1", as_of=datetime(2020, 2, 5))
    with pytest.raises(TypeError, match="precedence"):
        _run(snaps, ids, date(2020, 2, 5), precedence="srcx")
    for bad in (3.0, "3", True):
        with pytest.raises(TypeError, match="staleness_days"):
            _run(snaps, ids, date(2020, 2, 5), staleness_days=bad)


def test_staleness_days_accepts_any_integral():
    snaps, ids = _snaps((date(2020, 1, 31), "TOKA")), _ids(("srcx:TOKA", M1))
    out = _run(snaps, ids, date(2020, 2, 5), staleness_days=np.int64(3))
    assert _rows(out) == [(M1, date(2020, 1, 31), date(2020, 2, 3), "srcx")]


# --- source choice, restatements and staleness -------------------------------------------------


def test_all_non_member_snapshot_closes_every_interval():
    snaps = _snaps(
        (date(2020, 1, 31), "TOKA"),
        (date(2020, 1, 31), "TOKB"),
        (date(2020, 2, 29), "TOKA", dict(is_member=False)),
        (date(2020, 2, 29), "TOKB", dict(is_member=False)),
    )
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    out = _run(snaps, ids, date(2020, 3, 5))
    assert _rows(out) == [
        (M1, date(2020, 1, 31), date(2020, 2, 29), "srcx"),
        (M2, date(2020, 1, 31), date(2020, 2, 29), "srcx"),
    ]


def test_losing_source_unmapped_symbol_not_reported():
    y = dict(source_code="srcy")
    snaps = pa.concat_tables(
        [
            _snaps((date(2020, 1, 31), "TOKA")),
            _snaps((date(2020, 1, 31), "TOKA", y), (date(2020, 1, 31), "TOKQ", y)),
        ]
    )
    ids = _ids(("srcx:TOKA", M1), ("srcy:TOKA", M1))  # srcy:TOKQ unmapped, but srcy loses
    as_of = _at(date(2020, 2, 5))
    kw = dict(index_code="idx1", as_of=as_of, precedence=("srcx", "srcy"))
    assert unresolved_members(snaps, ids, **kw).num_rows == 0
    assert _rows(membership_intervals(snaps, ids, **kw)) == [(M1, date(2020, 1, 31), None, "srcx")]


def test_preferred_source_unmapped_leaves_universe_hole():
    y = dict(source_code="srcy")
    snaps = pa.concat_tables(
        [
            _snaps((date(2020, 1, 31), "TOKA"), (date(2020, 2, 29), "TOKA")),
            _snaps((date(2020, 1, 31), "TOKA", y), (date(2020, 2, 29), "TOKA", y)),
        ]
    )
    # srcx:TOKA maps only from 2020-02-01; srcy:TOKA maps on both dates.
    ids = _ids(
        ("srcx:TOKA", M1, dict(valid_date=date(2020, 2, 1))),
        ("srcy:TOKA", M1),
    )
    kw = dict(index_code="idx1", as_of=_at(date(2020, 3, 5)), precedence=("srcx", "srcy"))
    # No fallback to srcy on 2020-01-31: the instrument is missing that month.
    assert _rows(membership_intervals(snaps, ids, **kw)) == [(M1, date(2020, 2, 29), None, "srcx")]
    assert [tuple(r.values()) for r in unresolved_members(snaps, ids, **kw).to_pylist()] == [
        ("srcx", "TOKA", date(2020, 1, 31), date(2020, 1, 31), date(2020, 2, 29))
    ]


def test_restatement_omitting_symbol_keeps_old_row():
    d = date(2020, 1, 31)
    late = dict(known_at=_at(date(2020, 2, 10)))
    ids = _ids(("srcx:TOKA", M1), ("srcx:TOKB", M2))
    # Restatement re-files TOKA only: the TOKB row of the original filing stays visible.
    omitted = _snaps((d, "TOKA"), (d, "TOKB"), (d, "TOKA", late))
    assert [r[0] for r in _rows(_run(omitted, ids, date(2020, 2, 15)))] == [M1, M2]
    # The ingestor contract: a removal is an explicit is_member = false row.
    explicit = _snaps(
        (d, "TOKA"), (d, "TOKB"), (d, "TOKA", late), (d, "TOKB", dict(is_member=False, **late))
    )
    assert [r[0] for r in _rows(_run(explicit, ids, date(2020, 2, 15)))] == [M1]


def test_last_snapshot_staleness_uses_utc_date_of_non_utc_as_of():
    snaps, ids = _snaps((date(2020, 1, 31), "TOKA")), _ids(("srcx:TOKA", M1))

    def run(as_of):
        return _rows(membership_intervals(snaps, ids, index_code="idx1", as_of=as_of))

    # 2020-03-16 20:00 at UTC-5 is 2020-03-17 01:00 UTC: 46 days, closed at D + 45.
    west = datetime(2020, 3, 16, 20, tzinfo=timezone(timedelta(hours=-5)))
    assert run(west) == [(M1, date(2020, 1, 31), date(2020, 3, 16), "srcx")]
    # 2020-03-17 02:00 at UTC+5 is 2020-03-16 21:00 UTC: 45 days, still open.
    east = datetime(2020, 3, 17, 2, tzinfo=timezone(timedelta(hours=5)))
    assert run(east) == [(M1, date(2020, 1, 31), None, "srcx")]


def test_unresolved_valid_to_follows_the_interval_rules():
    """valid_to closes the last unresolved snapshot exactly as membership_intervals would."""
    snaps = _snaps(
        (date(2020, 1, 31), "TOKZ"),
        (date(2020, 1, 31), "TOKA"),
        (date(2020, 6, 30), "TOKA"),  # gap > staleness after 2020-01-31
    )
    ids = _ids(("srcx:TOKA", M1))
    kw = dict(index_code="idx1", staleness_days=45)

    def valid_to(as_of):
        un = unresolved_members(snaps, ids, as_of=_at(as_of, 12), **kw)
        return un.column("valid_to").to_pylist()

    assert valid_to(date(2020, 2, 10)) == [None]  # last snapshot, still fresh
    assert valid_to(date(2020, 5, 1)) == [date(2020, 3, 16)]  # last snapshot, stale
    assert valid_to(date(2020, 7, 1)) == [date(2020, 3, 16)]  # gap capped at 45 days
    iv = membership_intervals(snaps, ids, as_of=_at(date(2020, 7, 1), 12), **kw)
    assert _rows(iv)[0] == (M1, date(2020, 1, 31), date(2020, 3, 16), "srcx")
