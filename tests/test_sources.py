import pyarrow as pa
import pytest

from open_data_store import sources
from open_data_store.contracts.base import ContractError
from open_data_store.sources import (
    GROUP,
    SourceInfo,
    discover_sources,
    filter_sources,
    ml_forbidden_codes,
    restricted_codes,
)


def _src(code, access="open", licence="L", **kw):
    return SourceInfo(
        code=code, licence=licence, terms_url="https://x", attribution="a", access=access, **kw
    )


def _table():
    return pa.table({"source_code": ["tiingo", "sec", "tiingo"], "v": [1, 2, 3]})


def test_restricted_rows_dropped_by_default():
    reg = {"tiingo": _src("tiingo", "restricted")}
    out = filter_sources(_table(), allowed_restricted=frozenset(), restricted=restricted_codes(reg))
    assert out.column("source_code").to_pylist() == ["sec"]


def test_allowed_restricted_kept():
    reg = {"tiingo": _src("tiingo", "restricted")}
    out = filter_sources(
        _table(), allowed_restricted=frozenset({"tiingo"}), restricted=restricted_codes(reg)
    )
    assert out.column("source_code").to_pylist() == ["tiingo", "sec", "tiingo"]


def test_conflicting_declarations_raise():
    with pytest.raises(ContractError):
        discover_sources([_src("a", licence="X"), _src("a", licence="Y")])


def test_identical_redeclaration_ok():
    assert discover_sources([_src("a"), _src("a")])["a"].code == "a"


def test_table_without_source_code_untouched():
    t = pa.table({"v": [1, 2]})
    assert filter_sources(t, allowed_restricted=frozenset(), restricted=frozenset({"tiingo"})) is t


def test_ml_forbidden_codes():
    reg = {"m": _src("m", no_ml_training=True), "o": _src("o")}
    assert ml_forbidden_codes(reg) == frozenset({"m"})


def test_bad_access_raises():
    for bad in ("Restricted", "closed"):
        with pytest.raises(ContractError):
            _src("x", bad)


def test_null_source_code_dropped():
    t = pa.table({"source_code": ["sec", None], "v": [1, 2]})
    out = filter_sources(t, allowed_restricted=frozenset(), restricted=frozenset())
    assert out.column("source_code").to_pylist() == ["sec"]


def test_two_restricted_one_allowed():
    t = pa.table({"source_code": ["a", "b", "sec"], "v": [1, 2, 3]})
    out = filter_sources(t, allowed_restricted=frozenset({"a"}), restricted=frozenset({"a", "b"}))
    assert out.column("source_code").to_pylist() == ["a", "sec"]


def test_restricted_codes_multi():
    reg = {"a": _src("a", "restricted"), "b": _src("b"), "c": _src("c", "restricted")}
    assert restricted_codes(reg) == frozenset({"a", "c"})


# --- source codes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", " tiingo", "tiingo ", "Tiingo", "ti-ingo", "1tiingo", "_x", "a" * 64]
)
def test_bad_source_code_raises(bad):
    with pytest.raises(ContractError, match="code"):
        _src(bad)


@pytest.mark.parametrize("good", ["a", "sec", "srcx_2", "a" * 63])
def test_good_source_code_accepted(good):
    assert _src(good).code == good


# --- signature and strict mode -----------------------------------------------------------------


def test_filter_sources_sets_are_keyword_only():
    with pytest.raises(TypeError):
        filter_sources(_table(), frozenset(), frozenset())  # type: ignore[misc]


def test_strict_denies_unregistered_codes():
    t = pa.table({"source_code": ["sec", "ghost", "tiingo", "vend"], "v": [1, 2, 3, 4]})
    kw = dict(
        restricted=frozenset({"tiingo", "vend"}), registered=frozenset({"sec", "tiingo", "vend"})
    )
    out = filter_sources(t, allowed_restricted=frozenset({"vend"}), strict=True, **kw)
    assert out.column("source_code").to_pylist() == ["sec", "vend"]
    # Allowing an unregistered code does not let it through.
    out = filter_sources(t, allowed_restricted=frozenset({"ghost"}), strict=True, **kw)
    assert out.column("source_code").to_pylist() == ["sec"]
    # Non-strict keeps the unregistered code (open by default).
    out = filter_sources(
        t, allowed_restricted=frozenset(), restricted=frozenset({"tiingo", "vend"})
    )
    assert out.column("source_code").to_pylist() == ["sec", "ghost"]


def test_strict_needs_the_registered_codes():
    with pytest.raises(ValueError, match="registered"):
        filter_sources(
            _table(), allowed_restricted=frozenset(), restricted=frozenset(), strict=True
        )


def _identifiers():
    return pa.table(
        {
            "source_code": ["mapper", "mapper", "mapper", "mapper", "mapper"],
            "id_scheme": [
                "source_symbol",
                "source_symbol",
                "ticker",
                "source_symbol",
                "source_symbol",
            ],
            "id_value": ["vend:TOKA", "sec:TOKB", "vend:TOKC", "ghost:TOKD", "TOKE"],
        }
    )


def test_source_symbol_rows_of_a_denied_source_are_dropped():
    kw = dict(restricted=frozenset({"vend"}))
    out = filter_sources(_identifiers(), allowed_restricted=frozenset(), **kw)
    # The ticker row names no source; the malformed token (no prefix) fails closed.
    assert out.column("id_value").to_pylist() == ["sec:TOKB", "vend:TOKC", "ghost:TOKD"]
    out = filter_sources(_identifiers(), allowed_restricted=frozenset({"vend"}), **kw)
    assert out.column("id_value").to_pylist() == [
        "vend:TOKA",
        "sec:TOKB",
        "vend:TOKC",
        "ghost:TOKD",
    ]


def test_source_symbol_rows_of_an_unregistered_source_are_dropped_in_strict_mode():
    out = filter_sources(
        _identifiers(),
        allowed_restricted=frozenset({"vend"}),
        restricted=frozenset({"vend"}),
        strict=True,
        registered=frozenset({"mapper", "sec", "vend"}),
    )
    assert out.column("id_value").to_pylist() == ["vend:TOKA", "sec:TOKB", "vend:TOKC"]


# --- entry points ------------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, declared):
        self.declared = declared

    def load(self):
        return lambda: list(self.declared)


def test_discover_sources_reads_the_entry_point_group(monkeypatch):
    asked = []

    def fake_entry_points(*, group):
        asked.append(group)
        return [_FakeEntryPoint([_src("plug", "restricted")]), _FakeEntryPoint([_src("open1")])]

    monkeypatch.setattr(sources, "entry_points", fake_entry_points)
    got = discover_sources([_src("extra")])
    assert asked == [GROUP]
    assert sorted(got) == ["extra", "open1", "plug"]
    assert restricted_codes(got) == frozenset({"plug"})


def test_entry_point_conflicting_with_extra_raises(monkeypatch):
    monkeypatch.setattr(
        sources, "entry_points", lambda *, group: [_FakeEntryPoint([_src("plug", "restricted")])]
    )
    with pytest.raises(ContractError, match="plug"):
        discover_sources([_src("plug", "open")])
