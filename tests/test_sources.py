import pyarrow as pa
import pytest

from open_data_store.contracts.base import ContractError
from open_data_store.sources import (
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
    out = filter_sources(_table(), frozenset(), restricted_codes(reg))
    assert out.column("source_code").to_pylist() == ["sec"]


def test_allowed_restricted_kept():
    reg = {"tiingo": _src("tiingo", "restricted")}
    out = filter_sources(_table(), frozenset({"tiingo"}), restricted_codes(reg))
    assert out.column("source_code").to_pylist() == ["tiingo", "sec", "tiingo"]


def test_conflicting_declarations_raise():
    with pytest.raises(ContractError):
        discover_sources([_src("a", licence="X"), _src("a", licence="Y")])


def test_identical_redeclaration_ok():
    assert discover_sources([_src("a"), _src("a")])["a"].code == "a"


def test_table_without_source_code_untouched():
    t = pa.table({"v": [1, 2]})
    assert filter_sources(t, frozenset(), frozenset({"tiingo"})) is t


def test_ml_forbidden_codes():
    reg = {"m": _src("m", no_ml_training=True), "o": _src("o")}
    assert ml_forbidden_codes(reg) == frozenset({"m"})


def test_bad_access_raises():
    for bad in ("Restricted", "closed"):
        with pytest.raises(ContractError):
            _src("x", bad)


def test_null_source_code_dropped():
    t = pa.table({"source_code": ["sec", None], "v": [1, 2]})
    out = filter_sources(t, frozenset(), frozenset())
    assert out.column("source_code").to_pylist() == ["sec"]


def test_two_restricted_one_allowed():
    t = pa.table({"source_code": ["a", "b", "sec"], "v": [1, 2, 3]})
    out = filter_sources(t, frozenset({"a"}), frozenset({"a", "b"}))
    assert out.column("source_code").to_pylist() == ["a", "sec"]


def test_restricted_codes_multi():
    reg = {"a": _src("a", "restricted"), "b": _src("b"), "c": _src("c", "restricted")}
    assert restricted_codes(reg) == frozenset({"a", "c"})
