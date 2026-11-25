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
    assert out.num_rows == 3


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
