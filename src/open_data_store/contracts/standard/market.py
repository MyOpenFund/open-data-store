"""Standard contracts for market data: prices, corporate actions, index snapshots, identifiers.

Prices attach to a provider series (source_code, source_symbol), not to an instrument; a dated
mapping in instrument_identifiers serves them per instrument. Only raw prices are stored: any
adjustment is a recipe over prices and corporate_actions.
"""

import pyarrow as pa

from open_data_store.contracts.base import Column, Contract
from open_data_store.contracts.standard._common import BITEMPORAL_COLUMNS

__all__ = [
    "ACTION_TYPES",
    "AMOUNT_BASES",
    "CORPORATE_ACTIONS",
    "FETCH_LOG",
    "FETCH_STATUSES",
    "INDEX_SNAPSHOTS",
    "INSTRUMENT_IDENTIFIERS",
    "KNOWN_AT_BASES",
    "LINK_REASONS",
    "MARKET_CONTRACTS",
    "MARKET_ID_SCHEMES",
    "PER_SHARE_BASES",
    "PRICES",
    "PRICE_BASES",
    "VENDOR_TOTAL_RETURN",
]

PRICE_BASES = ("raw",)
KNOWN_AT_BASES = (
    "session_close",
    "fetch_time",
    "announce_date",
    "ex_date_assumed",
    "snapshot_date",
    "load_time",
    "effective_date",
)
ACTION_TYPES = (
    "cash_dividend_regular",
    "cash_dividend_special",
    "split",
    "stock_dividend",
    "spin_off",
    "return_of_capital",
    "rights",
)
AMOUNT_BASES = ("gross", "net")
PER_SHARE_BASES = ("old_share", "new_share")
# Own vocabulary: the shared ID_SCHEMES is not touched (it is part of every standard fingerprint).
MARKET_ID_SCHEMES = (
    "figi",
    "composite_figi",
    "share_class_figi",
    "isin",
    "cik",
    "ticker",
    "source_symbol",
    "export_member_id",
    "export_name",
    "successor_of",
)
FETCH_STATUSES = ("ok", "empty", "not_found", "partial", "error")
LINK_REASONS = ("successor", "venue_change", "manual")

_LICENCE = "per-source; see the source registry"
_CURRENCY_DOC = "ISO 4217, plus minor units as quoted: GBp, ZAc, ILA."

_SOURCE_SYMBOL = Column(
    "source_symbol",
    pa.string(),
    nullable=False,
    doc="The series' symbol at its source, as filed (opaque token for private sources).",
)
_KNOWN_AT_BASIS = Column(
    "known_at_basis",
    pa.string(),
    nullable=False,
    allowed=KNOWN_AT_BASES,
    doc="How known_at was derived (session close, fetch time, announce date, ...).",
)


def _contract(name, doc, columns, natural_key):
    return Contract(
        name=name,
        version="1.0.0",
        kind="fact",
        owner="open-data-store",
        licence=_LICENCE,
        doc=doc,
        columns=(*columns, _KNOWN_AT_BASIS, *BITEMPORAL_COLUMNS),
        natural_key=natural_key,
        entity_column_pair=None,
        partition_by=("source_code",),
    )


PRICES = _contract(
    "prices",
    "One row per provider series and session, raw (unadjusted) values as filed. valid_date is "
    "the session date. No adjusted value lives here.",
    (
        _SOURCE_SYMBOL,
        Column("open", pa.float64(), doc="Session open; null when the source has none."),
        Column("high", pa.float64(), doc="Session high; null when the source has none."),
        Column("low", pa.float64(), doc="Session low; null when the source has none."),
        Column("close", pa.float64(), nullable=False, doc="Session close, raw."),
        Column("volume", pa.int64(), doc="Shares traded; null when the source has none."),
        Column("currency", pa.string(), nullable=False, doc=_CURRENCY_DOC),
        Column(
            "price_basis",
            pa.string(),
            nullable=False,
            allowed=PRICE_BASES,
            doc="Only raw prices are stored.",
        ),
    ),
    ("source_code", "source_symbol", "valid_date"),
)

CORPORATE_ACTIONS = _contract(
    "corporate_actions",
    "One row per corporate action on a provider series. valid_date is the ex-date.",
    (
        _SOURCE_SYMBOL,
        Column(
            "action_type",
            pa.string(),
            nullable=False,
            allowed=ACTION_TYPES,
            doc="Normalised action type; the source's own label is source_action_type.",
        ),
        Column(
            "action_seq",
            pa.int16(),
            nullable=False,
            doc="Discriminator for two actions of one type on one ex-date (0, 1, ...).",
        ),
        Column("source_action_type", pa.string(), doc="The action type verbatim at the source."),
        Column("announce_date", pa.date32(), doc="Date the action was announced, if known."),
        Column("record_date", pa.date32(), doc="Record date, if known."),
        Column("pay_date", pa.date32(), doc="Payment date, if known."),
        Column("amount", pa.float64(), doc="Cash amount per share; null for non-cash actions."),
        Column("currency", pa.string(), doc=_CURRENCY_DOC + " Null for non-cash actions."),
        Column(
            "amount_basis",
            pa.string(),
            allowed=AMOUNT_BASES,
            doc="Whether the amount is gross or net of withholding; null for non-cash actions.",
        ),
        Column(
            "per_share_basis",
            pa.string(),
            allowed=PER_SHARE_BASES,
            doc="Whether the amount is per old or per new share.",
        ),
        Column("ratio_old", pa.float64(), doc="Old shares in the ratio (split, stock dividend)."),
        Column("ratio_new", pa.float64(), doc="New shares in the ratio (split, stock dividend)."),
        Column("spinoff_symbol", pa.string(), doc="Symbol of the spun-off security."),
    ),
    ("source_code", "source_symbol", "valid_date", "action_type", "action_seq"),
)

VENDOR_TOTAL_RETURN = _contract(
    "vendor_total_return",
    "One row per provider series and session: the vendor's own total-return index. Control and "
    "fallback only; never mixed into prices.",
    (
        _SOURCE_SYMBOL,
        Column("tr_index", pa.float64(), nullable=False, doc="Vendor total-return index level."),
    ),
    ("source_code", "source_symbol", "valid_date"),
)

INDEX_SNAPSHOTS = _contract(
    "index_snapshots",
    "One row per index member and snapshot. valid_date is the snapshot date as filed.",
    (
        Column(
            "index_code", pa.string(), nullable=False, doc="Index code, as the source names it."
        ),
        _SOURCE_SYMBOL,
        Column("snapshot_size", pa.int32(), nullable=False, doc="Members in this snapshot."),
        Column("weight", pa.float64(), doc="Member weight; null when the source has none."),
        Column(
            "is_member",
            pa.bool_(),
            nullable=False,
            doc="True by default; false only for an explicit correction.",
        ),
    ),
    ("source_code", "index_code", "source_symbol", "valid_date"),
)

INSTRUMENT_IDENTIFIERS = _contract(
    "instrument_identifiers",
    "One row per identifier observation of an instrument. valid_date is the start of validity; "
    "a correction is a later known_at.",
    (
        Column(
            "instrument_id",
            pa.string(),
            nullable=False,
            doc="Opaque id minted by the mapping: 'mof:' + UUID4 hex.",
        ),
        Column(
            "id_scheme",
            pa.string(),
            nullable=False,
            allowed=MARKET_ID_SCHEMES,
            doc="Identifier kind; a vocabulary of its own, not the shared ID_SCHEMES.",
        ),
        Column(
            "id_value",
            pa.string(),
            nullable=False,
            doc="The identifier. For id_scheme source_symbol: '<source_code>:<symbol>', "
            "e.g. 'massive:AAPL'.",
        ),
        Column("valid_to", pa.date32(), doc="End of validity; null while open."),
        Column(
            "link_reason",
            pa.string(),
            allowed=LINK_REASONS,
            doc="Why the link exists, if not a plain observation.",
        ),
    ),
    ("source_code", "instrument_id", "id_scheme", "id_value", "valid_date"),
)

FETCH_LOG = _contract(
    "fetch_log",
    "One row per fetch of one provider series: the coverage ledger. valid_date is the fetch date.",
    (
        _SOURCE_SYMBOL,
        Column("fetch_id", pa.string(), nullable=False, doc="Identifier of the fetch run."),
        Column("requested_from", pa.date32(), doc="Start of the requested range; null when none."),
        Column("requested_to", pa.date32(), doc="End of the requested range; null when none."),
        Column("returned_from", pa.date32(), doc="Null when nothing was returned."),
        Column("returned_to", pa.date32(), doc="Null when nothing was returned."),
        Column(
            "status",
            pa.string(),
            nullable=False,
            allowed=FETCH_STATUSES,
            doc="Outcome of the fetch.",
        ),
        Column("row_count", pa.int64(), nullable=False, doc="Rows the source returned."),
        Column("archive_ref", pa.string(), doc="Reference to the raw archive of the response."),
    ),
    ("source_code", "source_symbol", "fetch_id", "valid_date"),
)

MARKET_CONTRACTS = (
    PRICES,
    CORPORATE_ACTIONS,
    VENDOR_TOTAL_RETURN,
    INDEX_SNAPSHOTS,
    INSTRUMENT_IDENTIFIERS,
    FETCH_LOG,
)
