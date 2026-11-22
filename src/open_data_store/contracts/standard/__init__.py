"""Standard contracts owned by the library: domains several producers feed (spec §8)."""

from open_data_store.contracts.standard.entity_observations import ENTITY_OBSERVATIONS
from open_data_store.contracts.standard.filing_statements import FILING_STATEMENTS
from open_data_store.contracts.standard.filings import FILINGS
from open_data_store.contracts.standard.fundamentals import FUNDAMENTALS


def contracts():
    return [ENTITY_OBSERVATIONS, FILING_STATEMENTS, FILINGS, FUNDAMENTALS]
