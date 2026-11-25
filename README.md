# open-data-store

Created and maintained by [Marc Jeulin](https://github.com/jeulinmarc), part of [MyOpenFund](https://github.com/MyOpenFund).

Bitemporal open-data store for MyOpenFund: dataset **contracts**, a light **writer** for producers,
a **publisher** (the only writer to the lake) and a **reader** for analyses, the MCP and strategies.
Facts are DuckLake tables (Parquet + a Postgres catalogue), read through DuckDB. Every fact carries
`valid_date` (what it is about) and `known_at` (when it became knowable): reads are look-ahead-free.

> This library is a tool. It manages nobody's money and gives no investment advice.

Renamed on 2026-11-22 (package, import, command and environment variables): see
[docs/MIGRATION.md](https://github.com/MyOpenFund/open-data-store/blob/main/docs/MIGRATION.md).

## Install

```
pip install "myopenfund-open-data-store[writer]"     # producers: pyarrow only
pip install "myopenfund-open-data-store[publisher]"  # store operator: DuckDB + DuckLake
pip install "myopenfund-open-data-store[reader]"     # analyses, MCP, strategies
pip install "myopenfund-open-data-store[recipes]"    # market-data recipes (exchange calendars)
```

If this environment installed the old distribution (`open-data-store`, from a git URL), run
`pip uninstall -y open-data-store` before installing `myopenfund-open-data-store`: both own the same import package.

To try the latest `main` before a release (dependencies still come from PyPI):

```
pip install "myopenfund-open-data-store[reader] @ git+https://github.com/MyOpenFund/open-data-store"
```

**Working across repos**

Clone the repos side by side and install this package editable over the PyPI version in the
consumer's environment: `pip install -e ../open-data-store`. A consumer change that needs an
unreleased version waits for this package's release before the consumer's PR merges.

**How to release**

Bump `__version__` and `version` in a PR. After it merges, go to Actions, then release, then Run workflow: pick the `main` branch and target `pypi`. Approve the `pypi` deployment. The workflow builds the package, checks it, publishes it, then creates the tag `v<version>` and the GitHub Release itself.
It fails before publishing if it is not run on the default branch or if that tag already exists. Target `testpypi` is a rehearsal: it publishes to TestPyPI and creates no tag and no release.
Pushing a tag `v<version>` by hand still works: the workflow checks that the tag equals the package version and is on the default branch, then publishes and attaches the files to the Release.
If PyPI got the package but the tag or the Release is missing, re-run the failed `github-release` job from Actions; do not push the tag by hand (the re-run would fail at the PyPI upload).

## Configuration

| Variable | Meaning |
|---|---|
| `OPEN_DATA_STORE_DATA_ROOT` | Holds `landing/` (batches), `lake/` (Parquet), `reports/` |
| `OPEN_DATA_STORE_CATALOG` | DuckLake catalogue: `postgres:dbname=... host=... user=...`, or a file path (tests, single user) |
| `OPEN_DATA_STORE_DATA_INLINING_ROW_LIMIT` | Default `0`: every batch becomes a Parquet file |
| `OPEN_DATA_STORE_KNOWN_AT_FUTURE_TOLERANCE_S` | Clock-skew tolerance, default `300` |
| `OPEN_DATA_STORE_RESOLVING_SCHEMES` | Identifier schemes that may merge entities, default `cik,lei,isin,figi,cb_code,national_id` |
| `OPEN_DATA_STORE_SINGLE_VALUED_SCHEMES` | Schemes an entity holds one value of, default `cik,lei,cb_code` |

## Reader interface (stable for v0.x)

```python
from datetime import datetime, timezone
from open_data_store.reader import Recipe, Store, export

store = Store.open()                                   # config from env (or `with Store.open() as store:`)
store.datasets()                                       # name, major, version, kind, licence, table
t = store.read_asof("fundamentals", datetime(2025, 12, 1, tzinfo=timezone.utc),
                    where="original_tag = 'us-gaap:Revenues'")  # pyarrow.Table, latest known per key
t1 = store.read_asof("fundamentals", filters={"source_code": ["sec"]},
                     columns=["entity_id", "original_tag", "value"], limit=100)  # bound, safe for servers
store.read_asof("fundamentals", filters={"entity_value": ["0000320193"]}, vintages=True)  # every vintage
store.freshness()                                      # last published_at, batch count
store.describe("fundamentals")                         # columns, docs, natural key, licence
t0 = store.read_asof("fundamentals", snapshot=42)      # what the store held at DuckLake version 42
r = Recipe("debt_by_source", "1.0.0", {"f": "fundamentals"},
           "SELECT source_code, sum(value) AS debt FROM f GROUP BY 1")
out = store.assemble(r, as_of=datetime.now(timezone.utc))  # save=True arrives with metrics
# {"f": "fundamentals:vintages"} gives every row knowable at the date, not the latest per key
export(out, "debt.xlsx")                               # .csv | .xlsx | .parquet
```

**Time zone.** The reader's sessions are pinned to UTC: timestamps come back as UTC, and `where`
and recipe SQL evaluate in UTC whatever the process `TZ`. Under a non-UTC `TZ`, earlier versions
evaluated them in local time, so a naive literal (`known_at < '2025-07-31'`) or a date cast can
select other rows now; write explicit offsets (`TIMESTAMPTZ '2025-07-31 00:00:00+02'`).

**Performance.** Filters on natural-key columns (e.g. `entity_value`) are always pushed explicitly
before the as-of window, so DuckLake reads only the matching files: on a 20-issuer test store a
filter on `entity_value` read 1 fundamentals data file (5 kB) instead of 60 files (139 kB). DuckDB
already pushes a filter on the as-of window's partition key (the natural key) through the window by
itself (filtering `entity_value` outside the window also read 1 file / 5 kB); the explicit push in
`read_asof` guarantees that pruning whatever the optimizer version. Filters on other columns read
the whole table. Keep one batch per issuer, and do not merge data files
(`ducklake_merge_adjacent_files`) without measuring: merging issuers into one file defeats the
pruning.

## Market data

Six standard contracts (v1, all `fact`, partitioned by `source_code`): `prices` (raw OHLCV per
provider series and session), `corporate_actions` (splits, dividends, spin-offs, as filed),
`vendor_total_return` (a vendor's own total-return series: a control, and the documented fallback
where corporate actions are missing; never mixed into `prices`),
`index_snapshots` (index membership per snapshot date), `instrument_identifiers` (dated mappings
of provider symbols and FIGI/ISIN/CIK to an opaque `mof:` instrument id) and `fetch_log` (what was
fetched, with its status). Approve them on a store before producers publish.

**Source registry.** Sources are declared through the `open_data_store.sources` entry-point group
(`SourceInfo`: licence, terms URL, attribution, `access` open or restricted, and the flags
`no_public_display` and `no_ml_training`; `code` is lowercase, `[a-z][a-z0-9_]{0,62}`).
Restricted sources are denied by default to every served channel. The gate
(`sources.filter_sources`) is enforced by `load_market_inputs` and by the serving layers (the
MCP); `Store.read_asof` is **not** gated, so a caller that serves its rows filters them first,
before any column projection. Market data is read in strict mode: a `source_code` missing from
the registry is denied, and a `source_symbol` identifier row is denied when the source of its
`id_value` prefix is denied. Outside strict mode a `source_code` that is not registered is open.

**Recipes** are pure functions over Arrow tables (extra `pip install "myopenfund-open-data-store[recipes]"`).
Signatures, from the submodules `market.membership`, `market.returns`, `market.coverage`,
`market.asof` (`asof`, `latest`, `sessions`) and `market.store`:

```python
membership_intervals(snapshots, identifiers, *, index_code, as_of, staleness_days=45, precedence=())
unresolved_members(snapshots, identifiers, *, index_code, as_of, staleness_days=45, precedence=())
resolve_prices(prices, identifiers, *, as_of, precedence)
resolve_actions(actions, identifiers, *, as_of, precedence)
total_return(prices, actions, sessions, *, reinvest_at="ex_date_close")      # resolved inputs
coverage(intervals, prices_resolved, sessions, fetch_coverage=None, *, unresolved=None,
         max_missing_share=0.02, min_delisting_gap_sessions=5)               # -> CoverageResult
delisting_exits(intervals, prices_resolved, sessions, *, rule="last_close", fixed_return=None,
                post_prices=None, fetch_coverage=None, min_delisting_gap_sessions=5)
load_market_inputs(store, *, as_of, allowed_restricted=frozenset(),
                   identifiers_as_of=None)                                   # -> MarketInputs
```

`load_market_inputs` (`open_data_store.market.store`) reads each contract at its own major
(`prices_v1`, ...), prices, actions, snapshots and the fetch log as-of and the identifiers latest,
applies the strict licence gate (restricted sources dropped unless allowed, unregistered sources
always dropped) and drops `entity_id`/`batch_id`. `identifiers_as_of` (default now, UTC) pins the identifier read to
reproduce a load. A dataset approved but not published yet, or not approved, gives an empty table
with its contract schema. Only the latest vintage knowable at `as_of` is kept per key: call the
recipes with the same `as_of` as the load.

Recipe outputs name instruments by `instrument_id` only: `resolve_prices` and `resolve_actions`
keep `source_code` (provenance) but no provider symbol (`source_symbol`, `spinoff_symbol`).

Coverage reasons: `ok`, `price_invalid`, `not_fetched`, `unmapped`, `source_has_no_data`,
`recycled_at_source`, `delisted`, `session_not_published`. Total-return flags: `currency_mismatch`,
`currency_change`, `spin_off_unvalued`, `rights_unvalued`, `action_invalid`, `gap`,
`price_invalid`.

Caller contracts:

- **Pass `unresolved=unresolved_members(...)` to `coverage`.** Without it, coverage ignores
  unmapped index members and can pass on a universe with holes. With it, each unresolved symbol
  counts as a missing member (`per_session.unmapped`) on every session of
  `[first_seen, valid_to)`.
- `fetch_coverage` callers must drop superseded `error`/`partial` fetch rows: a failed row vetoes
  a delisting whatever its order.
- A restated snapshot that omits a symbol leaves the old row visible: producers emit
  `is_member = false` on restatement.
- Adapters must fill the currency of every cash action, or the cash is dropped (flagged
  `currency_mismatch`).
- Two symbols of one source mapped to one instrument on one date (overlapping mapping windows,
  e.g. a venue change): `resolve_prices` / `resolve_actions` keep their rows once when every
  value is identical, and raise otherwise. Make each source's mapping windows disjoint.

## Producers

```python
from open_data_store.writer import write_batch
ref = write_batch(contract, table, landing_root=data_root / "landing",
                  run_id=run_id, producer="company-corpus", source_inputs=[...])
```

The writer validates the table against its contract and drops an atomic, idempotent batch in
`landing/`. It never touches Postgres. A producer declares its own contracts through the
`open_data_store.contracts` entry-point group; ten standard contracts ship with this library
(`fundamentals`, `entity_observations`, `filing_statements`, `filings`, and the six market
contracts of "Market data").

## Operator

```
open-data-store contracts list
open-data-store contracts approve fundamentals 3 --by <name>        # reviewed step; refuses conflicting definitions
open-data-store contracts approve filing_statements 1 --by <name>   # likewise, for a store that publishes it
open-data-store contracts approve filings 1 --by <name>             # likewise
open-data-store publish                                             # exit 0 ok · 1 fatal · 3 degraded
```

`publish` audits every landed batch again (file checksum, contract, recomputed batch id),
quarantines failures in `landing/_quarantine/` with a `reason.json`, publishes the rest atomically
(one DuckLake snapshot per batch, commit message = batch id) and writes a run report under
`reports/publisher/`. A batch that hits an infrastructure error (listed under `errors`, exit 1)
stays in `landing/` for the next run, and so do the fact batches behind a failed identity batch,
instead of being quarantined as `unknown_entity`. Those deferrals are `errors` entries too (no
separate field): their `error` text starts with `deferred:`. Design, security model and gotchas: [docs/ARCHITECTURE.md](https://github.com/MyOpenFund/open-data-store/blob/main/docs/ARCHITECTURE.md).

Producers may re-emit full history: rows already published with the same values (compared
exactly, so re-emit bit-identical floats) are skipped and counted (`skipped_rows` in the report,
`rows_skipped` in `ref.batches`, where `rows` counts only the inserted rows); the same key and
`known_at` with another value quarantines the batch (`conflicting_republish`). Columns of the
major's first approved definition are compared strictly; columns added by a later minor are compared
only where the published row holds a value (changing it is a conflict), and filling a NULL in one
needs a later `known_at` to be recorded.

## Licence

MIT (code). Each dataset carries its own data licence in its contract.
