# Migration: my-open-data → open-data-store

On 2026-11-22 the project was renamed from `my-open-data` to `open-data-store`, everywhere, with no
compatibility layer: code written for the old names fails at import until it is updated. The data is
not affected (see "What does not change").

## Names

| Before | After |
|---|---|
| repository `MyOpenFund/my-open-data` (GitHub redirects the old URL) | `MyOpenFund/open-data-store` |
| distribution `my-open-data` (`my-open-data[writer\|publisher\|reader]`) | `open-data-store` (`open-data-store[writer\|publisher\|reader]`) |
| import `myopendata` (`from myopendata.reader import Store`) | `open_data_store` (`from open_data_store.reader import Store`) |
| command `myopendata` (`myopendata publish`, `myopendata contracts list`) | `open-data-store` (`open-data-store publish`, …) |
| entry-point group `myopendata.contracts` | `open_data_store.contracts` |
| environment variables `MYOPENDATA_*` | `OPEN_DATA_STORE_*` (same suffix, list below) |
| reader schema metadata keys `myopendata.as_of`, `myopendata.snapshot`, `myopendata.recipe` | `open_data_store.as_of`, `open_data_store.snapshot`, `open_data_store.recipe` |
| contracts' `owner` field `my-open-data` (display only) | `open-data-store` |
| `deploy/nas/.env.example` store directory `/srv/my_open_data` | `/srv/open_data_store` (an example: keep your own path if you like) |

Environment variables, old suffix = new suffix after `MYOPENDATA_` / `OPEN_DATA_STORE_`:
library `DATA_ROOT`, `CATALOG`, `DATA_INLINING_ROW_LIMIT`, `KNOWN_AT_FUTURE_TOLERANCE_S`,
`RESOLVING_SCHEMES`, `SINGLE_VALUED_SCHEMES`; tests `TEST_PG`; stack (`deploy/nas/.env`) `REF`,
`HOST_DIR`, `UID_GID`, `PG_BIND`, `CATALOG_DB`, `ADMIN_PASSWORD`, `PUBLISHER_PASSWORD`,
`READER_PASSWORD`, `APPROVE`, `APPROVED_BY`.

## What does not change

- **Contract fingerprints.** They hash the contract fields only; `open-data-store contracts list`
  prints the same fingerprints as `myopendata contracts list` did. Existing approvals stay valid and
  batches written by either version publish identically.
- **Stores.** Catalogue tables, Parquet files, batch ids, the landing layout and the publisher
  reports carry no package name. The Postgres roles (`lake_publisher`, `lake_reader`) and the
  database name (`OPEN_DATA_STORE_CATALOG_DB`, e.g. `lake_catalog`) are unchanged.

## Upgrading

1. **Uninstall the old distribution first**, in every environment that had it:
   `pip uninstall -y my-open-data`, then install `open-data-store[...]`. pip sees two different
   distributions; the old one would otherwise stay installed and importable.
2. Replace imports `myopendata` → `open_data_store`, the command `myopendata` →
   `open-data-store`, and the `MYOPENDATA_` prefix → `OPEN_DATA_STORE_` in environments, `.env`
   files and CI.
3. Producers that declare their own contracts register them under the entry-point group
   `open_data_store.contracts`. Under the old group they are no longer discovered.
4. Code that reads the reader's schema metadata reads the `open_data_store.*` keys.
5. A running Compose stack (`deploy/nas/`): rename every `.env` key `MYOPENDATA_*` →
   `OPEN_DATA_STORE_*` (values unchanged), use the new `compose.yaml`, and set
   `OPEN_DATA_STORE_REF` to a commit of `open-data-store`. The catalogue lives in the stack's
   `pgdata` volume, whose name starts with the Compose project (stack) name: renaming the stack
   gives it a new, empty volume. Either keep the stack name, or dump before and `pg_restore` into
   the new stack before its first publish (`deploy/nas/README.md`, restore).
