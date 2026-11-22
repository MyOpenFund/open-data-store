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
| contracts' `owner` field `my-open-data` (third column of `contracts list`; display only, not fingerprinted) | `open-data-store` |
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
  reports carry no package name. Two cosmetic traces of the old name stay where they were written:
  the DuckLake snapshots committed before the rename have the commit author
  `my-open-data publisher` (new ones `open-data-store publisher`; nothing reads it), and files
  exported earlier with `export()` keep the `myopendata.*` schema metadata keys. The Postgres
  roles (`lake_publisher`, `lake_reader`) and the database name (`OPEN_DATA_STORE_CATALOG_DB`,
  e.g. `lake_catalog`) are unchanged.

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
   `OPEN_DATA_STORE_REF` to a commit at or after the rename (a commit from before it still
   installs as `my-open-data` and pip refuses the name mismatch). The catalogue lives in the
   stack's `pgdata` volume, whose name starts with the Compose project (stack) name: renaming the
   stack gives it a new, empty volume. Either keep the stack name, or move the catalogue:
   1. On the old stack, after its last publish, take a dump that holds it: start the stack once
      with an empty `landing/` (its `backup` runs after the previous publish), or run
      `docker compose run --rm backup`. The dump taken at the start of a publishing run does not
      hold that run. Stop the old stack; do not delete it or its volume yet.
   2. Create the new stack with the renamed `.env`. Keep `OPEN_DATA_STORE_HOST_DIR` pointing at
      the same files (move the directory first if you change it).
   3. Before its first full start, restore (`deploy/nas/README.md`, restore):
      `docker compose up -d --wait catalog`, `docker compose run --rm init`, then
      `docker compose run --rm backup sh -c 'pg_restore -d "$PGDATABASE" /data/backups/<dump>'`.
   4. Start the new stack with an empty `landing/`: `contracts list` shows the same fingerprints,
      `publish` finds nothing to do, and a `lake_reader` read gives the same row counts as before.
   5. Only then delete the old stack, and its `pgdata` volume once you no longer need a way back.
