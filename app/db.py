"""Database connection and schema management."""

from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg_pool import AsyncConnectionPool

_SCHEMA = Path(__file__).parent / "schema.sql"


def make_pool(database_url: str) -> AsyncConnectionPool:
    """A connection pool. Opened lazily so importing this module does not require a live DB."""
    return AsyncConnectionPool(database_url, min_size=1, max_size=10, open=False)


async def init_schema(database_url: str) -> None:
    """Create extensions and tables. Idempotent, safe to run on every boot."""
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await conn.execute(_SCHEMA.read_text())


async def reset_schema(database_url: str) -> None:
    """Drop everything and recreate. Tests only, never call this against anything you care about."""
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await conn.execute(
            """
            DROP TABLE IF EXISTS retrievals, queries, chunks, doc_acl, documents, users CASCADE;
            """
        )
        await conn.execute(_SCHEMA.read_text())
