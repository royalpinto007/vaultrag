"""Test fixtures.

Tests run against a real Postgres with pgvector (docker compose up -d db), because the ACL
enforcement lives in SQL. Mocking the database would mean mocking the thing under test, which
would make these tests worthless: they would pass while the real query leaked.
"""

from __future__ import annotations

import os

import psycopg
import pytest
import pytest_asyncio

from app.db import reset_schema
from app.embeddings import FakeEmbedder
from app.ingest import Doc, ingest

DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql://vaultrag:vaultrag@localhost:5433/vaultrag"
)


@pytest.fixture(scope="session")
def embedder():
    # Deterministic and offline. These tests are about access control, not semantics.
    return FakeEmbedder()


@pytest_asyncio.fixture
async def conn():
    await reset_schema(DATABASE_URL)
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as c:
        yield c


@pytest_asyncio.fixture
async def corpus(conn, embedder):
    """A small corpus with deliberately overlapping content across permission boundaries.

    The overlap is the point. Every document talks about "the quarterly bonus payout policy", so
    a naive retriever asked about bonuses would happily surface all of them. Only the ACL should
    decide who sees which. If access control were bolted on after retrieval, these tests would
    catch it.
    """
    await conn.execute(
        "INSERT INTO users (id, email, groups) VALUES (%s, %s, %s)",
        ("alice", "alice@corp.test", ["engineering"]),
    )
    await conn.execute(
        "INSERT INTO users (id, email, groups) VALUES (%s, %s, %s)",
        ("bob", "bob@corp.test", ["sales"]),
    )
    await conn.execute(
        "INSERT INTO users (id, email, groups) VALUES (%s, %s, %s)",
        ("carol", "carol@corp.test", ["hr", "engineering"]),
    )
    await conn.execute(
        "INSERT INTO users (id, email, groups) VALUES (%s, %s, %s)",
        ("dave", "dave@corp.test", []),  # no groups: should see only doc-level grants
    )

    docs = [
        Doc(
            id="eng-handbook",
            title="Engineering Handbook",
            source="wiki",
            acl=["engineering"],
            department="engineering",
            text="# Bonuses\nThe quarterly bonus payout policy for engineering is 10% of base.\n",
        ),
        Doc(
            id="sales-comp",
            title="Sales Compensation",
            source="wiki",
            acl=["sales"],
            department="sales",
            text="# Bonuses\nThe quarterly bonus payout policy for sales is commission based.\n",
        ),
        Doc(
            id="hr-salaries",
            title="HR Salary Bands",
            source="hr",
            acl=["hr"],
            department="hr",
            is_official=True,
            text="# Bonuses\nThe quarterly bonus payout policy and every individual salary band.\n",
        ),
        Doc(
            id="ceo-private",
            title="CEO Private Notes",
            source="drive",
            acl=["ceo"],  # nobody in this corpus is in the ceo group
            text="# Bonuses\nThe quarterly bonus payout policy is under review. Confidential.\n",
        ),
        Doc(
            id="all-hands",
            title="All Hands Notes",
            source="wiki",
            acl=["engineering", "sales", "hr"],
            text="# Bonuses\nThe quarterly bonus payout policy will be discussed next month.\n",
        ),
        Doc(
            id="dave-direct",
            title="Onboarding For Dave",
            source="wiki",
            acl=["dave"],  # granted to a user id, not a group
            text="# Bonuses\nYour quarterly bonus payout policy is explained by your manager.\n",
        ),
        Doc(
            id="orphan",
            title="Orphaned Doc",
            source="drive",
            acl=[],  # empty ACL: deny by default
            text="# Bonuses\nThe quarterly bonus payout policy draft nobody owns.\n",
        ),
    ]
    for d in docs:
        await ingest(conn, embedder, d)
    return docs
