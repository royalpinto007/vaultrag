"""End-to-end API tests.

The ACL tests prove the retrieval query is safe. These prove the HTTP layer does not undo that:
that identity comes from the server, that an unknown caller gets nothing, and that the audit trail
records what actually happened.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("EMBEDDER", "fake")
os.environ.setdefault("LLM", "fake")
os.environ.setdefault(
    "DATABASE_URL",
    os.getenv("TEST_DATABASE_URL", "postgresql://vaultrag:vaultrag@localhost:5433/vaultrag"),
)

from app.db import reset_schema  # noqa: E402
from app.main import app  # noqa: E402

pytestmark = pytest.mark.asyncio

DATABASE_URL = os.environ["DATABASE_URL"]


@pytest_asyncio.fixture
async def client():
    await reset_schema(DATABASE_URL)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # lifespan doesn't run under ASGITransport, so wire the app state by hand
        from app.config import get_settings
        from app.db import make_pool
        from app.embeddings import get_embedder
        from app.generate import FakeLLM

        settings = get_settings()
        pool = make_pool(settings.database_url)
        await pool.open()
        app.state.pool = pool
        app.state.embedder = get_embedder("fake")
        app.state.llm = FakeLLM(
            '{"answer": "The quarterly bonus is 10% of base [1].", "cited": [1], "conflict": false}'
        )
        try:
            yield c
        finally:
            await pool.close()


async def _seed(client):
    async with client._transport.app.state.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO users (id, email, groups) VALUES (%s,%s,%s)",
            ("alice", "alice@corp.test", ["engineering"]),
        )
        await conn.execute(
            "INSERT INTO users (id, email, groups) VALUES (%s,%s,%s)",
            ("bob", "bob@corp.test", ["sales"]),
        )
        await conn.commit()

    await client.post(
        "/documents",
        json={
            "id": "eng-handbook",
            "title": "Engineering Handbook",
            "source": "wiki",
            "acl": ["engineering"],
            "text": "# Bonuses\nThe quarterly bonus payout policy for engineering is 10% of base.\n",
        },
    )
    await client.post(
        "/documents",
        json={
            "id": "ceo-private",
            "title": "CEO Private Notes",
            "source": "drive",
            "acl": ["ceo"],
            "text": "# Bonuses\nThe quarterly bonus payout policy is confidential.\n",
        },
    )


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200


async def test_ask_requires_identity(client):
    r = await client.post("/ask", json={"question": "what is the bonus"})
    assert r.status_code == 422, "no user header should be rejected, not defaulted"


async def test_unknown_user_is_denied_not_treated_as_groupless(client):
    await _seed(client)
    r = await client.post(
        "/ask", json={"question": "bonus"}, headers={"X-User-Id": "mallory"}
    )
    assert r.status_code == 403, "an unknown principal must fail closed"


async def test_ask_returns_only_permitted_documents(client):
    await _seed(client)
    r = await client.post(
        "/ask",
        json={"question": "quarterly bonus payout policy"},
        headers={"X-User-Id": "alice"},
    )
    assert r.status_code == 200
    body = r.json()
    cited = {c["doc_id"] for c in body["citations"]}
    assert "ceo-private" not in cited, "LEAK: CEO notes surfaced over the API"


async def test_user_with_no_access_gets_a_refusal_not_a_leak(client):
    """bob is in sales. Neither seeded document is his, so he should be told nothing was found,
    rather than quietly receiving somebody else's document."""
    await _seed(client)
    r = await client.post(
        "/ask", json={"question": "quarterly bonus payout policy"}, headers={"X-User-Id": "bob"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is False
    assert body["refusal_reason"] == "no_permitted_documents"
    assert body["citations"] == []


async def test_groups_cannot_be_asserted_by_the_caller(client):
    """Sending a groups field must not grant access. Pydantic ignores unknown fields, so this
    documents the intent: identity is resolved server-side, full stop."""
    await _seed(client)
    r = await client.post(
        "/ask",
        json={"question": "quarterly bonus payout policy", "groups": ["ceo", "hr"]},
        headers={"X-User-Id": "bob"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is False, "LEAK: caller escalated privileges via the request body"
    assert body["citations"] == []


async def test_audit_records_what_was_retrieved(client):
    await _seed(client)
    await client.post(
        "/ask",
        json={"question": "quarterly bonus payout policy"},
        headers={"X-User-Id": "alice"},
    )
    r = await client.get("/audit/document/eng-handbook")
    assert r.status_code == 200
    exposures = r.json()["exposures"]
    assert len(exposures) >= 1
    assert exposures[0]["user_id"] == "alice"


async def test_audit_shows_no_exposure_for_a_document_nobody_can_see(client):
    await _seed(client)
    for user in ("alice", "bob"):
        await client.post(
            "/ask", json={"question": "quarterly bonus payout policy"}, headers={"X-User-Id": user}
        )
    r = await client.get("/audit/document/ceo-private")
    assert r.json()["exposures"] == [], "LEAK: CEO doc was retrieved for someone"


async def test_empty_acl_document_warns_on_ingest(client):
    r = await client.post(
        "/documents",
        json={"id": "orphan", "title": "Orphan", "source": "drive", "acl": [], "text": "# X\nbody\n"},
    )
    assert r.status_code == 201
    assert any("empty ACL" in w for w in r.json()["warnings"]), (
        "deny-by-default is correct but silent denial is a support ticket"
    )


async def test_deleted_document_stops_being_retrievable(client):
    await _seed(client)
    r = await client.post(
        "/ask",
        json={"question": "quarterly bonus payout policy"},
        headers={"X-User-Id": "alice"},
    )
    assert r.json()["answered"] is True

    await client.delete("/documents/eng-handbook")

    r = await client.post(
        "/ask",
        json={"question": "quarterly bonus payout policy"},
        headers={"X-User-Id": "alice"},
    )
    assert r.json()["answered"] is False, "LEAK: deleted document still answerable"
