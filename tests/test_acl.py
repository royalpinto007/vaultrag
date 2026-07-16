"""The restricted-access suite.

This is the reason vaultrag exists. Every test here asks a question that is answerable from
documents the user is not allowed to see, and asserts those documents never come back.

Every document in the corpus contains the phrase "quarterly bonus payout policy". So a retriever
with no access control, or with access control applied after ranking, would surface all of them
for every user. The only thing standing between alice and the CEO's private notes is the ACL
predicate inside the retrieval query.

If someone refactors retrieval.py and moves the filter out of SQL into Python, these tests fail.
That is the point of them.
"""

from __future__ import annotations

import pytest

from app.retrieval import resolve_principal, search

pytestmark = pytest.mark.asyncio

QUESTION = "what is the quarterly bonus payout policy"


async def _search_as(conn, embedder, user_id: str, question: str = QUESTION, limit: int = 20):
    principal = await resolve_principal(conn, user_id)
    assert principal is not None, f"user {user_id} should exist"
    vec = embedder.embed([question])[0]
    hits = await search(conn, principal, question, vec, limit=limit)
    return {h.doc_id for h in hits}


async def test_group_isolation_alice_cannot_see_sales_or_hr(conn, corpus, embedder):
    docs = await _search_as(conn, embedder, "alice")
    assert "eng-handbook" in docs, "alice is in engineering and must see the eng handbook"
    assert "sales-comp" not in docs, "LEAK: alice retrieved a sales-only document"
    assert "hr-salaries" not in docs, "LEAK: alice retrieved an HR-only document"
    assert "ceo-private" not in docs, "LEAK: alice retrieved the CEO's private notes"


async def test_group_isolation_bob_cannot_see_engineering_or_hr(conn, corpus, embedder):
    docs = await _search_as(conn, embedder, "bob")
    assert "sales-comp" in docs
    assert "eng-handbook" not in docs, "LEAK: bob retrieved an engineering-only document"
    assert "hr-salaries" not in docs, "LEAK: bob retrieved an HR-only document"


async def test_multi_group_user_sees_union_not_everything(conn, corpus, embedder):
    """carol is in hr AND engineering, so she sees both, but still not sales or the CEO."""
    docs = await _search_as(conn, embedder, "carol")
    assert {"hr-salaries", "eng-handbook"} <= docs
    assert "sales-comp" not in docs, "LEAK: carol is not in sales"
    assert "ceo-private" not in docs, "LEAK: being in two groups is not being in every group"


async def test_nobody_sees_the_ceo_document(conn, corpus, embedder):
    """No user in this corpus is in the ceo group, so the document is unreachable for all of them."""
    for user in ("alice", "bob", "carol", "dave"):
        docs = await _search_as(conn, embedder, user)
        assert "ceo-private" not in docs, f"LEAK: {user} retrieved the CEO's private notes"


async def test_empty_acl_is_deny_by_default(conn, corpus, embedder):
    """A document with no ACL rows is visible to nobody, not to everybody.

    This is the failure mode that turns a missing config into a data breach: fail open, and an
    ingestion bug silently publishes the document to the whole company.
    """
    for user in ("alice", "bob", "carol", "dave"):
        docs = await _search_as(conn, embedder, user)
        assert "orphan" not in docs, f"LEAK: {user} retrieved a document with an empty ACL"


async def test_user_level_grant_works_without_any_group(conn, corpus, embedder):
    """dave has no groups. He should still see a document granted directly to his user id."""
    docs = await _search_as(conn, embedder, "dave")
    assert "dave-direct" in docs, "a direct user grant must work without group membership"
    assert docs == {"dave-direct"}, f"dave should see exactly one document, saw {docs}"


async def test_shared_document_reaches_every_permitted_group(conn, corpus, embedder):
    for user in ("alice", "bob", "carol"):
        docs = await _search_as(conn, embedder, user)
        assert "all-hands" in docs, f"{user} should see the all-hands doc"
    assert "all-hands" not in await _search_as(conn, embedder, "dave"), (
        "LEAK: dave is in none of the granted groups"
    )


async def test_revoking_acl_takes_effect_immediately(conn, corpus, embedder):
    """Access is evaluated per query, not baked into the index at ingest time.

    If revocation required reindexing, every ACL change would be a race between the policy and the
    pipeline. It does not.
    """
    assert "eng-handbook" in await _search_as(conn, embedder, "alice")

    await conn.execute(
        "DELETE FROM doc_acl WHERE doc_id = %s AND principal = %s", ("eng-handbook", "engineering")
    )

    docs = await _search_as(conn, embedder, "alice")
    assert "eng-handbook" not in docs, "LEAK: revoked access still retrievable"


async def test_removing_user_from_group_takes_effect_immediately(conn, corpus, embedder):
    assert "hr-salaries" in await _search_as(conn, embedder, "carol")

    await conn.execute("UPDATE users SET groups = %s WHERE id = %s", (["engineering"], "carol"))

    docs = await _search_as(conn, embedder, "carol")
    assert "hr-salaries" not in docs, "LEAK: user removed from hr still sees hr documents"
    assert "eng-handbook" in docs, "carol should keep the access she still has"


async def test_soft_deleted_document_is_unreachable(conn, corpus, embedder):
    from app.ingest import soft_delete

    assert "eng-handbook" in await _search_as(conn, embedder, "alice")
    await soft_delete(conn, "eng-handbook")
    docs = await _search_as(conn, embedder, "alice")
    assert "eng-handbook" not in docs, "LEAK: deleted document still retrievable"


async def test_unknown_user_resolves_to_nothing(conn, corpus, embedder):
    """An unknown principal gets no principal object, not an empty-but-valid one.

    Failing closed here matters: a bug that produced Principal(user_id="", groups=()) would match
    nothing today, but it is one careless default away from matching everything.
    """
    assert await resolve_principal(conn, "mallory") is None


async def test_groups_are_not_taken_from_the_caller(conn, corpus, embedder):
    """Sanity check on the trust boundary.

    resolve_principal reads groups from the database. There is no code path that lets a caller
    assert its own group membership. This test documents that intent: if someone adds a
    `groups` parameter to the API later, they have to delete this test to do it.
    """
    principal = await resolve_principal(conn, "dave")
    assert principal.groups == (), "dave has no groups in the database"
    assert principal.principals == ["dave"]
