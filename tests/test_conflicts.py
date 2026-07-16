"""Tests for the corpus-health signals.

These checks fire on a corpus that disagrees with itself or has rotted. Both are the kind of
problem a team tries to solve with a better prompt for six months before realising the wiki was
wrong the whole time.

The discipline here is the same as everywhere else in this repo: a checker that cries wolf gets
switched off, so roughly half of these tests exist to prove it stays quiet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.conflicts import corpus_freshness, detect_conflicts, detect_stale
from app.embeddings import FakeEmbedder
from app.ingest import Doc, ingest
from app.retrieval import Hit


def _hit(doc_id, *, official=True, owner="a@corp.test", age_days=1, title=None) -> Hit:
    return Hit(
        chunk_id=hash(doc_id) % 10_000,
        doc_id=doc_id,
        title=title or doc_id.replace("-", " ").title(),
        heading="Bonus",
        text="The quarterly bonus payout policy is 10% of base.",
        score=0.03,
        updated_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        is_official=official,
        owner=owner,
        url=None,
    )


# ------------------------------------------------------------------------------------ conflicts


def test_two_official_documents_answering_the_same_question_is_flagged():
    """Not because they definitely disagree, but because nobody can tell whether they do. Silently
    picking one and sounding confident is the failure this exists to prevent."""
    conflicts = detect_conflicts([_hit("hr-policy"), _hit("finance-policy")])
    assert len(conflicts) == 1
    assert {conflicts[0].doc_a, conflicts[0].doc_b} == {"hr-policy", "finance-policy"}


def test_a_conflict_names_the_owners_so_a_human_can_resolve_it():
    """"These two documents disagree" is not actionable. "Ask these two people" is."""
    conflicts = detect_conflicts(
        [_hit("hr-policy", owner="chro@corp.test"), _hit("finance-policy", owner="cfo@corp.test")]
    )
    assert {conflicts[0].owner_a, conflicts[0].owner_b} == {"chro@corp.test", "cfo@corp.test"}


def test_official_versus_informal_is_flagged_with_a_preference():
    """The common real case: someone's stale notes contradicting the actual policy."""
    conflicts = detect_conflicts([_hit("hr-policy"), _hit("random-notes", official=False)])
    assert len(conflicts) == 1
    assert "prefer the official one" in conflicts[0].reason


def test_chunks_from_one_document_are_not_a_conflict_with_themselves():
    """A document is allowed to have two relevant paragraphs. Flagging that would fire on nearly
    every query and train everyone to ignore the flag."""
    assert detect_conflicts([_hit("hr-policy"), _hit("hr-policy")]) == []


def test_a_single_source_of_truth_produces_no_conflict():
    assert detect_conflicts([_hit("hr-policy")]) == []


def test_two_informal_documents_are_not_flagged_as_a_policy_conflict():
    """Neither claims authority, so there is no authority to be in conflict."""
    assert detect_conflicts([_hit("notes-a", official=False), _hit("notes-b", official=False)]) == []


# ------------------------------------------------------------------------------------ staleness


def test_an_old_document_is_flagged_with_its_age_and_owner():
    stale = detect_stale([_hit("old-handbook", age_days=900)])
    assert len(stale) == 1
    assert stale[0].age_days == 900
    assert stale[0].owner == "a@corp.test"


def test_a_recent_document_is_not_flagged():
    assert detect_stale([_hit("current-policy", age_days=10)]) == []


def test_the_staleness_threshold_is_a_caller_decision():
    """What counts as stale is a business question. A legal policy from 2019 may be perfectly
    current; a pricing page from last quarter is not."""
    hits = [_hit("doc", age_days=100)]
    assert detect_stale(hits, max_age_days=365) == []
    assert len(detect_stale(hits, max_age_days=30)) == 1


def test_each_stale_document_is_reported_once_not_once_per_chunk():
    assert len(detect_stale([_hit("old", age_days=900), _hit("old", age_days=900)])) == 1


# ------------------------------------------------------------------------------- corpus freshness


@pytest.mark.asyncio
async def test_freshness_counts_the_corpus(corpus, conn):
    health = await corpus_freshness(conn)
    assert health["live_documents"] > 0
    assert health["stale_documents"] == 0, "the fixture corpus was just ingested"


@pytest.mark.asyncio
async def test_a_document_with_no_acl_is_reported_as_orphaned(conn):
    """Retrievable by nobody. Technically the safest possible document, and almost always someone
    who forgot the ACL and will file a bug saying search is broken."""
    await ingest(
        conn,
        FakeEmbedder(),
        Doc(id="orphan", title="Orphan", source="drive", acl=[], text="# Bonus\nNobody sees this.\n"),
    )
    health = await corpus_freshness(conn)
    assert health["orphaned_no_acl"] == 1


@pytest.mark.asyncio
async def test_ingesting_without_an_acl_warns_loudly(conn):
    """The warning is the whole defence. An empty ACL is silent by nature: no error, no results."""
    result = await ingest(
        conn, FakeEmbedder(), Doc(id="q", title="Q", source="drive", acl=[], text="# H\nbody\n")
    )
    assert any("empty ACL" in w for w in result.warnings)
