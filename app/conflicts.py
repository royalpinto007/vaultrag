"""Conflict and freshness signals.

Two things that quietly poison a document assistant, both of which are properties of the *corpus*
rather than the model, and therefore invisible to every prompt-level fix:

1. **Conflicts.** Two documents the user can see say different things. Picking one and sounding
   confident is the worst available behaviour, because the user cannot tell it happened. The
   correct move is to surface the disagreement and name who owns it.

2. **Staleness.** A document nobody has touched in two years is still retrievable and still
   answers with total confidence. Most "the AI gave me wrong information" incidents are really
   "the wiki was wrong and nobody noticed for a year".

Neither is a modelling problem. You cannot prompt your way out of a corpus that disagrees with
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .retrieval import Hit


@dataclass
class Conflict:
    doc_a: str
    doc_b: str
    reason: str
    owner_a: str | None = None
    owner_b: str | None = None


@dataclass
class StaleDoc:
    doc_id: str
    title: str
    age_days: int
    owner: str | None


def detect_conflicts(hits: list[Hit]) -> list[Conflict]:
    """Heuristic: retrieved chunks from different documents that both look authoritative.

    This is a signal, not a proof. Real semantic contradiction detection needs a model; what this
    does is cheap and catches the common shape: two official documents, from different sources,
    both answering the same question. That is worth a flag even when they happen to agree.
    """
    out: list[Conflict] = []
    official = [h for h in hits if h.is_official]

    for i, a in enumerate(official):
        for b in official[i + 1 :]:
            if a.doc_id == b.doc_id:
                continue
            out.append(
                Conflict(
                    doc_a=a.doc_id,
                    doc_b=b.doc_id,
                    reason="two official documents both answer this; they may disagree",
                    owner_a=a.owner,
                    owner_b=b.owner,
                )
            )

    # An official doc and an informal one both answering is the more common real case: someone's
    # notes contradicting the policy. The official one should win, and the user should know why.
    unofficial = [h for h in hits if not h.is_official]
    for a in official:
        for b in unofficial:
            if a.doc_id != b.doc_id:
                out.append(
                    Conflict(
                        doc_a=a.doc_id,
                        doc_b=b.doc_id,
                        reason="an official and an informal document both answer this; prefer the official one",
                        owner_a=a.owner,
                        owner_b=b.owner,
                    )
                )
                break  # one flag per official doc is enough; more is noise
    return out


def detect_stale(hits: list[Hit], max_age_days: int = 365) -> list[StaleDoc]:
    """Documents old enough that answering from them confidently is a risk."""
    now = datetime.now(timezone.utc)
    out: list[StaleDoc] = []
    seen: set[str] = set()
    for h in hits:
        if h.doc_id in seen:
            continue
        seen.add(h.doc_id)
        updated = h.updated_at
        if not isinstance(updated, datetime):
            continue
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age = (now - updated).days
        if age > max_age_days:
            out.append(StaleDoc(doc_id=h.doc_id, title=h.title, age_days=age, owner=h.owner))
    return out


async def corpus_freshness(conn, max_age_days: int = 365) -> dict:
    """Whole-corpus health. The thing to put on a dashboard and alert on.

    A retrieval system degrades silently as its corpus rots, and nobody notices because the answers
    keep sounding just as confident.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT count(*) FILTER (WHERE deleted_at IS NULL),
                   count(*) FILTER (WHERE deleted_at IS NULL AND updated_at < %s),
                   count(*) FILTER (WHERE deleted_at IS NULL AND owner IS NULL),
                   count(*) FILTER (WHERE deleted_at IS NOT NULL)
            FROM documents
            """,
            (cutoff,),
        )
        live, stale, ownerless, deleted = await cur.fetchone()

        # A document with no ACL rows is retrievable by nobody. Correct, but almost always a bug.
        await cur.execute(
            """
            SELECT count(*) FROM documents d
            WHERE d.deleted_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM doc_acl a WHERE a.doc_id = d.id)
            """
        )
        (orphaned,) = await cur.fetchone()

    return {
        "live_documents": live,
        "stale_documents": stale,
        "stale_pct": round(100 * stale / live, 1) if live else 0.0,
        "ownerless_documents": ownerless,
        "orphaned_no_acl": orphaned,
        "soft_deleted": deleted,
    }
