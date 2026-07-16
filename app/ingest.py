"""Ingestion: documents in, retrievable chunks out.

Two decisions worth defending here.

1. Chunking splits on structure (headings, then paragraphs), not on a fixed token count.
   Fixed-size chunks cut sentences in half and strand the subject of a paragraph away from its
   predicate, which then embeds badly and retrieves badly. Structure is a free signal that the
   document's author already provided; use it. Overlap exists to catch the case where the answer
   straddles a boundary anyway.

2. Reindexing is delete-then-insert per document, inside a transaction.
   The alternative (diff the chunks) is more efficient and much easier to get subtly wrong. A
   document's chunks are small and reindexing is rare; correctness wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import Embedder

# Chunk sizing, in characters rather than tokens: a token estimate would be another dependency and
# another thing to be wrong about. ~1200 chars is roughly 300 tokens for English prose.
_TARGET = 1200
_OVERLAP = 150
_MIN = 100  # below this, fold into the neighbour rather than emit a stub


@dataclass
class Doc:
    id: str
    title: str
    source: str
    text: str
    acl: list[str]  # principals: user ids and/or group names. Empty = visible to nobody.
    owner: str | None = None
    department: str | None = None
    url: str | None = None
    is_official: bool = False


@dataclass
class Chunk:
    position: int
    text: str
    heading: str | None = None


@dataclass
class IngestResult:
    doc_id: str
    chunks: int
    warnings: list[str] = field(default_factory=list)


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def chunk_text(text: str, target: int = _TARGET, overlap: int = _OVERLAP) -> list[Chunk]:
    """Split on markdown headings first, then pack paragraphs up to `target`.

    Each chunk carries the heading it lives under, so a retrieved fragment still says what it is
    about even when read in isolation.
    """
    sections = _split_headings(text)
    chunks: list[Chunk] = []
    pos = 0

    for heading, body in sections:
        for piece in _pack_paragraphs(body, target, overlap):
            chunks.append(Chunk(position=pos, text=piece, heading=heading))
            pos += 1
    return chunks


def _split_headings(text: str) -> list[tuple[str | None, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(None, text.strip())]

    sections: list[tuple[str | None, str]] = []
    # anything before the first heading still belongs to the document
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _pack_paragraphs(body: str, target: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    cur = ""

    for p in paras:
        # a single oversized paragraph gets hard-split; nothing else to do with it
        if len(p) > target:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_hard_split(p, target, overlap))
            continue

        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= target:
            cur = f"{cur}\n\n{p}"
        else:
            out.append(cur)
            # carry the tail of the previous chunk so a boundary-straddling answer survives
            cur = (_tail(cur, overlap) + "\n\n" + p) if overlap else p

    if cur:
        out.append(cur)

    # fold a runt trailing chunk into its predecessor
    if len(out) > 1 and len(out[-1]) < _MIN:
        out[-2] = out[-2] + "\n\n" + out.pop()
    return out


def _hard_split(p: str, target: int, overlap: int) -> list[str]:
    step = max(1, target - overlap)
    return [p[i : i + target] for i in range(0, len(p), step)]


def _tail(s: str, n: int) -> str:
    return s[-n:] if len(s) > n else s


async def ingest(conn, embedder: Embedder, doc: Doc) -> IngestResult:
    """Index one document and its ACL. Idempotent: re-ingesting replaces its chunks."""
    warnings: list[str] = []
    if not doc.acl:
        # Deny-by-default is the correct behaviour, but it is almost always a mistake by the
        # caller, so say so loudly rather than silently indexing something nobody can retrieve.
        warnings.append(
            f"document {doc.id!r} has an empty ACL and will be visible to nobody"
        )

    chunks = chunk_text(doc.text)
    if not chunks:
        warnings.append(f"document {doc.id!r} produced no chunks")
        return IngestResult(doc_id=doc.id, chunks=0, warnings=warnings)

    vectors = embedder.embed([c.text for c in chunks])

    async with conn.transaction():
        await conn.execute(
            """
            INSERT INTO documents (id, source, title, owner, department, url, is_official, updated_at, deleted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now(), NULL)
            ON CONFLICT (id) DO UPDATE SET
                source = EXCLUDED.source,
                title = EXCLUDED.title,
                owner = EXCLUDED.owner,
                department = EXCLUDED.department,
                url = EXCLUDED.url,
                is_official = EXCLUDED.is_official,
                updated_at = now(),
                deleted_at = NULL
            """,
            (doc.id, doc.source, doc.title, doc.owner, doc.department, doc.url, doc.is_official),
        )

        # ACL is replaced wholesale: a principal removed upstream must lose access here.
        await conn.execute("DELETE FROM doc_acl WHERE doc_id = %s", (doc.id,))
        for principal in dict.fromkeys(doc.acl):  # dedupe, preserve order
            await conn.execute(
                "INSERT INTO doc_acl (doc_id, principal) VALUES (%s, %s)", (doc.id, principal)
            )

        await conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc.id,))
        for c, vec in zip(chunks, vectors):
            await conn.execute(
                """
                INSERT INTO chunks (doc_id, position, heading, text, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (doc.id, c.position, c.heading, c.text, str(vec)),
            )

    return IngestResult(doc_id=doc.id, chunks=len(chunks), warnings=warnings)


async def soft_delete(conn, doc_id: str) -> None:
    """Retire a document from retrieval without losing the audit trail.

    Retrieval filters on deleted_at IS NULL, so this takes effect immediately. The rows stay so
    that "which documents did we show user X last March" still answers correctly.
    """
    await conn.execute("UPDATE documents SET deleted_at = now() WHERE id = %s", (doc_id,))
