"""Hybrid retrieval with access control enforced in the query itself.

The one idea in this file:

    A chunk the user is not allowed to see is never SELECTed.

Not "fetched then filtered". Not "ranked then trimmed". Never selected. The ACL join is part of
the same query that does the vector search and the keyword search, so an unauthorized chunk cannot
reach the process at all, and therefore cannot reach a log, a trace, a prompt, or an answer.

Why hybrid: vector search is good at "what's our policy on remote work" and bad at "ERR_4021" or
"Policy 7.3". Keyword search is the opposite. Real questions contain both, so we run both and fuse
the rankings.
"""

from __future__ import annotations

from dataclasses import dataclass

from psycopg.rows import dict_row


@dataclass(frozen=True)
class Principal:
    """Who is asking. Their id plus every group they belong to.

    This is the complete set of ACL principals we will match against, and it is resolved from the
    database rather than trusted from the request. A caller cannot claim to be in a group.
    """

    user_id: str
    groups: tuple[str, ...]

    @property
    def principals(self) -> list[str]:
        return [self.user_id, *self.groups]


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    heading: str | None
    text: str
    score: float
    updated_at: object
    is_official: bool
    owner: str | None
    url: str | None


# Reciprocal Rank Fusion: combine two ranked lists without needing their scores to be comparable.
# score = sum over lists of 1/(k + rank). k=60 is the value from the original paper and is a
# reasonable default; it damps the influence of any single list's top hit.
_RRF_K = 60


async def search(
    conn,
    principal: Principal,
    query_text: str,
    query_embedding: list[float],
    limit: int = 5,
    candidates: int = 50,
) -> list[Hit]:
    """Hybrid ACL-scoped search.

    Args:
        conn: an open async psycopg connection.
        principal: resolved user + groups. Nothing outside this set is retrievable.
        query_text: the raw question, for keyword search.
        query_embedding: the question embedded, for vector search.
        limit: how many chunks to return.
        candidates: how many to consider from each arm before fusion.

    Returns:
        Up to `limit` chunks the user is allowed to see, best first.
    """
    # A chunk is visible iff its document has an ACL row whose principal is one of ours.
    # EXISTS rather than JOIN so a document with several matching ACL rows yields one chunk, not N.
    #
    # This CTE is referenced by BOTH arms of the search. That is the point: there is no code path
    # in this function that can see an unauthorized chunk, because both arms start from here.
    sql = """
    WITH visible AS (
        SELECT c.id, c.doc_id, c.heading, c.text, c.embedding, c.tsv,
               d.title, d.updated_at, d.is_official, d.owner, d.url
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE d.deleted_at IS NULL
          AND EXISTS (
              SELECT 1 FROM doc_acl a
              WHERE a.doc_id = d.id
                AND a.principal = ANY(%(principals)s)
          )
    ),
    vec AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(embedding)s::vector) AS rank
        FROM visible
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(candidates)s
    ),
    kw AS (
        SELECT id, ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %(q)s)) DESC
               ) AS rank
        FROM visible
        WHERE tsv @@ websearch_to_tsquery('english', %(q)s)
        ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %(q)s)) DESC
        LIMIT %(candidates)s
    ),
    fused AS (
        SELECT COALESCE(vec.id, kw.id) AS id,
               COALESCE(1.0 / (%(k)s + vec.rank), 0)
             + COALESCE(1.0 / (%(k)s + kw.rank), 0) AS score
        FROM vec
        FULL OUTER JOIN kw ON kw.id = vec.id
    )
    SELECT v.id AS chunk_id, v.doc_id, v.title, v.heading, v.text,
           f.score, v.updated_at, v.is_official, v.owner, v.url
    FROM fused f
    JOIN visible v ON v.id = f.id
    ORDER BY f.score DESC,
             -- tie-break the way a human would: official first, then most recently updated.
             -- This is the cheap version of conflict handling: when two sources say different
             -- things, the newer official one should be the one the model reads first.
             v.is_official DESC,
             v.updated_at DESC
    LIMIT %(limit)s
    """
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            sql,
            {
                "principals": principal.principals,
                "embedding": str(query_embedding),
                "q": query_text,
                "candidates": candidates,
                "limit": limit,
                "k": _RRF_K,
            },
        )
        rows = await cur.fetchall()

    return [
        Hit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            title=r["title"],
            heading=r["heading"],
            text=r["text"],
            score=float(r["score"]),
            updated_at=r["updated_at"],
            is_official=r["is_official"],
            owner=r["owner"],
            url=r["url"],
        )
        for r in rows
    ]


async def resolve_principal(conn, user_id: str) -> Principal | None:
    """Load a user's groups from the database.

    Deliberately not taking groups from the request. If the caller could assert its own groups,
    the ACL would be decorative.
    """
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT id, groups FROM users WHERE id = %s", (user_id,))
        row = await cur.fetchone()
    if row is None:
        return None
    return Principal(user_id=row["id"], groups=tuple(row["groups"] or ()))
