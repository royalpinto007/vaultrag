"""vaultrag HTTP API.

The trust boundary lives here. Everything below this file assumes the caller is already
authenticated; this file is where we decide who the caller actually is.

Note what the /ask request body does NOT contain: groups. The caller states who it is, and the
server looks up what that means. If a client could send its own group list, the ACL would be
decorative and every test in tests/test_acl.py would be theatre.

Auth here is a header, because auth is not what this project is demonstrating. In a real
deployment this is your OIDC/JWT middleware, and the only thing that changes is where `user_id`
comes from. The important part, that authorization is resolved server-side and enforced in the
retrieval query, does not change.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .db import init_schema, make_pool
from .embeddings import get_embedder
from .generate import FakeLLM, GroqLLM, generate
from .ingest import Doc, ingest, soft_delete
from .retrieval import resolve_principal, search


# --------------------------------------------------------------------------- models
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    # deliberately no `groups` field. see the module docstring.


class CitationOut(BaseModel):
    doc_id: str
    title: str
    chunk_id: int
    url: str | None = None
    owner: str | None = None


class AskResponse(BaseModel):
    answer: str
    answered: bool
    citations: list[CitationOut] = []
    conflict: bool = False
    refusal_reason: str | None = None
    query_id: int


class DocIn(BaseModel):
    id: str
    title: str
    source: str
    text: str
    acl: list[str] = Field(description="user ids and/or group names. Empty means nobody can see it.")
    owner: str | None = None
    department: str | None = None
    url: str | None = None
    is_official: bool = False


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    warnings: list[str] = []


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_schema(settings.database_url)
    pool = make_pool(settings.database_url)
    await pool.open()
    app.state.pool = pool
    app.state.embedder = get_embedder(settings.embedder, settings.embed_model)
    app.state.llm = _make_llm(settings)
    yield
    await pool.close()


def _make_llm(settings: Settings):
    if settings.llm == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("LLM=groq but GROQ_API_KEY is not set")
        return GroqLLM(settings.groq_api_key, settings.groq_model)
    return FakeLLM('{"answer": "fake", "cited": [1], "conflict": false}')


app = FastAPI(
    title="vaultrag",
    description="Permission-aware RAG. Access control is enforced in the retrieval query, not after it.",
    version="0.1.0",
    lifespan=lifespan,
)


async def current_user(x_user_id: str = Header(...)) -> str:
    """Stand-in for real auth. Replace with OIDC/JWT; the rest of the system does not care."""
    if not x_user_id.strip():
        raise HTTPException(status_code=401, detail="missing user")
    return x_user_id.strip()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, user_id: str = Depends(current_user)) -> AskResponse:
    """Answer a question using only documents this user is permitted to see."""
    pool = app.state.pool
    async with pool.connection() as conn:
        principal = await resolve_principal(conn, user_id)
        if principal is None:
            # Unknown user gets nothing. Failing closed matters: a "helpful" default that treats an
            # unknown principal as a groupless-but-valid user is one careless change away from
            # treating it as an admin.
            raise HTTPException(status_code=403, detail="unknown user")

        # Audit first, so the record exists even if generation blows up. An audit log that only
        # captures successful queries is not an audit log.
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO queries (user_id, question) VALUES (%s, %s) RETURNING id",
                (principal.user_id, req.question),
            )
            query_id = (await cur.fetchone())[0]

        vec = app.state.embedder.embed([req.question])[0]
        hits = await search(conn, principal, req.question, vec, limit=req.limit)

        for h in hits:
            await conn.execute(
                """INSERT INTO retrievals (query_id, chunk_id, doc_id, score)
                   VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (query_id, h.chunk_id, h.doc_id, h.score),
            )

        answer = generate(app.state.llm, req.question, hits)

        cited_ids = {c.chunk_id for c in answer.citations}
        if cited_ids:
            await conn.execute(
                "UPDATE retrievals SET cited = true WHERE query_id = %s AND chunk_id = ANY(%s)",
                (query_id, list(cited_ids)),
            )
        await conn.execute(
            "UPDATE queries SET answered = %s WHERE id = %s", (answer.answered, query_id)
        )
        await conn.commit()

    return AskResponse(
        answer=answer.text,
        answered=answer.answered,
        conflict=answer.conflict,
        refusal_reason=answer.refusal_reason,
        query_id=query_id,
        citations=[
            CitationOut(doc_id=c.doc_id, title=c.title, chunk_id=c.chunk_id, url=c.url, owner=c.owner)
            for c in answer.citations
        ],
    )


@app.post("/documents", response_model=IngestResponse, status_code=201)
async def create_document(doc: DocIn) -> IngestResponse:
    """Index a document. Re-posting the same id replaces its chunks and ACL."""
    pool = app.state.pool
    async with pool.connection() as conn:
        result = await ingest(
            conn,
            app.state.embedder,
            Doc(
                id=doc.id,
                title=doc.title,
                source=doc.source,
                text=doc.text,
                acl=doc.acl,
                owner=doc.owner,
                department=doc.department,
                url=doc.url,
                is_official=doc.is_official,
            ),
        )
        await conn.commit()
    return IngestResponse(doc_id=result.doc_id, chunks=result.chunks, warnings=result.warnings)


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str) -> None:
    """Soft delete: unreachable from retrieval immediately, audit trail preserved."""
    pool = app.state.pool
    async with pool.connection() as conn:
        await soft_delete(conn, doc_id)
        await conn.commit()


@app.get("/audit/document/{doc_id}")
async def audit_document(doc_id: str) -> dict:
    """Who has this document ever been surfaced to?

    This endpoint is the reason the audit tables exist. On the day someone asks "was this leaked",
    you want a query, not an archaeology project.
    """
    pool = app.state.pool
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT q.user_id, q.question, q.asked_at, r.cited
                FROM retrievals r
                JOIN queries q ON q.id = r.query_id
                WHERE r.doc_id = %s
                ORDER BY q.asked_at DESC
                LIMIT 100
                """,
                (doc_id,),
            )
            rows = await cur.fetchall()
    return {
        "doc_id": doc_id,
        "exposures": [
            {"user_id": r[0], "question": r[1], "asked_at": r[2].isoformat(), "cited": r[3]}
            for r in rows
        ],
    }
