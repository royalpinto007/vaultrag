-- vaultrag schema
--
-- The design decision this whole project exists to demonstrate:
-- access control is a RETRIEVAL concern, not a post-processing concern.
--
-- The naive RAG pipeline is: retrieve top-k -> filter out what the user can't see -> generate.
-- That is a leak waiting to happen. By the time you filter, the wrong chunks are already in your
-- process, they can end up in logs, traces, or a prompt you assembled before the filter ran. And a
-- top-k of 5 that filters down to 1 silently degrades answer quality with no signal.
--
-- vaultrag filters INSIDE the retrieval query. A chunk the user cannot see is never selected,
-- never scored, never ranked, never logged. It cannot leak because it was never fetched.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- principals: users and the groups they belong to
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    groups      TEXT[] NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- documents + their ACLs
--
-- A "principal" is either a user id or a group name. A document is visible to a
-- user if any of its ACL principals intersects {user.id} UNION user.groups.
-- Deny-by-default: a document with no ACL rows is visible to nobody.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,           -- where it came from (wiki, hr, runbook...)
    title       TEXT NOT NULL,
    owner       TEXT,                    -- who to route questions to
    department  TEXT,
    url         TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_official BOOLEAN NOT NULL DEFAULT false,  -- official beats informal on conflict
    deleted_at  TIMESTAMPTZ              -- soft delete; excluded from retrieval
);

CREATE TABLE IF NOT EXISTS doc_acl (
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    principal   TEXT NOT NULL,           -- a user id or a group name
    PRIMARY KEY (doc_id, principal)
);

-- the index that makes the ACL join cheap
CREATE INDEX IF NOT EXISTS doc_acl_principal_idx ON doc_acl (principal);

-- ---------------------------------------------------------------------------
-- chunks: the retrievable unit
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position    INT NOT NULL,            -- ordinal within the document
    heading     TEXT,
    text        TEXT NOT NULL,
    embedding   vector(384),             -- all-MiniLM-L6-v2 dims
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (doc_id, position)
);

-- vector index for semantic search.
-- ivfflat needs data before it can be built well; for a demo corpus this is fine.
-- Cosine distance, because the embedder returns normalized vectors.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- keyword index for exact terms: policy names, error codes, ticket numbers.
-- This is why retrieval is hybrid: vectors are bad at exact tokens.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);

-- ---------------------------------------------------------------------------
-- audit: who asked what, and what was actually retrieved for them
--
-- This exists so that "did we ever leak document X to user Y" is a query, not an
-- archaeology project. It is the thing you want on the day someone asks.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS queries (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    question    TEXT NOT NULL,
    answered    BOOLEAN NOT NULL DEFAULT false,  -- false = refused / no evidence
    asked_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrievals (
    query_id    BIGINT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    chunk_id    BIGINT NOT NULL,
    doc_id      TEXT NOT NULL,
    score       REAL NOT NULL,
    cited       BOOLEAN NOT NULL DEFAULT false,  -- did it make it into the answer
    PRIMARY KEY (query_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS retrievals_doc_idx ON retrievals (doc_id);
