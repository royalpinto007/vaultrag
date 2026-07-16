"""Answer generation with citation verification.

Retrieval decides what the model is allowed to read. This file decides what it is allowed to say.

Three rules, in order of how often they are broken by RAG systems in the wild:

1. Answer only from the retrieved context. If the context does not support an answer, say so.
   A confident wrong answer is worse than "I don't know", especially at a company where the
   answer is a policy someone will act on.

2. Every claim carries a citation, and the citation is verified.
   Asking a model to cite its sources is not the same as it citing correctly. Models cite chunk 3
   while paraphrasing chunk 1. So we check the citations the model returned against the chunks we
   actually gave it, and we drop the ones it invented.

3. Conflicts are surfaced, not resolved silently.
   If two documents the user can see disagree, picking one and sounding confident is the worst
   outcome. Say they disagree, cite both, name the owner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .retrieval import Hit

# Below this fused RRF score, we treat retrieval as having found nothing useful. Tuned for the
# demo corpus; in a real deployment this is a number you set from an eval set, not a hunch.
_WEAK_EVIDENCE = 0.02


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass
class Citation:
    chunk_id: int
    doc_id: str
    title: str
    url: str | None
    owner: str | None


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    answered: bool = True
    refusal_reason: str | None = None
    conflict: bool = False
    dropped_citations: list[int] = field(default_factory=list)  # model cited these; we couldn't verify


_SYSTEM = """You answer questions using ONLY the provided context.

Rules:
- Use only the numbered sources below. Do not use outside knowledge.
- Every factual claim must cite the source it came from, like [2].
- If the sources do not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE
- If two sources disagree, say so explicitly, cite both, and do not pick a winner.
- Be concise. No preamble.

Reply as JSON: {"answer": "...", "cited": [1, 2], "conflict": true|false}"""


def _build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        label = f"[{i}] {h.title}"
        if h.heading:
            label += f" > {h.heading}"
        if h.is_official:
            label += " (official)"
        label += f" (updated {h.updated_at:%Y-%m-%d})" if hasattr(h.updated_at, "strftime") else ""
        blocks.append(f"{label}\n{h.text}")
    context = "\n\n".join(blocks)
    return f"Question: {question}\n\nSources:\n{context}"


def generate(llm: LLM, question: str, hits: list[Hit]) -> Answer:
    """Answer from `hits` only, then verify whatever the model claimed to cite."""
    if not hits:
        return Answer(
            text="I don't have any documents you have access to that cover this.",
            answered=False,
            refusal_reason="no_permitted_documents",
        )

    # Weak evidence is not the same as no evidence, but it should be treated the same way.
    # Retrieval returning something is not proof that the something is relevant.
    if max(h.score for h in hits) < _WEAK_EVIDENCE:
        return Answer(
            text="I found some documents but nothing that clearly answers this. "
            "Worth asking the document owner directly.",
            answered=False,
            refusal_reason="weak_evidence",
        )

    raw = llm.complete(_SYSTEM, _build_prompt(question, hits))
    parsed = _parse(raw)

    if parsed is None:
        # A model that returns unparseable output is a model whose answer we cannot verify.
        # Refusing is the honest response; retrying blind is how you ship a hallucination.
        return Answer(
            text="I could not produce a verifiable answer for this.",
            answered=False,
            refusal_reason="unparseable_model_output",
        )

    text, cited_idx, conflict = parsed

    if text.strip() == "INSUFFICIENT_EVIDENCE" or "INSUFFICIENT_EVIDENCE" in text:
        return Answer(
            text="The documents you have access to don't answer this.",
            answered=False,
            refusal_reason="model_declined_insufficient_evidence",
        )

    # Verify: the model can only cite sources we actually handed it. Anything else it invented.
    citations: list[Citation] = []
    dropped: list[int] = []
    for idx in cited_idx:
        if 1 <= idx <= len(hits):
            h = hits[idx - 1]
            citations.append(
                Citation(chunk_id=h.chunk_id, doc_id=h.doc_id, title=h.title, url=h.url, owner=h.owner)
            )
        else:
            dropped.append(idx)

    # An answer with no surviving citation is an unsourced claim. Do not serve it as fact.
    if not citations:
        return Answer(
            text="I found relevant documents but couldn't tie an answer to a specific source.",
            answered=False,
            refusal_reason="no_verifiable_citation",
            dropped_citations=dropped,
        )

    return Answer(
        text=text.strip(),
        citations=citations,
        answered=True,
        conflict=conflict,
        dropped_citations=dropped,
    )


def _parse(raw: str) -> tuple[str, list[int], bool] | None:
    """Pull JSON out of a model response, tolerating code fences and stray prose."""
    candidate = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", candidate, re.S)
        if brace:
            candidate = brace.group(0)

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        # Last resort: the model ignored the format but may still have declined honestly.
        if "INSUFFICIENT_EVIDENCE" in raw:
            return ("INSUFFICIENT_EVIDENCE", [], False)
        return None

    if not isinstance(data, dict) or "answer" not in data:
        return None

    answer = str(data.get("answer", ""))
    cited = data.get("cited", [])
    if not isinstance(cited, list):
        cited = []
    idx = [int(c) for c in cited if isinstance(c, (int, str)) and str(c).strip().isdigit()]
    return (answer, idx, bool(data.get("conflict", False)))


class FakeLLM:
    """Scripted LLM for tests. No network, no key, deterministic."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return self._response


class GroqLLM:
    """Groq free tier. Chosen because it is free and fast, not because it is special."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, system: str, user: str) -> str:
        from groq import Groq  # lazy: importing this module shouldn't require the SDK

        client = Groq(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,  # this is a retrieval task, not a creative one
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
