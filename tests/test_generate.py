"""Generation guardrails.

Retrieval decides what the model may read. These tests are about what it may say.

The theme: an unverifiable answer is worse than no answer. Every test here asserts that vaultrag
refuses rather than guesses.
"""

from __future__ import annotations

from datetime import datetime

from app.generate import Answer, FakeLLM, generate
from app.retrieval import Hit


def _hit(chunk_id: int = 1, score: float = 0.5, title: str = "Handbook") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title=title,
        heading="Bonuses",
        text="The quarterly bonus is 10% of base salary.",
        score=score,
        updated_at=datetime(2026, 1, 1),
        is_official=True,
        owner="hr@corp.test",
        url=None,
    )


def test_no_permitted_documents_is_a_refusal_not_an_answer():
    """The user can see nothing relevant. The model is never called."""
    llm = FakeLLM('{"answer": "the bonus is 10%", "cited": [1], "conflict": false}')
    ans = generate(llm, "what is the bonus", [])
    assert ans.answered is False
    assert ans.refusal_reason == "no_permitted_documents"
    assert not ans.citations


def test_weak_evidence_refuses_even_though_retrieval_returned_something():
    """Retrieval returning rows is not proof the rows are relevant.

    This is the failure people miss: top-k always returns k things. If the best of them is barely
    related, answering from it produces a confident irrelevant answer.
    """
    llm = FakeLLM('{"answer": "the bonus is 10%", "cited": [1], "conflict": false}')
    ans = generate(llm, "what is the parental leave policy", [_hit(score=0.001)])
    assert ans.answered is False
    assert ans.refusal_reason == "weak_evidence"


def test_model_declining_is_respected():
    llm = FakeLLM("INSUFFICIENT_EVIDENCE")
    ans = generate(llm, "what is the bonus", [_hit()])
    assert ans.answered is False
    assert ans.refusal_reason == "model_declined_insufficient_evidence"


def test_hallucinated_citation_is_dropped():
    """The model cites source [7] when it was given one source.

    Asking a model to cite is not the same as it citing correctly. We check what it claimed
    against what we actually handed it.
    """
    llm = FakeLLM('{"answer": "the bonus is 10% [7]", "cited": [7], "conflict": false}')
    ans = generate(llm, "what is the bonus", [_hit()])
    assert ans.answered is False, "an answer whose only citation is invented must not be served"
    assert ans.refusal_reason == "no_verifiable_citation"
    assert 7 in ans.dropped_citations


def test_partially_hallucinated_citations_keep_the_real_ones():
    llm = FakeLLM('{"answer": "the bonus is 10%", "cited": [1, 9], "conflict": false}')
    ans = generate(llm, "what is the bonus", [_hit(chunk_id=1)])
    assert ans.answered is True
    assert [c.chunk_id for c in ans.citations] == [1]
    assert ans.dropped_citations == [9], "the invented citation is recorded, not silently ignored"


def test_unparseable_output_refuses_rather_than_guessing():
    """If we cannot parse it, we cannot verify it. Refuse."""
    llm = FakeLLM("I think the bonus is probably around 10 percent or so?")
    ans = generate(llm, "what is the bonus", [_hit()])
    assert ans.answered is False
    assert ans.refusal_reason == "unparseable_model_output"


def test_good_answer_is_served_with_verified_citations():
    llm = FakeLLM('{"answer": "The quarterly bonus is 10% of base [1].", "cited": [1], "conflict": false}')
    ans = generate(llm, "what is the bonus", [_hit(chunk_id=42)])
    assert ans.answered is True
    assert len(ans.citations) == 1
    assert ans.citations[0].chunk_id == 42
    assert ans.citations[0].doc_id == "doc-42"


def test_conflict_is_surfaced_not_resolved():
    """Two sources disagree. Saying so is the correct behaviour; picking one is not."""
    llm = FakeLLM(
        '{"answer": "Sources disagree: [1] says 10%, [2] says 15%.", "cited": [1, 2], "conflict": true}'
    )
    ans = generate(llm, "what is the bonus", [_hit(chunk_id=1), _hit(chunk_id=2)])
    assert ans.answered is True
    assert ans.conflict is True
    assert len(ans.citations) == 2


def test_json_in_a_code_fence_is_parsed():
    """Models wrap JSON in fences constantly. Refusing over formatting would be silly."""
    llm = FakeLLM('```json\n{"answer": "10% of base [1].", "cited": [1], "conflict": false}\n```')
    ans = generate(llm, "what is the bonus", [_hit()])
    assert ans.answered is True
    assert ans.text == "10% of base [1]."
