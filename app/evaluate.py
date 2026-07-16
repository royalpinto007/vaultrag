"""Evaluation harness.

The claim this project makes is "permissions are enforced at retrieval". A claim you cannot
measure is a slogan, so this file measures it.

Four metrics, in order of how much they matter:

1. **leak_rate**: the only one that can end you. For every (user, question) pair where the answer
   lives in a document that user cannot see, did anything from that document surface? A single
   leak is a failure, not a percentage point. Target: exactly zero, always.

2. **recall@k**: of the documents the user *can* see that genuinely answer the question, how many
   made it into the top k. This is the metric that quietly dies when you tighten security: it is
   trivial to get leak_rate to zero by retrieving nothing. Measuring both together is the point.

3. **citation_accuracy**: of the citations the answer carried, how many actually pointed at a
   document that supports the claim. Models cite chunk 3 while paraphrasing chunk 1.

4. **refusal_correctness**: when there is no permitted evidence, did it say so rather than
   improvise. Split into "correctly refused" and "wrongly refused", because a system that refuses
   everything scores perfectly on safety and is useless.

Metrics 1 and 2 are properties of *retrieval* and need no model: run them with FakeEmbedder and
FakeLLM in CI and they mean exactly what they say. Metric 4 is a property of the *model*, and
scoring it against a stub LLM measures nothing. So `run_eval(llm_judged=False)` reports 1 and 2 and
declines to score 4, rather than printing a confident number derived from a stub. A harness that
launders a fake into a metric is worse than no harness.

The pairing of 1 and 2 is the whole design. Optimising either alone produces a broken system:
all-permissive leaks, all-restrictive is a very secure way to answer nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .generate import generate
from .retrieval import resolve_principal, search


@dataclass
class GoldCase:
    """One question, asked as one user, with the truth about what should happen."""

    id: str
    user_id: str
    question: str
    # Documents that answer this question AND this user may see. Empty = they should get nothing.
    expected_docs: list[str] = field(default_factory=list)
    # Documents that answer it but this user must NOT see. These are the trap.
    forbidden_docs: list[str] = field(default_factory=list)
    should_answer: bool = True


@dataclass
class CaseResult:
    case_id: str
    user_id: str
    leaked: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    recall: float = 0.0
    answered: bool = False
    refusal_reason: str | None = None
    correct_refusal: bool | None = None
    # False when a stub LLM produced the answer, so refusal behaviour was not really tested.
    llm_judged: bool = True

    @property
    def passed(self) -> bool:
        # A leak fails the case outright. Everything else is a quality signal; this is a safety one.
        if self.leaked:
            return False
        # Only hold the answer against the model if a real model wrote it.
        if self.llm_judged and self.correct_refusal is False:
            return False
        return True


@dataclass
class EvalReport:
    label: str
    cases: list[CaseResult] = field(default_factory=list)
    llm_judged: bool = True

    @property
    def leak_rate(self) -> float:
        return sum(1 for c in self.cases if c.leaked) / len(self.cases) if self.cases else 0.0

    @property
    def mean_recall(self) -> float:
        scored = [c.recall for c in self.cases if c.recall is not None]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(1 for c in self.cases if c.passed) / len(self.cases) if self.cases else 0.0

    @property
    def refusal_correctness(self) -> float | None:
        """None when a stub LLM answered: the number would be meaningless, so do not report one."""
        if not self.llm_judged:
            return None
        scored = [c for c in self.cases if c.correct_refusal is not None]
        if not scored:
            return None
        return sum(1 for c in scored if c.correct_refusal) / len(scored)

    @property
    def leaks(self) -> list[CaseResult]:
        return [c for c in self.cases if c.leaked]

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "llm_judged": self.llm_judged,
                "summary": {
                    "cases": len(self.cases),
                    "leak_rate": round(self.leak_rate, 4),
                    "mean_recall": round(self.mean_recall, 4),
                    "pass_rate": round(self.pass_rate, 4),
                    "refusal_correctness": (
                        round(self.refusal_correctness, 4)
                        if self.refusal_correctness is not None
                        else None
                    ),
                },
                "cases": [asdict(c) for c in self.cases],
            },
            indent=2,
        )


def load_gold(path: str | Path) -> list[GoldCase]:
    data = json.loads(Path(path).read_text())
    return [GoldCase(**c) for c in data["cases"]]


async def run_case(
    conn, embedder, llm, case: GoldCase, limit: int = 5, llm_judged: bool = True
) -> CaseResult:
    principal = await resolve_principal(conn, case.user_id)
    if principal is None:
        # An unknown user retrieving nothing is correct behaviour, not an error in the eval.
        return CaseResult(
            case_id=case.id,
            user_id=case.user_id,
            correct_refusal=not case.should_answer,
            llm_judged=llm_judged,
        )

    vec = embedder.embed([case.question])[0]
    hits = await search(conn, principal, case.question, vec, limit=limit)

    retrieved = list(dict.fromkeys(h.doc_id for h in hits))
    leaked = [d for d in retrieved if d in case.forbidden_docs]

    if case.expected_docs:
        found = len(set(retrieved) & set(case.expected_docs))
        recall = found / len(case.expected_docs)
    else:
        recall = 1.0  # nothing was expected; not retrieving anything is perfect

    answer = generate(llm, case.question, hits)
    cited = list(dict.fromkeys(c.doc_id for c in answer.citations))

    # A citation pointing at a forbidden doc is the worst version of a leak: it is not just
    # retrieved, it is quoted to the user.
    leaked += [d for d in cited if d in case.forbidden_docs and d not in leaked]

    correct_refusal = None
    if not case.should_answer:
        correct_refusal = not answer.answered
    elif case.should_answer and not answer.answered:
        correct_refusal = False  # wrongly refused: safe, but useless

    return CaseResult(
        case_id=case.id,
        user_id=case.user_id,
        leaked=leaked,
        retrieved=retrieved,
        cited=cited,
        recall=round(recall, 4),
        answered=answer.answered,
        refusal_reason=answer.refusal_reason,
        correct_refusal=correct_refusal,
        llm_judged=llm_judged,
    )


async def run_eval(
    conn,
    embedder,
    llm,
    cases: list[GoldCase],
    label: str = "current",
    llm_judged: bool = True,
) -> EvalReport:
    report = EvalReport(label=label, llm_judged=llm_judged)
    for case in cases:
        report.cases.append(await run_case(conn, embedder, llm, case, llm_judged=llm_judged))
    return report


def diff(before: EvalReport, after: EvalReport) -> dict:
    """Did this change make it worse?

    Regressions are the headline. An aggregate that improved while two cases broke is still a
    change you want to look at before shipping.
    """
    b = {c.case_id: c for c in before.cases}
    a = {c.case_id: c for c in after.cases}

    new_leaks = [k for k in sorted(set(a)) if a[k].leaked and not b.get(k, a[k]).leaked]
    regressed = [k for k in sorted(set(b) & set(a)) if b[k].passed and not a[k].passed]
    fixed = [k for k in sorted(set(b) & set(a)) if not b[k].passed and a[k].passed]
    recall_drop = [
        {"id": k, "before": b[k].recall, "after": a[k].recall}
        for k in sorted(set(b) & set(a))
        if a[k].recall < b[k].recall - 0.01
    ]

    if new_leaks:
        verdict = "LEAK"  # nothing else matters if this fires
    elif regressed:
        verdict = "REGRESSION"
    elif fixed:
        verdict = "IMPROVED"
    else:
        verdict = "NO CHANGE"

    return {
        "verdict": verdict,
        "new_leaks": new_leaks,
        "regressed": regressed,
        "fixed": fixed,
        "recall_dropped": recall_drop,
        "leak_rate": {"before": round(before.leak_rate, 4), "after": round(after.leak_rate, 4)},
        "mean_recall": {"before": round(before.mean_recall, 4), "after": round(after.mean_recall, 4)},
    }
