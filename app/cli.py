"""vaultrag CLI.

    vaultrag seed                     load the demo corpus
    vaultrag ask "..." --user alice   ask as a specific person
    vaultrag eval demo/gold.json      measure leak rate and recall
    vaultrag health                   corpus freshness
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import psycopg
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .conflicts import corpus_freshness, detect_conflicts, detect_stale
from .db import init_schema
from .embeddings import get_embedder
from .evaluate import diff, load_gold, run_eval
from .generate import FakeLLM, GroqLLM, generate
from .ingest import Doc, ingest
from .retrieval import resolve_principal, search

console = Console()


def _llm(settings):
    if settings.llm == "groq" and settings.groq_api_key:
        return GroqLLM(settings.groq_api_key, settings.groq_model)
    return FakeLLM('{"answer": "[fake LLM] set LLM=groq and GROQ_API_KEY for real answers", "cited": [1], "conflict": false}')


async def _seed(args) -> int:
    settings = get_settings()
    await init_schema(settings.database_url)
    corpus = json.loads(Path(args.file).read_text())
    embedder = get_embedder(settings.embedder, settings.embed_model)

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        for u in corpus["users"]:
            await conn.execute(
                "INSERT INTO users (id,email,groups) VALUES (%s,%s,%s) "
                "ON CONFLICT (id) DO UPDATE SET groups=EXCLUDED.groups",
                (u["id"], u["email"], u["groups"]),
            )
        for d in corpus["documents"]:
            r = await ingest(conn, embedder, Doc(**d))
            for w in r.warnings:
                console.print(f"[yellow]warn[/] {w}")
        await conn.commit()
    console.print(
        f"seeded [bold]{len(corpus['documents'])}[/] documents, [bold]{len(corpus['users'])}[/] users"
    )
    return 0


async def _ask(args) -> int:
    settings = get_settings()
    embedder = get_embedder(settings.embedder, settings.embed_model)
    llm = _llm(settings)

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        principal = await resolve_principal(conn, args.user)
        if principal is None:
            console.print(f"[red]unknown user {args.user!r}[/]")
            return 1
        console.print(f"[dim]asking as {principal.user_id}, principals: {principal.principals}[/]\n")

        vec = embedder.embed([args.question])[0]
        hits = await search(conn, principal, args.question, vec, limit=args.limit)

        if not hits:
            console.print("[yellow]no permitted documents matched[/]")
            return 0

        table = Table(title="retrieved (only what this user may see)")
        table.add_column("doc"); table.add_column("title"); table.add_column("score", justify="right")
        for h in hits:
            table.add_row(h.doc_id, h.title, f"{h.score:.4f}")
        console.print(table)

        for c in detect_conflicts(hits):
            console.print(f"[yellow]conflict[/] {c.doc_a} vs {c.doc_b}: {c.reason}")
        for s in detect_stale(hits):
            console.print(f"[yellow]stale[/] {s.doc_id} last updated {s.age_days}d ago (owner: {s.owner})")

        answer = generate(llm, args.question, hits)
        console.print(f"\n[bold]{answer.text}[/]")
        if answer.citations:
            console.print("[dim]sources: " + ", ".join(c.doc_id for c in answer.citations) + "[/]")
        if not answer.answered:
            console.print(f"[dim]refused: {answer.refusal_reason}[/]")
    return 0


async def _eval(args) -> int:
    settings = get_settings()
    embedder = get_embedder(settings.embedder, settings.embed_model)
    llm = _llm(settings)
    cases = load_gold(args.gold)
    # A stub LLM cannot be judged on whether it refused correctly, so do not pretend to.
    llm_judged = settings.llm == "groq" and bool(settings.groq_api_key)

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        report = await run_eval(conn, embedder, llm, cases, label=args.label, llm_judged=llm_judged)

    table = Table(show_header=False, box=None)
    leak_colour = "red" if report.leak_rate > 0 else "green"
    table.add_row("cases", str(len(report.cases)))
    table.add_row("leak rate", f"[{leak_colour}]{report.leak_rate:.1%}[/]")
    table.add_row("mean recall", f"{report.mean_recall:.1%}")
    table.add_row("pass rate", f"{report.pass_rate:.1%}")
    if report.refusal_correctness is None:
        table.add_row("refusal correctness", "[dim]not measured (stub LLM)[/]")
    else:
        table.add_row("refusal correctness", f"{report.refusal_correctness:.1%}")
    console.print(table)

    for c in report.leaks:
        console.print(f"[red]LEAK[/] {c.case_id}: {c.user_id} retrieved {c.leaked}")

    if args.out:
        Path(args.out).write_text(report.to_json())
        console.print(f"[dim]wrote {args.out}[/]")

    # A leak is never an acceptable exit code.
    return 1 if (report.leak_rate > 0 and args.strict) else 0


async def _health(args) -> int:
    settings = get_settings()
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        h = await corpus_freshness(conn, args.max_age_days)
    table = Table(show_header=False, box=None)
    for k, v in h.items():
        colour = "yellow" if k in ("stale_documents", "orphaned_no_acl") and v else "white"
        table.add_row(k.replace("_", " "), f"[{colour}]{v}[/]")
    console.print(table)
    return 0


def _diff(args) -> int:
    from .evaluate import EvalReport, CaseResult

    def load(p):
        d = json.loads(Path(p).read_text())
        return EvalReport(label=d["label"], cases=[CaseResult(**c) for c in d["cases"]])

    d = diff(load(args.before), load(args.after))
    colour = {"LEAK": "red", "REGRESSION": "red", "IMPROVED": "green", "NO CHANGE": "dim"}[d["verdict"]]
    console.print(f"[{colour}][bold]{d['verdict']}[/][/]")
    console.print(f"  leak rate: {d['leak_rate']['before']:.1%} -> {d['leak_rate']['after']:.1%}")
    console.print(f"  mean recall: {d['mean_recall']['before']:.1%} -> {d['mean_recall']['after']:.1%}")
    for k in ("new_leaks", "regressed", "fixed"):
        if d[k]:
            console.print(f"  {k}: {', '.join(d[k])}")
    for r in d["recall_dropped"]:
        console.print(f"  [yellow]recall drop[/] {r['id']}: {r['before']:.0%} -> {r['after']:.0%}")
    return 1 if (d["verdict"] in ("LEAK", "REGRESSION") and args.strict) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vaultrag", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", help="load a corpus")
    s.add_argument("--file", default="demo/corpus.json")
    s.set_defaults(afunc=_seed)

    a = sub.add_parser("ask", help="ask a question as a user")
    a.add_argument("question")
    a.add_argument("--user", required=True)
    a.add_argument("--limit", type=int, default=5)
    a.set_defaults(afunc=_ask)

    e = sub.add_parser("eval", help="measure leak rate and recall against a gold set")
    e.add_argument("gold")
    e.add_argument("--label", default="current")
    e.add_argument("--out")
    e.add_argument("--strict", action="store_true", help="exit 1 on any leak")
    e.set_defaults(afunc=_eval)

    h = sub.add_parser("health", help="corpus freshness")
    h.add_argument("--max-age-days", type=int, default=365)
    h.set_defaults(afunc=_health)

    d = sub.add_parser("diff", help="compare two eval reports")
    d.add_argument("before"); d.add_argument("after")
    d.add_argument("--strict", action="store_true")
    d.set_defaults(func=_diff)

    args = p.parse_args(argv)
    if hasattr(args, "afunc"):
        return asyncio.run(args.afunc(args))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
