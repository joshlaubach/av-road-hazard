"""Measure written answers: 36 questions through the small Claude model, the big one, the cascade
that tries the small one first, and optionally the local Ollama model.

Usage: python -m tools.eval_answers --nuscenes-root data/nuscenes/raw [--version v1.0-mini]
           [--records records.json] [--device cuda] [--small claude-haiku-5-5]
           [--big claude-sonnet-5-5] [--llama llama3.2] [--out report.json]
Needs USE_STUBS=0, DEFECT_CHECKPOINT and ANTHROPIC_API_KEY in the environment (it spends a little
real money: about 36 short questions per Claude model). The records are the pipeline's own output on
the nuScenes front frames (cached like tools/eval_retrieval.py). Each question searches the 404
photos, the top 5 go to the model, and the answer is graded against those records
(eval/answers.py). The cascade is derived from the two Claude runs, not run a third time. The
report written to --out keeps per-question detail for debugging and stays on this machine; only the
aggregates go to eval/results.json (tools/build_results.py --add-answers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.answers import (
    Graded,
    Question,
    answers_section,
    cascade,
    expected_ids,
    grade,
    make_questions,
    summarize,
)
from rag.embed import ClipEmbedder
from rag.guards import cost_usd
from rag.index import CORPUS, Hit, Index
from rag.llm import LLM, ClaudeLLM, LLMUnavailable, OllamaLLM, answer_question, clean_question
from rag.text import describe
from tools.eval_detector import _load_frames, _rgb
from tools.eval_retrieval import build_records, pipeline_provenance

TOP_K = 5


def run_model(
    make: Any, questions: list[Question], hits: list[list[Hit]]
) -> tuple[list[Graded], list[dict[str, Any]]]:
    """Ask every question; `make(on_usage)` builds the model so each call's tokens can be priced."""
    graded: list[Graded] = []
    detail: list[dict[str, Any]] = []
    for question, found in zip(questions, hits, strict=True):
        usage: list[tuple[str, int, int]] = []
        llm: LLM = make(lambda m, i, o, usage=usage: usage.append((m, i, o)))
        cost = 0.0
        records = [h.record for h in found]
        expected = expected_ids(question, records)
        try:
            answer = answer_question(llm, question.text, found)
            cost = sum(cost_usd(m, i, o) for m, i, o in usage)
            graded.append(grade(expected, answer.cited, answer.unknown, cost_usd=cost))
            detail.append({"cited": list(answer.cited), "unknown": list(answer.unknown)})
        except LLMUnavailable as exc:
            cost = sum(cost_usd(m, i, o) for m, i, o in usage)
            graded.append(grade(expected, [], [], failed=True, cost_usd=cost))
            detail.append({"error": str(exc)})
    return graded, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="stop instead of re-running the pipeline when the cached records do not match",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--small", default="claude-haiku-5-5")
    parser.add_argument("--big", default="claude-sonnet-5-5")
    parser.add_argument("--llama", default=None, help="also run this local Ollama model")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from nuscenes.nuscenes import NuScenes

    prov = pipeline_provenance()
    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=False)
    frames = _load_frames(nusc)
    config = {"files": prov["files"], "thresholds": prov["thresholds"]}
    records = build_records(frames, args.version, config, args.records, rebuild=not args.no_rebuild)
    embedder = ClipEmbedder(args.device)
    images = embedder.embed_images([_rgb(path) for _, _, path, _ in frames])
    texts = embedder.embed_texts([describe(r) for r in records])
    index = Index(embedder.dim)
    index.add(records, images, texts)

    questions = make_questions()
    hits: list[list[Hit]] = []
    for question in questions:
        vector = embedder.embed_texts([clean_question(question.text)])[0]
        hits.append(index.search({"text": vector, "image": vector}, [CORPUS], top_k=TOP_K))

    runs: dict[str, tuple[list[Graded], list[dict[str, Any]]]] = {
        "small": run_model(lambda u: ClaudeLLM(args.small, on_usage=u), questions, hits),
        "big": run_model(lambda u: ClaudeLLM(args.big, on_usage=u), questions, hits),
    }
    if args.llama:
        runs["llama"] = run_model(lambda _: OllamaLLM(model=args.llama), questions, hits)
    chosen = cascade(runs["small"][0], runs["big"][0])
    configurations: dict[str, Any] = {
        f"{args.small} alone": summarize(runs["small"][0]),
        f"{args.big} alone": summarize(runs["big"][0]),
        f"{args.small} then {args.big}": summarize([g for g, _ in chosen], [e for _, e in chosen]),
    }
    if args.llama:
        configurations[f"{args.llama} (local)"] = summarize(runs["llama"][0])
    report: dict[str, Any] = {
        "top_k": TOP_K,
        "questions": len(questions),
        "configurations": configurations,
        "provenance": prov,
        "detail": {name: run[1] for name, run in runs.items()},  # stays local
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, "utf-8")
    print(json.dumps(answers_section(report)["configurations"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
