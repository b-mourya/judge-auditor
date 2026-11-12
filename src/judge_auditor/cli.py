"""Command line entry point: `judge-audit --judge simulated`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audits, report
from .data import load_pairs
from .judges import AnthropicJudge, CachedJudge, Judge, OpenAIJudge, SimulatedJudge


def build_judge(args, pairs) -> Judge:
    if args.judge == "simulated":
        return SimulatedJudge(pairs, noise=args.sim_noise,
                              position_bias=args.sim_position_bias, length_bias=args.sim_length_bias)
    if args.judge == "anthropic":
        judge = AnthropicJudge(args.model) if args.model else AnthropicJudge()
    elif args.judge == "openai":
        if not args.model:
            sys.exit("--model is required for --judge openai")
        judge = OpenAIJudge(args.model)
    else:
        raise ValueError(args.judge)
    return judge if args.no_cache else CachedJudge(judge)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit an LLM-as-judge for agreement, position bias and verbosity bias.")
    ap.add_argument("--judge", choices=["simulated", "anthropic", "openai"], default="simulated")
    ap.add_argument("--model", help="Model name for real judges")
    ap.add_argument("--data", default="data/sample_pairs.jsonl")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for confidence intervals")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--sim-noise", type=float, default=0.15)
    ap.add_argument("--sim-position-bias", type=float, default=0.25)
    ap.add_argument("--sim-length-bias", type=float, default=0.6)
    args = ap.parse_args(argv)

    pairs = load_pairs(args.data)
    judge = build_judge(args, pairs)
    print(f"Auditing {judge.name} on {len(pairs)} pairs...", file=sys.stderr)

    progress = lambda i, n: print(f"  judged {i}/{n}", end="\r", file=sys.stderr)  # noqa: E731
    js = audits.judge_both_orders(judge, pairs, progress)
    print(file=sys.stderr)

    agreement = audits.agreement_audit(js, args.bootstrap)
    position = audits.position_audit(js, args.bootstrap)
    verbosity = audits.verbosity_audit(judge, pairs, js, args.bootstrap)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = report.render(judge.name, args.data, agreement, position, verbosity)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({
        "judge": judge.name, "dataset": args.data, "agreement": agreement,
        "position": position, "verbosity": verbosity,
        "judgments": [j.__dict__ for j in js],
    }, indent=2), encoding="utf-8")
    print(md)
    print(f"\nSaved {out / 'report.md'} and {out / 'results.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
