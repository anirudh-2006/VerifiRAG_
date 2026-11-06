#!/usr/bin/env python3
"""Cost / latency price tag for standalone graders vs cascade.

We do *not* have per-call API RTT in the frozen artifacts. This script reports
what we can measure or estimate honestly:

  - Symbolic wall time: timed locally on the 29 labeled queries (CPU, $0).
  - LLM call counts: exact, from frozen verdicts (judge-alone vs cascade).
  - Token estimates: reconstruct the judge prompt template per human step;
    output size from frozen JSON rationales (~chars/4). Reasoning-model
    hidden tokens are NOT included → dollar figures are lower bounds for
    gpt-oss.
  - USD: published list prices (cite in paper; rates change).

Outputs: results/cost_latency.json / .md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_threeway_table import _load, _load_human_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_judge import STEP_RUBRIC, STEP_TEMPLATE  # noqa: E402
from src.verifier import verify_derivation  # noqa: E402
import data.eval_queries as eq  # noqa: E402

# Published list prices USD / 1M tokens (document access date in paper).
PRICE = {
    "gpt-oss-120b": {"in": 0.35, "out": 0.75, "source": "Cerebras list price"},
    "claude-sonnet-4-5": {"in": 3.00, "out": 15.00, "source": "Anthropic list price"},
}


def _tok(s: str) -> int:
    return max(1, (len(s) + 3) // 4)


def _render_prior(steps: List[Dict[str, Any]], upto: int) -> str:
    parts = []
    for s in steps[:upto]:
        parts.append(
            f"expr: {s.get('expression_given', '')}\n"
            f"ops: {s.get('description', '')}"
        )
    return "\n---\n".join(parts) if parts else "(none)"


def _human_prompt_tokens(labels: Dict, qmap: Dict[str, Dict]) -> Tuple[List[int], List[int]]:
    gpt = _load(ROOT / "results/judge_step_verdicts.json")
    in_toks: List[int] = []
    out_toks: List[int] = []
    for qid, jsteps in gpt["verdicts"].items():
        lab = labels.get((qid, 1))  # probe existence
        # reload full query steps
        path = ROOT / "results/label_templates" / f"{qid}.json"
        if not path.exists():
            continue
        qlab = json.loads(path.read_text())
        qtext = qmap.get(qid, {}).get("question", "")
        for s in jsteps:
            sn = int(s["step_number"])
            cur = next((x for x in qlab["steps"] if int(x["step_number"]) == sn), None)
            if cur is None:
                continue
            prompt = STEP_TEMPLATE.format(
                rubric=STEP_RUBRIC,
                question=qtext,
                prior_steps=_render_prior(qlab["steps"], sn - 1),
                expression=cur.get("expression_given", ""),
                operation=cur.get("description", ""),
            )
            in_toks.append(_tok(prompt))
            out_toks.append(_tok(s.get("judge_raw") or s.get("reasoning") or ""))
    return in_toks, out_toks


def _time_symbolic(label_dir: Path) -> Dict[str, Any]:
    times = []
    n_steps = 0
    for path in sorted(label_dir.glob("Q*.json")):
        d = json.loads(path.read_text())
        steps = [
            {
                "expression": s.get("expression_given", ""),
                "operation": s.get("description", ""),
                "description": s.get("description", ""),
            }
            for s in d["steps"]
        ]
        n_steps += len(steps)
        t0 = time.perf_counter()
        verify_derivation({"steps": steps, "final_answer": ""}, fast_mode=True)
        times.append(time.perf_counter() - t0)
    return {
        "n_queries": len(times),
        "n_steps": n_steps,
        "seconds_total": sum(times),
        "seconds_per_query_mean": statistics.mean(times) if times else None,
        "ms_per_step_mean": 1000 * sum(times) / n_steps if n_steps else None,
        "usd": 0.0,
    }


def _usd(n_calls: int, mean_in: float, mean_out: float, model: str) -> float:
    p = PRICE[model]
    return n_calls * (mean_in * p["in"] + mean_out * p["out"]) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=ROOT / "results/label_templates")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cost_latency.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "results/cost_latency.md")
    args = ap.parse_args()

    labels = _load_human_labels(args.labels)
    qmap = {q["id"]: q for q in eq.EVAL_QUERIES}
    in_toks, out_toks = _human_prompt_tokens(labels, qmap)
    mean_in = statistics.mean(in_toks)
    mean_out = statistics.mean(out_toks)

    sym = _time_symbolic(args.labels)

    # Call counts from human labels + cascade routing.
    n = len(labels)
    sym_commit = sum(1 for s in labels.values() if s.get("verifier_is_valid") is not None)
    sym_abstain = n - sym_commit

    cascade = _load(ROOT / "results/cascade_table.json")
    # used_judge + used_abstain = API calls that hit the judge
    def cascade_calls(dataset: str, model_suffix: str) -> int:
        target = f"cascade→{model_suffix}"
        for r in cascade["rows"]:
            if r["dataset"] == dataset and str(r["grader"]) == target:
                return int(r["used_judge"]) + int(r["used_abstain"])
        raise KeyError(target)

    inj = _load(ROOT / "results/error_injection_report_with_judge.json")
    n_inj = sum(1 for t in inj["trials"] if not t.get("skipped_reason"))

    policies = []
    for model in ("gpt-oss-120b", "claude-sonnet-4-5"):
        alone_h = n
        casc_h = cascade_calls("human_labels", model)
        alone_i = n_inj
        casc_i = cascade_calls("error_injection", model)
        policies.append({
            "model": model,
            "human": {
                "judge_alone_calls": alone_h,
                "cascade_judge_calls": casc_h,
                "cascade_call_reduction": 1 - casc_h / alone_h,
                "judge_alone_usd_est": _usd(alone_h, mean_in, mean_out, model),
                "cascade_usd_est": _usd(casc_h, mean_in, mean_out, model),
            },
            "injection": {
                "judge_alone_calls": alone_i,
                "cascade_judge_calls": casc_i,
                "cascade_call_reduction": 1 - casc_i / alone_i,
                "judge_alone_usd_est": _usd(alone_i, mean_in, mean_out, model),
                "cascade_usd_est": _usd(casc_i, mean_in, mean_out, model),
            },
        })

    payload = {
        "notes": [
            "Symbolic timed locally (CPU); USD=0.",
            "LLM RTT not logged in frozen runs; we report call counts, not wall latency.",
            "Token estimates: prompt reconstructed from STEP_TEMPLATE; output from "
            "frozen rationale JSON (~4 chars/token). Reasoning hidden tokens excluded "
            "→ gpt-oss USD is a lower bound.",
            "Prices are published list rates and will change; treat as order-of-magnitude.",
        ],
        "prices_usd_per_mtok": PRICE,
        "token_estimates_human_step": {
            "n": len(in_toks),
            "mean_input_tokens": mean_in,
            "mean_output_tokens": mean_out,
            "median_input_tokens": statistics.median(in_toks),
        },
        "symbolic": sym,
        "human_n_steps": n,
        "human_symbolic_commits": sym_commit,
        "human_symbolic_abstains": sym_abstain,
        "injection_n_trials": n_inj,
        "policies": policies,
    }
    args.out.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Cost / latency price tag",
        "",
        f"Symbolic (local): {sym['ms_per_step_mean']:.1f} ms/step mean, "
        f"{sym['seconds_total']:.1f}s for {sym['n_steps']} steps, **$0**.",
        "",
        f"Token estimate per human judge call: ~{mean_in:.0f} in / ~{mean_out:.0f} out "
        f"(n={len(in_toks)}).",
        "",
        "| policy (human, n=191) | LLM calls | est. USD (gpt-oss) | est. USD (claude) |",
        "| --- | ---: | ---: | ---: |",
    ]
    g = policies[0]["human"]
    c = policies[1]["human"]
    lines.append(
        f"| judge alone | {g['judge_alone_calls']} | "
        f"${g['judge_alone_usd_est']:.3f} | ${c['judge_alone_usd_est']:.3f} |"
    )
    lines.append(
        f"| cascade (sym→judge on abstain) | {g['cascade_judge_calls']} "
        f"({100*g['cascade_call_reduction']:.0f}% fewer) | "
        f"${g['cascade_usd_est']:.3f} | ${c['cascade_usd_est']:.3f} |"
    )
    lines += [
        "",
        "| policy (injection, n=118) | LLM calls | est. USD (gpt-oss) | est. USD (claude) |",
        "| --- | ---: | ---: | ---: |",
    ]
    g = policies[0]["injection"]
    c = policies[1]["injection"]
    lines.append(
        f"| judge alone | {g['judge_alone_calls']} | "
        f"${g['judge_alone_usd_est']:.3f} | ${c['judge_alone_usd_est']:.3f} |"
    )
    lines.append(
        f"| cascade | {g['cascade_judge_calls']} "
        f"({100*g['cascade_call_reduction']:.0f}% fewer) | "
        f"${g['cascade_usd_est']:.3f} | ${c['cascade_usd_est']:.3f} |"
    )
    lines += [
        "",
        "Prices: gpt-oss-120b $0.35/$0.75 per MTok in/out (Cerebras); "
        "claude-sonnet-4-5 $3/$15 (Anthropic). LLM wall latency not instrumented.",
        "",
    ]
    args.out_md.write_text("\n".join(lines))
    print(args.out_md.read_text())
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
