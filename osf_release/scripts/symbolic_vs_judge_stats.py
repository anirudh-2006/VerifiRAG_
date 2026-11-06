#!/usr/bin/env python3
"""Paired statistical comparison of the symbolic verifier vs the LLM judge.

Both graders are evaluated on the SAME human-labeled derivation steps, so the
comparison is paired at the step level (keyed by (query_id, step_number)).

For each step we derive, per grader, three binary indicators relative to the
human gold label:

  committed        : 1 if the grader gave a verdict (not abstain), else 0
  correct_committed: 1 if the grader committed AND matched the human label
  caught_error     : (only on human-error steps) 1 if grader flagged it wrong

We then run, for each metric, a paired bootstrap CI and a paired permutation
test on the mean delta (judge - symbolic), reusing src.stats.

Outputs:
  results/stats_symbolic_vs_judge.json
  results/stats_symbolic_vs_judge.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.stats import (  # noqa: E402
    mcnemar_exact_test,
    paired_bootstrap_mean_delta,
    paired_permutation_test_mean_delta,
)


def _load_human_labels(label_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Map (qid, step_number) -> {human_correct, verifier_is_valid, error_type}."""
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for fp in sorted(glob.glob(str(label_dir / "Q*.json"))):
        with open(fp) as f:
            d = json.load(f)
        if not d.get("_filled"):
            continue
        qid = d.get("id") or Path(fp).stem
        for s in d.get("steps", []):
            sn = s.get("step_number")
            if sn is None:
                continue
            if s.get("human_correct") is None:
                continue
            out[(qid, sn)] = {
                "human_correct": s.get("human_correct"),
                "verifier_is_valid": s.get("verifier_is_valid"),
                "error_type": s.get("error_type"),
            }
    return out


def _load_judge_verdicts(path: Path) -> Dict[Tuple[str, int], str]:
    """Map (qid, step_number) -> judge verdict string."""
    with open(path) as f:
        d = json.load(f)
    out: Dict[Tuple[str, int], str] = {}
    for qid, steps in d.get("verdicts", {}).items():
        for s in steps:
            sn = s.get("step_number")
            if sn is None:
                continue
            out[(qid, sn)] = s.get("verdict", "abstain")
    return out


def _verifier_committed_correct(
    verifier_is_valid: Optional[bool], human_correct: bool
) -> Tuple[int, Optional[int]]:
    """Return (committed, correct_if_committed)."""
    if verifier_is_valid is None:
        return 0, None
    predicted_correct = bool(verifier_is_valid)
    return 1, int(predicted_correct == bool(human_correct))


def _judge_committed_correct(
    verdict: str, human_correct: bool
) -> Tuple[int, Optional[int]]:
    if verdict == "passed":
        predicted_correct = True
    elif verdict == "failed":
        predicted_correct = False
    else:  # abstain / unparseable / rate-limited
        return 0, None
    return 1, int(predicted_correct == bool(human_correct))


def _caught_error(committed: int, correct_if_committed: Optional[int]) -> Optional[int]:
    """On a human-error step: 1 if grader committed and correctly said 'wrong'."""
    if committed == 0:
        return 0  # abstaining on an error is a miss
    # correct_if_committed==1 means grader's verdict matched human (=wrong)
    return int(correct_if_committed == 1)


def build_paired_vectors(
    labels: Dict[Tuple[str, int], Dict[str, Any]],
    judge: Dict[Tuple[str, int], str],
) -> Dict[str, Any]:
    keys = sorted(k for k in labels.keys() if k in judge)

    committed_sym: List[float] = []
    committed_jud: List[float] = []
    correct_sym: List[float] = []
    correct_jud: List[float] = []
    caught_sym: List[float] = []
    caught_jud: List[float] = []
    error_keys: List[Tuple[str, int]] = []

    for k in keys:
        lab = labels[k]
        human = bool(lab["human_correct"])

        c_sym, ok_sym = _verifier_committed_correct(lab["verifier_is_valid"], human)
        c_jud, ok_jud = _judge_committed_correct(judge[k], human)

        committed_sym.append(float(c_sym))
        committed_jud.append(float(c_jud))

        # correct-committed as accuracy over all steps: treat abstain as 0
        # (grader gets no credit if it did not commit)
        correct_sym.append(float(ok_sym) if ok_sym is not None else 0.0)
        correct_jud.append(float(ok_jud) if ok_jud is not None else 0.0)

        if human is False:  # human says this step is an ERROR
            error_keys.append(k)
            caught_sym.append(float(_caught_error(c_sym, ok_sym)))
            caught_jud.append(float(_caught_error(c_jud, ok_jud)))

    return {
        "paired_step_keys": [f"{q}:{s}" for q, s in keys],
        "n_steps": len(keys),
        "n_error_steps": len(error_keys),
        "error_step_keys": [f"{q}:{s}" for q, s in error_keys],
        "vectors": {
            "committed": (committed_sym, committed_jud),
            "correct_committed": (correct_sym, correct_jud),
            "caught_error": (caught_sym, caught_jud),
        },
    }


def _run_metric(
    name: str,
    sym: List[float],
    jud: List[float],
    *,
    n_boot: int,
    n_perm: int,
    ci: float,
    seed: int,
) -> Dict[str, Any]:
    n = len(sym)
    if n == 0:
        return {"metric": name, "n_pairs": 0, "note": "no paired observations"}
    boot = paired_bootstrap_mean_delta(sym, jud, n_boot=n_boot, ci=ci, seed=seed)
    perm = paired_permutation_test_mean_delta(sym, jud, n_perm=n_perm, seed=seed + 1)
    mcn = mcnemar_exact_test(sym, jud)
    ci_uninformative = bool(
        name == "caught_error"
        and (
            boot["ci_low"] <= 0.0 + 1e-12
            or boot["ci_high"] >= 1.0 - 1e-12
            or n < 20
        )
    )
    return {
        "metric": name,
        "delta_definition": "judge - symbolic",
        "n_pairs": n,
        "mean_symbolic": sum(sym) / n,
        "mean_judge": sum(jud) / n,
        "mean_delta": boot["mean_delta"],
        "ci_level": ci,
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "ci_uninformative": ci_uninformative,
        "p_value_permutation": perm["p_value_two_sided"],
        # Keep legacy key for older readers; primary binary test is McNemar.
        "p_value_two_sided": mcn["p_value_two_sided"],
        "mcnemar": mcn,
        "n_boot": n_boot,
        "n_perm": n_perm,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Symbolic verifier vs LLM judge — paired step-level stats",
        "",
        f"- Steps paired: {report['n_steps']}",
        f"- Human-error steps: {report['n_error_steps']}",
        f"- Bootstrap reps: {report['config']['n_boot']}, "
        f"permutations: {report['config']['n_perm']}, "
        f"CI: {report['config']['ci']}",
        "",
        "| metric | mean(symbolic) | mean(judge) | delta (judge-sym) | "
        f"{int(report['config']['ci']*100)}% CI | McNemar p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in report["metrics"]:
        if m.get("n_pairs", 0) == 0:
            lines.append(f"| {m['metric']} | — | — | — | — | — |")
            continue
        if m.get("ci_uninformative"):
            ci = "uninformative (n small / CI hits boundary)"
        else:
            ci = f"[{_fmt(m['ci_low'])}, {_fmt(m['ci_high'])}]"
        lines.append(
            f"| {m['metric']} | {_fmt(m['mean_symbolic'])} | {_fmt(m['mean_judge'])} "
            f"| {_fmt(m['mean_delta'])} | {ci} | {_fmt(m['p_value_two_sided'])} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: `committed` = coverage (fraction of steps with a "
        "verdict); `correct_committed` = fraction of steps where the grader "
        "committed AND matched the human label (abstains score 0); "
        "`caught_error` = recall on human-labeled error steps. Primary paired "
        "binary test is exact McNemar; bootstrap CIs are descriptive. "
        "Holm–Bonferroni adjustments over the six-test family are in "
        "`stats_paired_summary.json`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-dir", type=Path, default=Path("results/label_templates"))
    ap.add_argument(
        "--judge-verdicts", type=Path, default=Path("results/judge_step_verdicts.json")
    )
    ap.add_argument("--out-json", type=Path, default=Path("results/stats_symbolic_vs_judge.json"))
    ap.add_argument("--out-md", type=Path, default=Path("results/stats_symbolic_vs_judge.md"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = _load_human_labels(args.label_dir)
    judge = _load_judge_verdicts(args.judge_verdicts)
    paired = build_paired_vectors(labels, judge)

    metrics_out: List[Dict[str, Any]] = []
    for name, (sym, jud) in paired["vectors"].items():
        metrics_out.append(
            _run_metric(
                name,
                sym,
                jud,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                ci=args.ci,
                seed=args.seed,
            )
        )

    report = {
        "sources": {
            "label_dir": str(args.label_dir),
            "judge_verdicts": str(args.judge_verdicts),
        },
        "config": {
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "ci": args.ci,
            "seed": args.seed,
        },
        "n_steps": paired["n_steps"],
        "n_error_steps": paired["n_error_steps"],
        "error_step_keys": paired["error_step_keys"],
        "metrics": metrics_out,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    with args.out_md.open("w") as f:
        f.write(_to_markdown(report))

    print(f"Paired steps: {paired['n_steps']} (error steps: {paired['n_error_steps']})")
    for m in metrics_out:
        if m.get("n_pairs", 0) == 0:
            print(f"  {m['metric']}: (no pairs)")
            continue
        ci_note = (
            "CI=uninformative"
            if m.get("ci_uninformative")
            else f"CI=[{m['ci_low']:.3f},{m['ci_high']:.3f}]"
        )
        print(
            f"  {m['metric']}: sym={m['mean_symbolic']:.3f} judge={m['mean_judge']:.3f} "
            f"delta={m['mean_delta']:.3f} {ci_note} "
            f"McNemar_p={m['p_value_two_sided']:.4g} "
            f"perm_p={m.get('p_value_permutation', float('nan')):.4g}"
        )
    print(f"\nWrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
