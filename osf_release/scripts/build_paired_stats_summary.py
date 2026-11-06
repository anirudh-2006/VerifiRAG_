#!/usr/bin/env python3
"""Combine symbolic-vs-judge stats for both judges; apply Holm–Bonferroni.

Family = 6 tests (3 metrics × 2 judges). Primary p-values are exact McNemar.
Writes results/stats_paired_summary.json / .md for the paper table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import holm_bonferroni  # noqa: E402


def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-rerun", action="store_true",
                    help="Reuse existing per-judge JSON files")
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/stats_paired_summary.json"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("results/stats_paired_summary.md"))
    args = ap.parse_args()

    py = sys.executable
    if not args.skip_rerun:
        _run([
            py, "scripts/symbolic_vs_judge_stats.py",
            "--judge-verdicts", "results/judge_step_verdicts.json",
            "--out-json", "results/stats_symbolic_vs_judge.json",
            "--out-md", "results/stats_symbolic_vs_judge.md",
        ])
        _run([
            py, "scripts/symbolic_vs_judge_stats.py",
            "--judge-verdicts", "results/judge_step_verdicts_claude.json",
            "--out-json", "results/stats_symbolic_vs_claude.json",
            "--out-md", "results/stats_symbolic_vs_claude.md",
        ])

    gpt = json.loads(Path("results/stats_symbolic_vs_judge.json").read_text())
    claude = json.loads(Path("results/stats_symbolic_vs_claude.json").read_text())

    rows: List[Dict[str, Any]] = []
    named_p = []
    for judge_name, report in (("gpt-oss-120b", gpt), ("claude-sonnet-4-5", claude)):
        for m in report["metrics"]:
            name = f"{judge_name}/{m['metric']}"
            rows.append({
                "judge": judge_name,
                "metric": m["metric"],
                "n_pairs": m["n_pairs"],
                "mean_symbolic": m["mean_symbolic"],
                "mean_judge": m["mean_judge"],
                "mean_delta": m["mean_delta"],
                "ci_low": m["ci_low"],
                "ci_high": m["ci_high"],
                "ci_uninformative": m.get("ci_uninformative", False),
                "mcnemar": m.get("mcnemar"),
                "p_mcnemar": m["p_value_two_sided"],
                "p_permutation": m.get("p_value_permutation"),
            })
            named_p.append((name, m["p_value_two_sided"]))

    holm = holm_bonferroni(named_p, alpha=0.05)
    holm_by_name = {h["name"]: h for h in holm}
    for r in rows:
        key = f"{r['judge']}/{r['metric']}"
        h = holm_by_name[key]
        r["p_holm"] = h["p_holm"]
        r["reject_holm_05"] = h["reject_at_alpha"]

    payload = {
        "family": "3 metrics × 2 judges (exact McNemar); Holm–Bonferroni α=0.05",
        "rows": rows,
        "holm": holm,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Paired stats summary (McNemar + Holm–Bonferroni)",
        "",
        payload["family"],
        "",
        "| judge | metric | Δ | 95% CI | McNemar p | Holm p | reject@0.05 |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        if r["ci_uninformative"] or r["metric"] == "caught_error":
            ci = "uninformative (n=13)"
        else:
            ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(
            f"| {r['judge']} | {r['metric']} | {r['mean_delta']:+.3f} | {ci} | "
            f"{r['p_mcnemar']:.3g} | {r['p_holm']:.3g} | "
            f"{'yes' if r['reject_holm_05'] else 'no'} |"
        )
    lines.append("")
    args.out_md.write_text("\n".join(lines))
    print(args.out_md.read_text())
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
