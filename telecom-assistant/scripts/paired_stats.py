#!/usr/bin/env python3
"""
Compute paired bootstrap CIs + significance between two evaluation arms.

Examples
--------
# 1) Compare plain vs graph from one ablation file
PYTHONPATH=. python scripts/paired_stats.py \
  --from-ablation results/eval_ablation_20260420_220157.json \
  --arm-a plain --arm-b graph \
  --metric final_match_binary

# 2) Compare two independent eval files (e.g., symbolic vs judge reruns)
PYTHONPATH=. python scripts/paired_stats.py \
  --file-a results/eval_symbolic.json \
  --file-b results/eval_judge.json \
  --arm-a symbolic --arm-b judge \
  --metric final_match_binary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.stats import (  # noqa: E402
    BUILTIN_METRICS,
    compare_paired,
    load_results_rows,
    metric_spec,
    split_rows_by_mode,
)


def _resolve_rows(args: argparse.Namespace) -> Tuple[List[dict], List[dict], str, str]:
    if args.from_ablation:
        rows = load_results_rows(Path(args.from_ablation))
        rows_a, rows_b = split_rows_by_mode(
            rows,
            mode_path=args.mode_path,
            arm_a=args.arm_a,
            arm_b=args.arm_b,
        )
        return rows_a, rows_b, args.arm_a, args.arm_b

    if args.file_a and args.file_b:
        rows_a = load_results_rows(Path(args.file_a))
        rows_b = load_results_rows(Path(args.file_b))
        return rows_a, rows_b, args.arm_a, args.arm_b

    raise ValueError(
        "Provide either --from-ablation or both --file-a and --file-b"
    )


def _print_summary(report: Dict[str, Any]) -> None:
    arms = report["arms"]
    metric = report["metric"]
    counts = report["paired_counts"]
    boot = report["bootstrap"]
    perm = report["permutation"]
    means = report["mean_values"]

    print("=" * 86)
    print("PAIRED COMPARISON STATS")
    print("=" * 86)
    print(f"Metric                 : {metric['name']}")
    print(f"Delta definition       : {metric['delta_definition']}")
    print(f"Higher-is-better       : {metric['higher_is_better']}")
    print(f"Paired queries         : {counts['n_pairs']}")
    print(f"Wins ({arms['a']})         : {counts['wins_a']}")
    print(f"Wins ({arms['b']})         : {counts['wins_b']}")
    print(f"Ties                   : {counts['ties']}")
    print(f"Skipped pairs          : {counts['skipped_pairs']}")
    print()
    print(f"Mean {arms['a']:<16}: {means[arms['a']]:.6f}")
    print(f"Mean {arms['b']:<16}: {means[arms['b']]:.6f}")
    print(f"Mean delta ({arms['b']} - {arms['a']}): {boot['mean_delta']:.6f}")
    print(
        f"{int(boot['ci_level'] * 100)}% bootstrap CI     : "
        f"[{boot['ci_low']:.6f}, {boot['ci_high']:.6f}]"
    )
    print(f"Permutation p-value    : {perm['p_value_two_sided']:.6g}")
    print(f"Better arm (by delta)  : {report['better_arm_by_mean_delta']}")
    print("=" * 86)


def _run_one_metric(
    rows_a: List[dict],
    rows_b: List[dict],
    *,
    metric_name: str,
    metric_path: Optional[str],
    id_key: str,
    n_boot: int,
    n_perm: int,
    ci: float,
    seed: int,
    arm_a_name: str,
    arm_b_name: str,
) -> Dict[str, Any]:
    spec = metric_spec(metric_name, metric_path=metric_path)
    report = compare_paired(
        rows_a,
        rows_b,
        metric=spec,
        id_key=id_key,
        n_boot=n_boot,
        n_perm=n_perm,
        ci=ci,
        seed=seed,
        arm_a_name=arm_a_name,
        arm_b_name=arm_b_name,
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-ablation",
        type=str,
        help="Single ablation result file containing both arms in one results list.",
    )
    source.add_argument("--file-a", type=str, help="First result file.")
    ap.add_argument("--file-b", type=str, help="Second result file (with --file-a).")

    ap.add_argument("--arm-a", type=str, default="A", help="Display name for arm A.")
    ap.add_argument("--arm-b", type=str, default="B", help="Display name for arm B.")
    ap.add_argument(
        "--mode-path",
        type=str,
        default="ablation.pipeline_retrieval_mode",
        help="Dot path of mode field when using --from-ablation.",
    )

    ap.add_argument(
        "--metric",
        type=str,
        default="final_match_binary",
        help="Metric name. Builtins: " + ", ".join(sorted(BUILTIN_METRICS.keys())),
    )
    ap.add_argument(
        "--metric-path",
        type=str,
        default=None,
        help="Custom dot path to a numeric metric (overrides builtin getter).",
    )
    ap.add_argument("--id-key", type=str, default="id")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--bundle",
        type=str,
        default=None,
        choices=["paper"],
        help=(
            "Run a predefined metric bundle. "
            "Current bundle: 'paper' => final_match_binary, parse_rate_pct, "
            "equiv_rate_pct, checked_failed."
        ),
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path. Default: results/stats_<timestamp>.json",
    )
    args = ap.parse_args()

    if args.file_a and not args.file_b:
        raise SystemExit("--file-b is required when --file-a is set")

    rows_a, rows_b, arm_a_name, arm_b_name = _resolve_rows(args)
    if args.bundle == "paper":
        bundle_metrics = [
            "final_match_binary",
            "parse_rate_pct",
            "equiv_rate_pct",
            "checked_failed",
        ]
    else:
        bundle_metrics = [args.metric]

    bundle_reports: Dict[str, Dict[str, Any]] = {}
    for i, metric_name in enumerate(bundle_metrics):
        metric_path = args.metric_path if (args.bundle is None and i == 0) else None
        report = _run_one_metric(
            rows_a,
            rows_b,
            metric_name=metric_name,
            metric_path=metric_path,
            id_key=args.id_key,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            ci=args.ci,
            seed=args.seed + i,
            arm_a_name=arm_a_name,
            arm_b_name=arm_b_name,
        )
        bundle_reports[metric_name] = report
        _print_summary(report)

    out = args.output
    if not out:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"results/stats_{ts}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any]
    if args.bundle:
        payload = {
            "bundle": args.bundle,
            "arms": {"a": arm_a_name, "b": arm_b_name},
            "metrics": bundle_reports,
        }
    else:
        payload = bundle_reports[bundle_metrics[0]]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable stats to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
