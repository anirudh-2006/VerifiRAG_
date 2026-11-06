#!/usr/bin/env python3
"""Precision–coverage curves by thresholding LLM-judge confidence.

For each judge × testbed, vary τ ∈ [0, 1]: commit only when
  verdict ∈ {passed, failed} AND confidence ≥ τ;
otherwise abstain. This uses the confidence field already stored in the
frozen judge artifacts—no new API calls.

Metrics match build_threeway_table.py (strict recall, covered FPR/precision).

Outputs:
  results/confidence_curves.json
  results/figures/fig3_precision_coverage.{png,svg}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_threeway_table import (  # noqa: E402
    _find_judge_key,
    _load,
    _load_human_labels,
    _rates,
)


def _score_preds(
    items: List[Tuple[bool, Optional[str], Optional[float]]],
    tau: float,
) -> Dict[str, Any]:
    """items = list of (is_error, verdict, confidence)."""
    tp = fp = tn = fn = abs_e = abs_n = 0
    for is_error, verdict, conf in items:
        commit = (
            verdict in ("passed", "failed")
            and conf is not None
            and float(conf) >= tau
        )
        if not commit:
            if is_error:
                abs_e += 1
            else:
                abs_n += 1
            continue
        pred_err = verdict == "failed"
        if is_error and pred_err:
            tp += 1
        elif is_error and not pred_err:
            fn += 1
        elif (not is_error) and pred_err:
            fp += 1
        else:
            tn += 1
    row = _rates(tp, fp, tn, fn, abs_e, abs_n)
    committed = tp + fp + tn + fn
    row["accuracy_covered"] = (
        (tp + tn) / committed if committed else None
    )
    row["tau"] = tau
    return row


def _human_items(
    labels: Dict[Tuple[str, int], Dict[str, Any]],
    verdicts_by_qid: Dict[str, List[Dict[str, Any]]],
) -> List[Tuple[bool, Optional[str], Optional[float]]]:
    judge: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for qid, steps in verdicts_by_qid.items():
        for s in steps:
            judge[(qid, int(s["step_number"]))] = s
    items = []
    for key, lab in labels.items():
        if key not in judge:
            continue
        s = judge[key]
        items.append(
            (
                lab.get("human_correct") is False,
                s.get("verdict"),
                s.get("confidence"),
            )
        )
    return items


def _injection_items(
    report: Dict[str, Any], judge_key: str
) -> List[Tuple[bool, Optional[str], Optional[float]]]:
    items = []
    for t in report["trials"]:
        if t.get("skipped_reason"):
            continue
        v = (t.get("verdicts") or {}).get(judge_key) or {}
        items.append(
            (
                t["severity"] == "error",
                v.get("verdict"),
                v.get("confidence"),
            )
        )
    return items


def _thresholds(items: List[Tuple[bool, Optional[str], Optional[float]]]) -> List[float]:
    confs = sorted(
        {round(float(c), 4) for _, v, c in items if v in ("passed", "failed") and c is not None}
    )
    # Always include 0 (accept all commits) and a grid for smooth curves.
    grid = [i / 20 for i in range(0, 21)]  # 0.00, 0.05, ..., 1.00
    taus = sorted(set([0.0] + grid + confs))
    return taus


def _curve(items: List[Tuple[bool, Optional[str], Optional[float]]]) -> List[Dict[str, Any]]:
    return [_score_preds(items, tau) for tau in _thresholds(items)]


def _plot(curves: Dict[str, Dict[str, List[Dict[str, Any]]]], out_dir: Path) -> None:
    """curves[dataset][grader] = list of metric dicts over tau."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"gpt-oss-120b": "#DD8452", "claude-sonnet-4-5": "#55A868"}
    markers = {"gpt-oss-120b": "o", "claude-sonnet-4-5": "s"}

    for ax, dataset, title in zip(
        axes,
        ["human_labels", "error_injection"],
        ["Human labels", "Error injection"],
    ):
        for grader, pts in curves[dataset].items():
            xs, ys = [], []
            for p in pts:
                if p["precision"] is None or p["coverage"] is None:
                    continue
                if p["coverage"] <= 0:
                    continue
                xs.append(p["coverage"])
                ys.append(p["precision"])
            if not xs:
                continue
            # Sort by coverage for a clean left-to-right curve.
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            short = "gpt-oss" if grader.startswith("gpt") else "claude"
            ax.plot(
                xs, ys,
                color=colors.get(grader, "#333"),
                marker=markers.get(grader, "o"),
                markersize=4,
                linewidth=1.6,
                label=short,
            )
            # Mark the default operating point (tau=0, all committed verdicts).
            default = next((p for p in pts if p["tau"] == 0.0), None)
            if default and default["precision"] is not None:
                ax.scatter(
                    [default["coverage"]], [default["precision"]],
                    s=80, zorder=5, color=colors.get(grader, "#333"),
                    edgecolors="black", linewidths=0.8,
                )
        ax.set_xlabel("coverage")
        ax.set_title(title)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    axes[0].set_ylabel("error-detection precision")
    fig.suptitle(
        "Judge confidence thresholding: precision vs.\ coverage",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = out_dir / f"fig3_precision_coverage.{ext}"
        fig.savefig(path, dpi=200)
        print(f"Wrote: {path}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("results/label_templates"))
    ap.add_argument("--gptoss-human", type=Path,
                    default=Path("results/judge_step_verdicts.json"))
    ap.add_argument("--claude-human", type=Path,
                    default=Path("results/judge_step_verdicts_claude.json"))
    ap.add_argument("--gptoss-injection", type=Path,
                    default=Path("results/error_injection_report_with_judge.json"))
    ap.add_argument("--claude-injection", type=Path,
                    default=Path("results/error_injection_report_claude.json"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/confidence_curves.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    labels = _load_human_labels(args.labels)
    gptoss_h = _load(args.gptoss_human)
    claude_h = _load(args.claude_human)
    gptoss_i = _load(args.gptoss_injection)
    claude_i = _load(args.claude_injection)
    gptoss_key = _find_judge_key(gptoss_i["metrics"])
    claude_key = _find_judge_key(claude_i["metrics"])
    gptoss_model = gptoss_key.split(":", 1)[-1]
    claude_model = claude_key.split(":", 1)[-1]

    bundles = {
        "human_labels": {
            gptoss_model: _human_items(labels, gptoss_h["verdicts"]),
            claude_model: _human_items(labels, claude_h["verdicts"]),
        },
        "error_injection": {
            gptoss_model: _injection_items(gptoss_i, gptoss_key),
            claude_model: _injection_items(claude_i, claude_key),
        },
    }

    curves: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dataset, graders in bundles.items():
        curves[dataset] = {}
        for grader, items in graders.items():
            curves[dataset][grader] = _curve(items)

    payload = {
        "rule": "commit iff verdict in {passed,failed} and confidence >= tau; else abstain",
        "note": "Black-edged markers = default operating point (tau=0).",
        "curves": curves,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {args.out_json}")

    # Print a few operating points for the paper prose.
    for dataset in ("human_labels", "error_injection"):
        print(f"\n=== {dataset} ===")
        for grader, pts in curves[dataset].items():
            print(f"  {grader}:")
            # default + a mid/high tau with defined precision
            interesting = [p for p in pts if p["tau"] in (0.0, 0.7, 0.8, 0.9, 0.95, 0.99)]
            for p in interesting:
                print(
                    f"    tau={p['tau']:.2f} cov={p['coverage']:.3f} "
                    f"prec={p['precision']} rec={p['recall']} "
                    f"fpr={p['fpr']} acc_cov={p['accuracy_covered']}"
                )

    _plot(curves, args.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
