#!/usr/bin/env python3
"""Render publication figures from the frozen evaluation artifacts.

Now renders THREE graders: the symbolic verifier and two independent LLM
judges (gpt-oss-120b and claude-sonnet-4-5).

Figures produced (PNG + SVG) into results/figures/:
  fig1_grader_comparison   : coverage / recall / abstain for all three graders,
                             on both the human-labeled set and the
                             error-injection set.
  fig2_injection_by_family : per-mutation-family strict recall for all three
                             graders on the error-injection set.

Inputs:
  results/threeway_table.json                     (fig1)
  results/error_injection_report_with_judge.json  (fig2, symbolic + gpt-oss)
  results/error_injection_report_claude.json       (fig2, claude)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One color per grader; keys are matched by prefix so exact model tags work.
GRADER_COLORS = {
    "symbolic": "#4C72B0",
    "gpt-oss": "#DD8452",
    "claude": "#55A868",
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _color_for(grader: str) -> str:
    for key, color in GRADER_COLORS.items():
        if grader.startswith(key):
            return color
    return "#999999"


def _grouped_bar_n(ax, categories, series, title, ylabel):
    """series = list of (label, values) tuples, one per grader."""
    n = len(series)
    x = range(len(categories))
    total_width = 0.8
    width = total_width / n
    for si, (label, values) in enumerate(series):
        offset = -total_width / 2 + width * (si + 0.5)
        positions = [i + offset for i in x]
        ax.bar(positions, values, width, label=label, color=_color_for(label))
        for xi, v in zip(positions, values):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def _rows_by_dataset(table: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in table["rows"]:
        out.setdefault(r["dataset"], {})[r["grader"]] = r
    return out


def fig1_grader_comparison(table: Dict[str, Any], out_dir: Path) -> None:
    graders: List[str] = table["graders"]  # [symbolic, gpt-oss, claude]
    data = _rows_by_dataset(table)
    datasets = ["human_labels", "error_injection"]
    cat_labels = ["Human\nlabels", "Error\ninjection"]

    def series_for(metric: str, default: float = 0.0):
        out = []
        for g in graders:
            vals = []
            for ds in datasets:
                v = data[ds][g].get(metric)
                vals.append(float(v) if v is not None else default)
            out.append((g, vals))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _grouped_bar_n(axes[0], cat_labels, series_for("coverage"),
                   "Coverage (higher = better)", "coverage")
    _grouped_bar_n(axes[1], cat_labels, series_for("recall"),
                   "Error-detection recall (higher = better)", "recall")
    _grouped_bar_n(axes[2], cat_labels, series_for("abstain_rate"),
                   "Abstain rate (lower = better)", "abstain rate")
    fig.suptitle("Symbolic verifier vs two LLM judges", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "fig1_grader_comparison")


def _family_recall(report: Dict[str, Any], grader_key: str, family: str) -> float:
    fam = report["metrics"][grader_key]["by_family"].get(family) or {}
    v = fam.get("recall_strict")
    return float(v) if v is not None else 0.0


def fig2_injection_by_family(gptoss: Dict[str, Any], claude: Dict[str, Any],
                             out_dir: Path) -> None:
    gptoss_key = next(k for k in gptoss["metrics"] if k.startswith("llm_judge"))
    claude_key = next(k for k in claude["metrics"] if k.startswith("llm_judge"))
    gptoss_model = gptoss_key.split(":", 1)[-1]
    claude_model = claude_key.split(":", 1)[-1]

    families = ["algebraic", "bounds", "identifier"]
    series = [
        ("symbolic", [_family_recall(gptoss, "symbolic", f) for f in families]),
        (gptoss_model, [_family_recall(gptoss, gptoss_key, f) for f in families]),
        (claude_model, [_family_recall(claude, claude_key, f) for f in families]),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    _grouped_bar_n(ax, [f.capitalize() for f in families], series,
                   "Error-injection strict recall by mutation family", "recall")
    ax.set_xlabel("mutation family (error-severity)")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_injection_by_family")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    print(f"Wrote: {png}")
    print(f"Wrote: {svg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path,
                    default=Path("results/threeway_table.json"))
    ap.add_argument("--gptoss-injection", type=Path,
                    default=Path("results/error_injection_report_with_judge.json"))
    ap.add_argument("--claude-injection", type=Path,
                    default=Path("results/error_injection_report_claude.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    table = _load(args.table)
    gptoss = _load(args.gptoss_injection)
    claude = _load(args.claude_injection)

    fig1_grader_comparison(table, args.out_dir)
    fig2_injection_by_family(gptoss, claude, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
