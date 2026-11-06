#!/usr/bin/env python3
"""Build paper-ready headline artifacts from evaluation JSON outputs.

Outputs:
  - headline_table_1.json
  - headline_table_1.csv
  - headline_table_1.md
  - headline_figure_data.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _find_judge_key(keys: List[str]) -> str:
    for k in keys:
        if k.startswith("llm_judge"):
            return k
    raise KeyError("Could not find llm_judge* key in metrics")


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _extract_human_table(
    judge_vs_human: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    scores = judge_vs_human["scores"]
    verifier = scores["verifier_vs_human"]
    judge = scores["judge_vs_human"]

    def row(src: Dict[str, Any]) -> Dict[str, Any]:
        cm = src["confusion_matrix"]
        return {
            "coverage": src["coverage"],
            "precision_error_detection": src["precision_error_detection"],
            "recall_error_detection": src["recall_error_detection"],
            "accuracy_on_covered_steps": src["accuracy_on_covered_steps"],
            "tp": cm["true_positive"],
            "fp": cm["false_positive"],
            "tn": cm["true_negative"],
            "fn": cm["false_negative"],
            "steps_total": src["steps_total"],
            "steps_abstained": src["steps_abstained"],
        }

    return {"symbolic": row(verifier), "judge": row(judge)}


def _extract_injection_table(
    error_injection: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    metrics = error_injection["metrics"]
    judge_key = _find_judge_key(list(metrics.keys()))
    symbolic = metrics["symbolic"]["overall"]
    judge = metrics[judge_key]["overall"]

    def row(src: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "precision": src["precision"],
            "recall_strict": src["recall_strict"],
            "recall_committed": src["recall_committed"],
            "fpr": src["fpr"],
            "abstain_rate": src["abstain_rate"],
            "tp": src["tp"],
            "fp": src["fp"],
            "tn": src["tn"],
            "fn": src["fn"],
            "abstain_error": src["abstain_error"],
            "abstain_eq": src["abstain_eq"],
            "total": src["total"],
        }

    return {"symbolic": row(symbolic), "judge": row(judge)}, judge_key


def _build_csv_rows(
    human: Dict[str, Dict[str, Any]],
    injection: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for grader in ("symbolic", "judge"):
        h = human[grader]
        rows.append(
            {
                "dataset": "human_labels",
                "grader": grader,
                "coverage": h["coverage"],
                "precision": h["precision_error_detection"],
                "recall": h["recall_error_detection"],
                "fpr": None,
                "abstain_rate": h["steps_abstained"] / h["steps_total"],
                "tp": h["tp"],
                "fp": h["fp"],
                "tn": h["tn"],
                "fn": h["fn"],
                "total": h["steps_total"],
            }
        )
    for grader in ("symbolic", "judge"):
        e = injection[grader]
        rows.append(
            {
                "dataset": "error_injection",
                "grader": grader,
                "coverage": 1.0 - e["abstain_rate"],
                "precision": e["precision"],
                "recall": e["recall_strict"],
                "fpr": e["fpr"],
                "abstain_rate": e["abstain_rate"],
                "tp": e["tp"],
                "fp": e["fp"],
                "tn": e["tn"],
                "fn": e["fn"],
                "total": e["total"],
            }
        )
    return rows


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "dataset",
        "grader",
        "coverage",
        "precision",
        "recall",
        "fpr",
        "abstain_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "total",
    ]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_fmt(row[h]) for h in headers) + " |")
    return "\n".join(out) + "\n"


def _to_figure_payload(
    human: Dict[str, Dict[str, Any]],
    injection: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "coverage": {
            "human_labels": {
                "symbolic": human["symbolic"]["coverage"],
                "judge": human["judge"]["coverage"],
            },
            "error_injection": {
                "symbolic": 1.0 - injection["symbolic"]["abstain_rate"],
                "judge": 1.0 - injection["judge"]["abstain_rate"],
            },
        },
        "recall_error_detection": {
            "human_labels": {
                "symbolic": human["symbolic"]["recall_error_detection"],
                "judge": human["judge"]["recall_error_detection"],
            },
            "error_injection_strict": {
                "symbolic": injection["symbolic"]["recall_strict"],
                "judge": injection["judge"]["recall_strict"],
            },
        },
        "abstain_rate": {
            "human_labels": {
                "symbolic": human["symbolic"]["steps_abstained"] / human["symbolic"]["steps_total"],
                "judge": human["judge"]["steps_abstained"] / human["judge"]["steps_total"],
            },
            "error_injection": {
                "symbolic": injection["symbolic"]["abstain_rate"],
                "judge": injection["judge"]["abstain_rate"],
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--judge-vs-human",
        type=Path,
        default=Path("results/judge_step_verdicts.json"),
    )
    ap.add_argument(
        "--error-injection",
        type=Path,
        default=Path("results/error_injection_report_with_judge.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
    )
    args = ap.parse_args()

    judge_data = _load_json(args.judge_vs_human)
    injection_data = _load_json(args.error_injection)

    human = _extract_human_table(judge_data)
    injection, judge_key = _extract_injection_table(injection_data)
    rows = _build_csv_rows(human, injection)

    payload = {
        "sources": {
            "judge_vs_human": str(args.judge_vs_human),
            "error_injection": str(args.error_injection),
        },
        "judge_metric_key": judge_key,
        "table_rows": rows,
        "human_summary": human,
        "injection_summary": injection,
    }
    fig_payload = _to_figure_payload(human, injection)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "headline_table_1.json"
    csv_path = args.out_dir / "headline_table_1.csv"
    md_path = args.out_dir / "headline_table_1.md"
    fig_path = args.out_dir / "headline_figure_data.json"

    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as f:
        f.write(_to_markdown(rows))

    with fig_path.open("w") as f:
        json.dump(fig_payload, f, indent=2)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
