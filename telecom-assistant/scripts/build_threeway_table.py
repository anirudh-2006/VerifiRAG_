#!/usr/bin/env python3
"""Build a three-way grader comparison table: symbolic vs two LLM judges.

Metric definitions (ONE definition, used everywhere — including the paper):

  Coverage       = (# passed + # failed) / N
  Abstain rate   = 1 - coverage
                   (unparseable / grader-crash verdicts count as abstain)
  Precision      = TP / (TP + FP)           among committed predictions
  Recall         = TP / (TP + FN + AbsE)    STRICT: abstain-on-error is a miss
  FPR            = FP / (FP + TN)           among covered negatives only

  Confusion cells TP/FP/TN/FN are among *committed* verdicts only.
  AbsE = abstain (or crash) on a true-error case; AbsN on a non-error case.
  Identity: TP+FP+TN+FN+AbsE+AbsN = N.

Human-set recall therefore matches the paired ``caught_error`` denominator
(all 13 human-error steps). Injection N = 118 applicable trials
(78 mutation_not_applicable skips excluded).

Inputs (defaults):
  results/label_templates/Q*.json
  results/judge_step_verdicts.json
  results/judge_step_verdicts_claude.json
  results/error_injection_report_with_judge.json
  results/error_injection_report_claude.json

Outputs:
  results/threeway_table.json / .csv / .md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _rates(tp: int, fp: int, tn: int, fn: int, abs_e: int, abs_n: int) -> Dict[str, Any]:
    n = tp + fp + tn + fn + abs_e + abs_n
    committed = tp + fp + tn + fn
    errors = tp + fn + abs_e
    return {
        "coverage": committed / n if n else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / errors) if errors else None,  # STRICT
        "fpr": (fp / (fp + tn)) if (fp + tn) else None,  # covered negatives
        "abstain_rate": (abs_e + abs_n) / n if n else None,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "abs_e": abs_e,
        "abs_n": abs_n,
        "total": n,
    }


def _load_human_labels(label_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in sorted(label_dir.glob("Q*.json")):
        d = _load(path)
        qid = path.stem
        for s in d["steps"]:
            out[(qid, int(s["step_number"]))] = s
    return out


def _human_row_from_judge(
    labels: Dict[Tuple[str, int], Dict[str, Any]],
    verdicts_by_qid: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    judge: Dict[Tuple[str, int], str] = {}
    for qid, steps in verdicts_by_qid.items():
        for s in steps:
            judge[(qid, int(s["step_number"]))] = s["verdict"]

    tp = fp = tn = fn = abs_e = abs_n = 0
    for key, lab in labels.items():
        if key not in judge:
            continue
        human_err = lab.get("human_correct") is False
        v = judge[key]
        if v not in ("passed", "failed"):
            if human_err:
                abs_e += 1
            else:
                abs_n += 1
            continue
        pred_err = v == "failed"
        if human_err and pred_err:
            tp += 1
        elif human_err and not pred_err:
            fn += 1
        elif (not human_err) and pred_err:
            fp += 1
        else:
            tn += 1
    return _rates(tp, fp, tn, fn, abs_e, abs_n)


def _human_row_symbolic(labels: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = tn = fn = abs_e = abs_n = 0
    for lab in labels.values():
        human_err = lab.get("human_correct") is False
        v = lab.get("verifier_is_valid")  # True=OK, False=wrong, None=abstain
        if v is None:
            if human_err:
                abs_e += 1
            else:
                abs_n += 1
            continue
        pred_err = v is False
        if human_err and pred_err:
            tp += 1
        elif human_err and not pred_err:
            fn += 1
        elif (not human_err) and pred_err:
            fp += 1
        else:
            tn += 1
    return _rates(tp, fp, tn, fn, abs_e, abs_n)


def _injection_row(report: Dict[str, Any], grader_key: str) -> Dict[str, Any]:
    """Recompute from trials; treat verdict 'error' (crash) as abstain."""
    tp = fp = tn = fn = abs_e = abs_n = 0
    for t in report["trials"]:
        if t.get("skipped_reason"):
            continue
        v = (t.get("verdicts") or {}).get(grader_key, {}).get("verdict", "error")
        sev = t["severity"]
        is_error = sev == "error"
        if v not in ("passed", "failed"):
            if is_error:
                abs_e += 1
            else:
                abs_n += 1
            continue
        pred_err = v == "failed"
        if is_error and pred_err:
            tp += 1
        elif is_error and not pred_err:
            fn += 1
        elif (not is_error) and pred_err:
            fp += 1
        else:
            tn += 1
    return _rates(tp, fp, tn, fn, abs_e, abs_n)


def _find_judge_key(metrics: Dict[str, Any]) -> str:
    for k in metrics:
        if k.startswith("llm_judge"):
            return k
    raise KeyError("no llm_judge* key in metrics")


HEADERS = [
    "dataset", "grader", "coverage", "precision", "recall", "fpr",
    "abstain_rate", "tp", "fp", "tn", "fn", "abs_e", "abs_n", "total",
]


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    out = [
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---"] * len(HEADERS)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(h)) for h in HEADERS) + " |")
    return "\n".join(out) + "\n"


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
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
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

    rows: List[Dict[str, Any]] = []

    def add(dataset: str, grader: str, row: Dict[str, Any]) -> None:
        rows.append({"dataset": dataset, "grader": grader, **row})

    add("human_labels", "symbolic", _human_row_symbolic(labels))
    add("human_labels", gptoss_model,
        _human_row_from_judge(labels, gptoss_h["verdicts"]))
    add("human_labels", claude_model,
        _human_row_from_judge(labels, claude_h["verdicts"]))

    add("error_injection", "symbolic", _injection_row(gptoss_i, "symbolic"))
    add("error_injection", gptoss_model, _injection_row(gptoss_i, gptoss_key))
    add("error_injection", claude_model, _injection_row(claude_i, claude_key))

    payload = {
        "metric_definitions": {
            "recall": "strict: TP / (TP + FN + AbsE); abstain-on-error is a miss",
            "fpr": "covered: FP / (FP + TN)",
            "coverage": "(TP+FP+TN+FN) / N; crash/unparseable count as abstain",
            "confusion": "TP/FP/TN/FN among committed only; AbsE/AbsN are abstains",
        },
        "graders": ["symbolic", gptoss_model, claude_model],
        "sources": {
            "labels": str(args.labels),
            "gptoss_human": str(args.gptoss_human),
            "claude_human": str(args.claude_human),
            "gptoss_injection": str(args.gptoss_injection),
            "claude_injection": str(args.claude_injection),
        },
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "threeway_table.json").write_text(json.dumps(payload, indent=2))
    with (args.out_dir / "threeway_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "threeway_table.md").write_text(_to_markdown(rows))

    print("Wrote: results/threeway_table.json / .csv / .md")
    print()
    print(_to_markdown(rows))

    # Sanity: every row sums to total; human recall matches 13-error denom.
    for r in rows:
        s = r["tp"] + r["fp"] + r["tn"] + r["fn"] + r["abs_e"] + r["abs_n"]
        assert s == r["total"], r
        if r["dataset"] == "human_labels":
            assert r["tp"] + r["fn"] + r["abs_e"] == 13, r
        if r["dataset"] == "error_injection":
            assert r["total"] == 118, r
            assert r["tp"] + r["fn"] + r["abs_e"] == 94, r
            assert r["fp"] + r["tn"] + r["abs_n"] == 24, r
    print("Sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
