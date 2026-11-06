#!/usr/bin/env python3
"""Evaluate the symbolic→LLM cascade and emit Table 3.

Cascade rule (prescriptive ``use both''):
  1. If the symbolic verifier commits (passed/failed), take that verdict.
  2. Else (abstain / unparseable / crash), ask the LLM judge.
  3. If the judge also abstains/crashes, abstain.

Metrics match build_threeway_table.py (strict recall, covered FPR).

Outputs:
  results/cascade_table.json / .csv / .md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow `PYTHONPATH=. python scripts/build_cascade_table.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_threeway_table import (  # noqa: E402
    _find_judge_key,
    _fmt,
    _human_row_from_judge,
    _human_row_symbolic,
    _injection_row,
    _load,
    _load_human_labels,
    _rates,
)


def _cascade_verdict(sym: str, judge: str) -> Tuple[str, str]:
    """Return (final_verdict, source) where source in {symbolic, judge, abstain}."""
    if sym in ("passed", "failed"):
        return sym, "symbolic"
    if judge in ("passed", "failed"):
        return judge, "judge"
    return "abstain", "abstain"


def _accumulate(
    pred_err: Optional[bool],  # True=error, False=ok, None=abstain
    is_error: bool,
    counts: Dict[str, int],
) -> None:
    if pred_err is None:
        if is_error:
            counts["abs_e"] += 1
        else:
            counts["abs_n"] += 1
        return
    if is_error and pred_err:
        counts["tp"] += 1
    elif is_error and not pred_err:
        counts["fn"] += 1
    elif (not is_error) and pred_err:
        counts["fp"] += 1
    else:
        counts["tn"] += 1


def _empty_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "abs_e": 0, "abs_n": 0,
            "used_symbolic": 0, "used_judge": 0, "used_abstain": 0}


def cascade_human(
    labels: Dict[Tuple[str, int], Dict[str, Any]],
    verdicts_by_qid: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    judge: Dict[Tuple[str, int], str] = {}
    for qid, steps in verdicts_by_qid.items():
        for s in steps:
            judge[(qid, int(s["step_number"]))] = s["verdict"]

    c = _empty_counts()
    for key, lab in labels.items():
        if key not in judge:
            continue
        is_error = lab.get("human_correct") is False
        v_sym = lab.get("verifier_is_valid")
        if v_sym is True:
            sym = "passed"
        elif v_sym is False:
            sym = "failed"
        else:
            sym = "abstain"
        final, src = _cascade_verdict(sym, judge[key])
        c[f"used_{src}"] += 1
        pred_err = None if final == "abstain" else (final == "failed")
        _accumulate(pred_err, is_error, c)

    row = _rates(c["tp"], c["fp"], c["tn"], c["fn"], c["abs_e"], c["abs_n"])
    row["used_symbolic"] = c["used_symbolic"]
    row["used_judge"] = c["used_judge"]
    row["used_abstain"] = c["used_abstain"]
    return row


def cascade_injection(report: Dict[str, Any], judge_key: str) -> Dict[str, Any]:
    c = _empty_counts()
    for t in report["trials"]:
        if t.get("skipped_reason"):
            continue
        is_error = t["severity"] == "error"
        sym = (t.get("verdicts") or {}).get("symbolic", {}).get("verdict", "error")
        jud = (t.get("verdicts") or {}).get(judge_key, {}).get("verdict", "error")
        # Normalize crashes to abstain before cascade.
        if sym not in ("passed", "failed"):
            sym = "abstain"
        if jud not in ("passed", "failed"):
            jud = "abstain"
        final, src = _cascade_verdict(sym, jud)
        c[f"used_{src}"] += 1
        pred_err = None if final == "abstain" else (final == "failed")
        _accumulate(pred_err, is_error, c)

    row = _rates(c["tp"], c["fp"], c["tn"], c["fn"], c["abs_e"], c["abs_n"])
    row["used_symbolic"] = c["used_symbolic"]
    row["used_judge"] = c["used_judge"]
    row["used_abstain"] = c["used_abstain"]
    return row


HEADERS = [
    "dataset", "grader", "coverage", "precision", "recall", "fpr",
    "abstain_rate", "tp", "fp", "tn", "fn", "abs_e", "abs_n", "total",
    "used_symbolic", "used_judge", "used_abstain",
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
        # Fill routing columns for non-cascade rows.
        for k in ("used_symbolic", "used_judge", "used_abstain"):
            row.setdefault(k, None)
        rows.append({"dataset": dataset, "grader": grader, **row})

    # Baselines (same definitions as Table 1).
    add("human_labels", "symbolic", _human_row_symbolic(labels))
    add("human_labels", gptoss_model,
        _human_row_from_judge(labels, gptoss_h["verdicts"]))
    add("human_labels", claude_model,
        _human_row_from_judge(labels, claude_h["verdicts"]))
    add("human_labels", f"cascade→{gptoss_model}",
        cascade_human(labels, gptoss_h["verdicts"]))
    add("human_labels", f"cascade→{claude_model}",
        cascade_human(labels, claude_h["verdicts"]))

    add("error_injection", "symbolic", _injection_row(gptoss_i, "symbolic"))
    add("error_injection", gptoss_model, _injection_row(gptoss_i, gptoss_key))
    add("error_injection", claude_model, _injection_row(claude_i, claude_key))
    add("error_injection", f"cascade→{gptoss_model}",
        cascade_injection(gptoss_i, gptoss_key))
    add("error_injection", f"cascade→{claude_model}",
        cascade_injection(claude_i, claude_key))

    payload = {
        "rule": "symbolic commits win; else LLM judge; else abstain",
        "metric_definitions": {
            "recall": "strict: TP / (TP + FN + AbsE)",
            "fpr": "covered: FP / (FP + TN)",
            "coverage": "(TP+FP+TN+FN) / N",
        },
        "graders": [
            "symbolic", gptoss_model, claude_model,
            f"cascade→{gptoss_model}", f"cascade→{claude_model}",
        ],
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cascade_table.json").write_text(json.dumps(payload, indent=2))
    with (args.out_dir / "cascade_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        w.writerows(rows)
    md = _to_markdown(rows)
    (args.out_dir / "cascade_table.md").write_text(md)

    print("Wrote: results/cascade_table.json / .csv / .md")
    print(md)

    # Sanity.
    for r in rows:
        s = r["tp"] + r["fp"] + r["tn"] + r["fn"] + r["abs_e"] + r["abs_n"]
        assert s == r["total"], r
        if r["dataset"] == "human_labels":
            assert r["tp"] + r["fn"] + r["abs_e"] == 13, r
            assert r["total"] == 191, r
        if r["dataset"] == "error_injection":
            assert r["total"] == 118, r
            assert r["tp"] + r["fn"] + r["abs_e"] == 94, r
        if str(r["grader"]).startswith("cascade"):
            assert (
                r["used_symbolic"] + r["used_judge"] + r["used_abstain"] == r["total"]
            ), r
    print("Sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
