"""
Score the symbolic verifier against human ground-truth labels.

Inputs
------
- results/label_templates/Q??.json : one JSON per query.
  Produced automatically by evaluator.evaluate_single_query and then
  filled in by a human (set '_filled': true, per-step 'human_correct'
  to true/false, 'error_type' to one of {algebra, sign, missing_term,
  substitution, definition, none}, and 'final_answer_correct' to
  true/false).

What it computes
----------------
Per step, we pair (verifier_verdict, human_correct):

  verifier_verdict = "correct"     if is_valid is True
                     "incorrect"   if is_valid is False
                     "abstain"     if is_valid is None

  human_correct    = True / False  (from label file)

From the non-abstain rows we compute, treating "incorrect" as the
positive class:

  precision = TP / (TP + FP)   (of the steps the verifier flagged,
                                how many were actually wrong?)
  recall    = TP / (TP + FN)   (of the actually-wrong steps, how
                                many did the verifier catch?)

We also report:
  abstain_rate     = # abstains / # total steps
  coverage         = 1 - abstain_rate
  accuracy_covered = correct decisions / covered steps
  final_answer_accuracy = per-query flag vs human final_answer_correct

Usage
-----
  python -m src.label_scorer
  python -m src.label_scorer results/label_templates
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any

DEFAULT_DIR = "results/label_templates"


def load_labels(label_dir: str) -> list[dict[str, Any]]:
    """Load every *.json in `label_dir` that has _filled == true."""
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    out = []
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(label_dir, name)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  skipping {name}: {e}")
            continue
        if not data.get("_filled"):
            continue
        out.append(data)
    return out


def score(
    labeled: list[dict[str, Any]],
    *,
    valid_key: str = "verifier_is_valid",
    grader_label: str = "verifier",
) -> dict[str, Any]:
    """Score any grader against the human-labelled steps.

    `valid_key` selects which per-step field carries the grader's True/False/
    None verdict. `grader_label` only affects display and the namespaced
    breakdown keys in `grader_verdicts` (e.g. ``judge_correct``).
    """
    tp = fp = tn = fn = 0
    abstain = 0
    total = 0
    verdicts = Counter()
    error_types = Counter()

    final_total = 0
    final_correct_decisions = 0

    for q in labeled:
        for s in q.get("steps", []):
            total += 1
            v = s.get(valid_key)
            h = s.get("human_correct")
            if h is None:
                # step wasn't labelled — count as abstain on both sides
                abstain += 1
                continue

            if v is True:
                verdicts[f"{grader_label}_correct"] += 1
                if h is True:
                    tn += 1  # both say correct
                else:
                    fn += 1  # grader missed an error
            elif v is False:
                verdicts[f"{grader_label}_incorrect"] += 1
                if h is False:
                    tp += 1  # correctly flagged
                else:
                    fp += 1  # false alarm
            else:
                verdicts[f"{grader_label}_abstain"] += 1
                abstain += 1

            if h is False:
                et = (s.get("error_type") or "unspecified").lower()
                error_types[et] += 1

        gen = q.get("generated_final", "")
        exp = q.get("expected_final", "")
        human_final = q.get("final_answer_correct")
        if human_final is not None and exp:
            final_total += 1
            # The evaluator JSON records the grader's decision in the
            # main results file — for scoring we only need the human label
            # vs "did the generated string match expected symbolically?"
            # The easiest fair comparison: treat human label as truth.
            # Here we just count how often the human said it was correct.
            if human_final:
                final_correct_decisions += 1

    covered = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall    = tp / (tp + fn) if (tp + fn) else None
    accuracy_covered = (tp + tn) / covered if covered else None
    abstain_rate = abstain / total if total else 0.0
    coverage = 1 - abstain_rate

    report = {
        "grader_label":              grader_label,
        "queries_labelled":          len(labeled),
        "steps_total":               total,
        "steps_abstained":           abstain,
        "coverage":                  round(coverage, 3),
        "confusion_matrix": {
            "true_positive":  tp,
            "false_positive": fp,
            "true_negative":  tn,
            "false_negative": fn,
        },
        "precision_error_detection": round(precision, 3) if precision is not None else None,
        "recall_error_detection":    round(recall, 3)    if recall    is not None else None,
        "accuracy_on_covered_steps": round(accuracy_covered, 3) if accuracy_covered is not None else None,
        "grader_verdicts":           dict(verdicts),
        "human_error_types":         dict(error_types),
        "final_answer_queries":      final_total,
        "final_answer_correct":      final_correct_decisions,
    }
    # Backwards-compat alias so older callers / artifacts keep working.
    if grader_label == "verifier":
        report["verifier_verdicts"] = dict(verdicts)
    return report


def print_report(report: dict[str, Any]) -> None:
    label = (report.get("grader_label") or "verifier").upper()
    print("=" * 72)
    print(f"{label} vs HUMAN GROUND TRUTH")
    print("=" * 72)
    print(f"Queries labelled          : {report['queries_labelled']}")
    print(f"Total steps               : {report['steps_total']}")
    print(f"Abstained (no verdict)    : {report['steps_abstained']} "
          f"(coverage = {report['coverage']*100:.1f}%)")
    print()
    cm = report["confusion_matrix"]
    print("Confusion matrix (on covered steps):")
    print(f"               human_wrong  human_right")
    print(f"  flagged       {cm['true_positive']:>6}        {cm['false_positive']:>6}")
    print(f"  not-flagged   {cm['false_negative']:>6}        {cm['true_negative']:>6}")
    print()
    print(f"Error-detection precision : {report['precision_error_detection']}")
    print(f"Error-detection recall    : {report['recall_error_detection']}")
    print(f"Accuracy on covered steps : {report['accuracy_on_covered_steps']}")
    print()
    if report["human_error_types"]:
        print("Human-annotated error types:")
        for et, n in sorted(report["human_error_types"].items(),
                            key=lambda kv: -kv[1]):
            print(f"  {et:<20} {n}")
    print()
    print(f"Final-answer labelled queries: {report['final_answer_queries']}")
    print(f"Final-answer judged correct  : {report['final_answer_correct']}")
    print("=" * 72)


def main() -> None:
    label_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    labeled = load_labels(label_dir)
    if not labeled:
        print(f"No filled label files found in {label_dir}.")
        print("Set '_filled': true in a template after you annotate it.")
        return
    report = score(labeled)
    print_report(report)
    out_path = os.path.join(os.path.dirname(label_dir) or ".",
                            "label_score_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved machine-readable report to: {out_path}")


if __name__ == "__main__":
    main()
