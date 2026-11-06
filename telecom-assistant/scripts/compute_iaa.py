"""
Compute inter-annotator agreement (IAA) between two annotators on the overlap
subset produced by scripts/make_iaa_subset.py.

Annotator 1 = the frozen main ground truth (results/label_templates/, or a
snapshot copy). Annotator 2 = the blinded re-labels (results/iaa_subset/) after
a second human filled them in.

We match steps by (qid, step_number) over the intersection of steps that BOTH
annotators labelled (human_correct in {true,false}), then report:

  - n overlap steps compared
  - percent agreement on the binary pass/fail decision
  - Cohen's kappa on the binary decision
  - Cohen's kappa on error_type (restricted to steps BOTH marked false)
  - the full disagreement list, for adjudication

No third-party deps: kappa is computed directly from the confusion counts.

Usage
-----
  cd telecom-assistant
  PYTHONPATH=. ./.venv/bin/python scripts/compute_iaa.py \
      --annotator1 results/label_templates.annotator1 \
      --annotator2 results/iaa_subset \
      --out results/iaa_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stats import cohen_kappa as _cohen_kappa  # noqa: E402
from src.stats import landis_koch_label as _landis_koch  # noqa: E402


def _load_dir(label_dir: str, *, require_filled: bool) -> Dict[str, Dict[str, Any]]:
    """Return {qid: query} for every *.json (optionally only _filled ones)."""
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".json") or name.startswith("_"):
            continue
        with open(os.path.join(label_dir, name)) as f:
            data = json.load(f)
        if require_filled and not data.get("_filled"):
            continue
        qid = data.get("id") or os.path.splitext(name)[0]
        out[qid] = data
    return out


def _step_index(query: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    return {
        s.get("step_number"): s
        for s in query.get("steps", []) or []
        if s.get("step_number") is not None
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute IAA (Cohen's kappa).")
    ap.add_argument("--annotator1", default="results/label_templates.annotator1",
                    help="Frozen main ground truth (filled).")
    ap.add_argument("--annotator2", default="results/iaa_subset",
                    help="Blinded re-labels filled by annotator 2.")
    ap.add_argument("--out", default="results/iaa_report.json")
    args = ap.parse_args()

    a1 = _load_dir(args.annotator1, require_filled=True)
    a2 = _load_dir(args.annotator2, require_filled=True)
    if not a2:
        print(f"No _filled label files in {args.annotator2}. "
              "Has annotator 2 finished and set _filled:true?")
        return

    binary_pairs: List[Tuple[bool, bool]] = []
    errtype_pairs: List[Tuple[str, str]] = []
    disagreements: List[Dict[str, Any]] = []
    n_only_a1 = n_only_a2 = 0

    for qid, q2 in sorted(a2.items()):
        q1 = a1.get(qid)
        if q1 is None:
            continue
        s1 = _step_index(q1)
        s2 = _step_index(q2)
        for step_no, st2 in s2.items():
            st1 = s1.get(step_no)
            if st1 is None:
                continue
            h1 = st1.get("human_correct")
            h2 = st2.get("human_correct")
            if h1 is None and h2 is not None:
                n_only_a2 += 1
            if h2 is None and h1 is not None:
                n_only_a1 += 1
            if not isinstance(h1, bool) or not isinstance(h2, bool):
                continue

            binary_pairs.append((h1, h2))
            if h1 != h2:
                disagreements.append({
                    "qid": qid,
                    "step_number": step_no,
                    "annotator1": {"human_correct": h1, "error_type": st1.get("error_type")},
                    "annotator2": {"human_correct": h2, "error_type": st2.get("error_type")},
                    "expression": st1.get("expression_given", ""),
                    "description": st1.get("description", ""),
                })

            # error_type kappa only where BOTH called it an error (h==False).
            if h1 is False and h2 is False:
                et1 = (st1.get("error_type") or "unspecified").lower()
                et2 = (st2.get("error_type") or "unspecified").lower()
                errtype_pairs.append((et1, et2))

    n = len(binary_pairs)
    if n == 0:
        print("No overlapping steps where BOTH annotators committed a "
              "true/false label. Nothing to score.")
        return

    n_agree = sum(1 for a, b in binary_pairs if a == b)
    pct_agree = n_agree / n
    kappa_binary = _cohen_kappa(binary_pairs)
    kappa_errtype = _cohen_kappa(errtype_pairs) if errtype_pairs else None

    # Confusion of the binary decision (rows = annotator1, cols = annotator2).
    conf = Counter((a, b) for a, b in binary_pairs)

    report = {
        "annotator1_dir": args.annotator1,
        "annotator2_dir": args.annotator2,
        "n_overlap_steps": n,
        "percent_agreement": round(pct_agree, 4),
        "cohen_kappa_binary": round(kappa_binary, 4) if kappa_binary is not None else None,
        "cohen_kappa_binary_interpretation": _landis_koch(kappa_binary),
        "n_error_type_pairs": len(errtype_pairs),
        "cohen_kappa_error_type": round(kappa_errtype, 4) if kappa_errtype is not None else None,
        "cohen_kappa_error_type_interpretation": _landis_koch(kappa_errtype),
        "binary_confusion_a1_rows_a2_cols": {
            "both_correct(True,True)":   conf[(True, True)],
            "a1_correct_a2_error(True,False)":  conf[(True, False)],
            "a1_error_a2_correct(False,True)":  conf[(False, True)],
            "both_error(False,False)":   conf[(False, False)],
        },
        "n_labelled_only_by_a1": n_only_a1,
        "n_labelled_only_by_a2": n_only_a2,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 72)
    print(f"Overlap steps compared     : {n}")
    print(f"Percent agreement          : {pct_agree*100:.1f}%")
    print(f"Cohen's kappa (pass/fail)  : {kappa_binary:.3f}  "
          f"({_landis_koch(kappa_binary)})")
    if kappa_errtype is not None:
        print(f"Cohen's kappa (error_type) : {kappa_errtype:.3f}  "
              f"({_landis_koch(kappa_errtype)})  over {len(errtype_pairs)} both-error steps")
    else:
        print("Cohen's kappa (error_type) : n/a (no steps both marked error)")
    print(f"Disagreements              : {len(disagreements)} "
          "(see report for adjudication list)")
    print(f"\nSaved report to: {args.out}")


if __name__ == "__main__":
    main()
