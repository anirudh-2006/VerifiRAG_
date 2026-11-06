#!/usr/bin/env python3
"""
Re-score saved evaluation / ablation JSON with the current check_final_answer_match.

Use when you change verifier grading logic and want to compare old vs new
final_match without re-running LLM calls.

Usage (from telecom-assistant/ with PYTHONPATH=.):

  python scripts/rescore_eval_results.py results/eval_ablation_20260420_220157.json
  python scripts/rescore_eval_results.py results/*.json --write

With --write, updates each file in place (same path) and sets
pipeline.metrics.final_answer_match to the new verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)


def _load_queries_by_id() -> Dict[str, dict]:
    from data.eval_queries import EVAL_QUERIES

    return {q["id"]: q for q in EVAL_QUERIES}


def _resolve_expected(
    row: dict, queries_by_id: Dict[str, dict]
) -> Optional[Any]:
    pipe = row.get("pipeline") or {}
    if pipe.get("expected_answer") is not None:
        return pipe["expected_answer"]
    qid = row.get("id")
    if qid and qid in queries_by_id:
        from src.evaluator import resolve_expected_answer_spec

        return resolve_expected_answer_spec(queries_by_id[qid])
    return None


def rescore_file(path: Path, write: bool) -> List[dict]:
    from src.verifier import check_final_answer_match

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and "results" in data:
        rows = data["results"]
    elif isinstance(data, list):
        rows = data
        data = rows  # write back same shape
    else:
        print(f"{path}: expected dict with 'results' or a list", file=sys.stderr)
        return []
    if not isinstance(rows, list):
        print(f"{path}: results is not a list", file=sys.stderr)
        return []

    queries_by_id = _load_queries_by_id()
    report_rows = []
    changed = False

    for row in rows:
        pipe = row.get("pipeline") or {}
        final = pipe.get("final_result", "")
        exp = _resolve_expected(row, queries_by_id)
        old = (pipe.get("metrics") or {}).get("final_answer_match", {}) or {}
        old_status = old.get("status", "?")

        if exp is None:
            new_status = "no_expected"
            new_match = {"status": new_status, "generated": final, "expected": None}
        else:
            new_match = check_final_answer_match(final, exp)
            new_status = new_match.get("status", "?")

        report_rows.append(
            {
                "file": str(path.name),
                "id": row.get("id"),
                "mode": (row.get("ablation") or {}).get("pipeline_retrieval_mode"),
                "old": old_status,
                "new": new_status,
                "final": (final or "")[:120],
            }
        )

        if new_status != old_status:
            changed = True
        if write and "pipeline" in row and "metrics" in row["pipeline"]:
            row["pipeline"]["metrics"]["final_answer_match"] = new_match

    if write and changed:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote updated scores to {path}")

    return report_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more eval JSON files (schema v2 envelope or bare list)",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Update final_answer_match in each file in place",
    )
    args = ap.parse_args()

    all_rows: List[dict] = []
    for p in args.paths:
        p = p.resolve()
        if not p.is_file():
            print(f"Skip (not a file): {p}", file=sys.stderr)
            continue
        all_rows.extend(rescore_file(p, args.write))

    print(f"\n{'ID':6s} {'mode':7s} {'old':12s} {'new':12s}  file / final")
    print("-" * 100)
    for r in all_rows:
        mode = (r.get("mode") or "") or "-"
        print(
            f"{str(r.get('id') or ''):6s} {str(mode):7s} "
            f"{str(r.get('old')):12s} {str(r.get('new')):12s}  "
            f"{r.get('file')}  {r.get('final', '')[:60]}"
        )

    n_changed = sum(1 for r in all_rows if r["old"] != r["new"])
    print("-" * 100)
    print(f"Rows: {len(all_rows)}  status changes: {n_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
