#!/usr/bin/env python3
"""Smoke test the EVAL_QUERIES benchmark.

Checks:
- all queries load
- expected_answer resolves for every query
- concept and type coverage summary
- round-trip: feed expected_final_result into check_final_answer_match
  against expected_answer; for a sane benchmark this should match (or at
  minimum not crash and not silently return wrong statuses).

Run:
    ./.venv/bin/python scripts/smoke_eval_queries.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from data.eval_queries import EVAL_QUERIES  # noqa: E402
from src.evaluator import resolve_expected_answer_spec  # noqa: E402
from src.verifier import check_final_answer_match  # noqa: E402


def main() -> int:
    print(f"Total queries: {len(EVAL_QUERIES)}")

    cc: Counter = Counter()
    tc: Counter = Counter()
    dc: Counter = Counter()
    for q in EVAL_QUERIES:
        for c in q.get("expected_concepts", []):
            cc[c] += 1
        ea = q.get("expected_answer", {}) or {}
        t = ea.get("type", "legacy")
        if t == "composite":
            t = f"composite_{ea.get('aggregation','?')}"
        tc[t] += 1
        dc[q.get("difficulty", "?")] += 1

    print("\nConcept coverage:")
    for c, n in sorted(cc.items(), key=lambda kv: -kv[1]):
        print(f"  {c:25s} {n}")
    print("\nExpected-answer types:")
    for t, n in sorted(tc.items(), key=lambda kv: -kv[1]):
        print(f"  {t:20s} {n}")
    print("\nDifficulty:")
    for d, n in sorted(dc.items(), key=lambda kv: -kv[1]):
        print(f"  {d:15s} {n}")

    print("\nRound-trip (expected_final_result vs expected_answer):")
    fails = []
    for q in EVAL_QUERIES:
        qid = q["id"]
        spec = resolve_expected_answer_spec(q)
        gen = q.get("expected_final_result", "")
        try:
            res = check_final_answer_match(gen, spec)
            st = res.get("status")
            mark = "OK" if st == "match" else "??"
            print(f"  {mark}  {qid}: status={st:12s} gen={gen[:55]!r}")
            if st != "match":
                fails.append((qid, st, gen, res))
        except Exception as e:
            print(f"  EX  {qid}: {type(e).__name__}: {e}")
            fails.append((qid, "EXCEPTION", gen, None))

    n_ok = len(EVAL_QUERIES) - len(fails)
    print(f"\nRound-trip matches: {n_ok}/{len(EVAL_QUERIES)}")

    if fails:
        print("\nNon-matches detail:")
        for qid, st, gen, res in fails:
            print(f"\n  {qid}: {st}")
            print(f"    gen: {gen!r}")
            if res:
                print(f"    gen_parsed: {res.get('generated_parsed')!r}")
                print(f"    exp_parsed: {res.get('expected_parsed')!r}")
                parts = res.get("parts") or []
                for p in parts:
                    print(
                        f"    part {p.get('name', '?')}: status={p.get('status')} "
                        f"gen={p.get('generated')!r} exp={p.get('expected')!r}"
                    )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
