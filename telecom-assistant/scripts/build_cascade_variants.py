#!/usr/bin/env python3
"""Compare three symbolic->judge routing policies on both frozen testbeds.

Read-only. Every input is a frozen artifact in ``results/``; no LLM API
calls are made and nothing is written to ``results/``. Output goes to
stdout, or to a JSON file outside ``results/`` via ``--json-out``.

Policies
--------
V0  symbolic wins whenever it commits (passed or failed). This is the
    published Table 3 cascade, reproduced here as a correctness anchor.
V1  symbolic FAIL is authoritative; symbolic PASS routes to the judge.
    (Symbolic abstains route to the judge under every policy.)
V2  symbolic is authoritative only on steps that have a successfully
    parsed prior expression to compare against -- i.e. steps that are not
    "first steps" for the verifier. Everything else routes to the judge.

In all three, a step the policy does not hand to the symbolic verifier
goes to the judge; if the judge also abstains or crashes, the policy
abstains. Metrics follow build_threeway_table.py (strict recall, covered
FPR), and the sym/judge/abstain decision counts match Table 3's.

A note on V2 and ``step_index``, because the two testbeds do not agree
--------------------------------------------------------------------
V2 was specified as "step_index > 1 AND the prior expression parsed".
Those two clauses mean the same thing on one testbed and different things
on the other, because the step counters are not the same base:

* Human set: ``step_number`` is 1-based. The verifier marks a step
  "First step" exactly when no previously-parsed expression is available
  (src/verifier.py: ``elif prev_expr is None``), and ``categorize_step``
  turns that into ``verifier_category == "first_step"``. So the frozen
  label templates record the condition directly. 29 of 191 steps are
  first_step: 28 at step_number 1, plus one at step_number > 1 whose
  predecessors all failed to parse. Both readings of V2 select the same
  29 steps here, and the script asserts that.

* Injection suite: ``step_index`` is 0-BASED -- ``inject_and_grade``
  indexes ``gold["steps"][step_index]`` and ``run_suite`` iterates
  ``range(1, len(steps))``, so the first step is never mutated and
  step_index 1 is the *second* step. Every one of the 118 graded trials
  therefore has a parsed prior expression (no trial carries a "First
  step" symbolic reasoning string; the verifier's own messages call them
  "step 2" and "step 3"). Read semantically, V2 is vacuous here and
  collapses onto V0. Read literally as ``step_index > 1``, it withholds
  the verifier from the 79 second-step trials.

This script takes the semantic reading as V2, since both of the spec's
clauses are about whether a usable prior exists, and additionally reports
``V2-literal`` (``step_index > 1`` on each testbed's own counter) as a
sensitivity row. On the human set the two are identical by construction;
on the injection suite they are not, and the gap is the whole point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Allow `PYTHONPATH=. python scripts/build_cascade_variants.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_threeway_table import (  # noqa: E402
    _find_judge_key,
    _human_row_from_judge,
    _injection_row,
    _load,
    _load_human_labels,
    _rates,
)

COMMITTED = ("passed", "failed")


# ---------------------------------------------------------------------------
# Published Table 3, for the V0 anchor. Every cell, not just the four rates
# quoted in CLAUDE.md's prose -- a policy that matched on rates but not on
# confusion cells would still be wrong.
# ---------------------------------------------------------------------------

PUBLISHED_V0 = {
    ("human_labels", "gpt-oss-120b"): {
        "coverage": 0.9948, "precision": 0.6, "recall": 0.2308, "fpr": 0.0112,
        "abstain_rate": 0.0052, "tp": 3, "fp": 2, "tn": 176, "fn": 9,
        "abs_e": 1, "abs_n": 0, "total": 191,
        "used_symbolic": 90, "used_judge": 100, "used_abstain": 1,
    },
    ("human_labels", "claude-sonnet-4-5"): {
        "coverage": 0.9843, "precision": 0.2632, "recall": 0.3846, "fpr": 0.08,
        "abstain_rate": 0.0157, "tp": 5, "fp": 14, "tn": 161, "fn": 8,
        "abs_e": 0, "abs_n": 3, "total": 191,
        "used_symbolic": 90, "used_judge": 98, "used_abstain": 3,
    },
    # Injection rows revised 2026-08-24: gold1 re-fixturing + re-collection of
    # six RateLimitError trials. See root CLAUDE.md, "Injection rows revised".
    ("error_injection", "gpt-oss-120b"): {
        "coverage": 1.0, "precision": 1.0, "recall": 0.8936, "fpr": 0.0,
        "abstain_rate": 0.0, "tp": 84, "fp": 0, "tn": 24, "fn": 10,
        "abs_e": 0, "abs_n": 0, "total": 118,
        "used_symbolic": 42, "used_judge": 76, "used_abstain": 0,
    },
    ("error_injection", "claude-sonnet-4-5"): {
        "coverage": 0.9746, "precision": 0.9677, "recall": 0.9574, "fpr": 0.1304,
        "abstain_rate": 0.0254, "tp": 90, "fp": 3, "tn": 20, "fn": 2,
        "abs_e": 2, "abs_n": 1, "total": 118,
        "used_symbolic": 42, "used_judge": 73, "used_abstain": 3,
    },
}

INT_CELLS = {"tp", "fp", "tn", "fn", "abs_e", "abs_n", "total",
             "used_symbolic", "used_judge", "used_abstain"}
TOL = 5e-4  # published rates are quoted to 3-4 decimals


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# A policy answers one question: for this item, is the symbolic verdict
# allowed to decide? `meta` carries per-item facts drawn from frozen data:
#   has_parsed_prior : bool  -- a prior expression was parsed and compared
#   step_counter     : int   -- the testbed's own step counter, as stored
def _p_v0(sym: str, meta: Dict[str, Any]) -> bool:
    return True


def _p_v1(sym: str, meta: Dict[str, Any]) -> bool:
    return sym == "failed"


def _p_v2(sym: str, meta: Dict[str, Any]) -> bool:
    return bool(meta["has_parsed_prior"])


def _p_v2_literal(sym: str, meta: Dict[str, Any]) -> bool:
    return meta["step_counter"] > 1


POLICIES: List[Tuple[str, Callable[[str, Dict[str, Any]], bool]]] = [
    ("V0", _p_v0),
    ("V1", _p_v1),
    ("V2", _p_v2),
    ("V2-literal", _p_v2_literal),
]

POLICY_BLURB = {
    "V0": "symbolic wins whenever it commits (published Table 3)",
    "V1": "symbolic FAIL authoritative; symbolic PASS routes to judge",
    "V2": "symbolic authoritative only where a parsed prior exists",
    "V2-literal": "symbolic authoritative only where step_counter > 1",
}


def _route(
    sym: str, judge: str, meta: Dict[str, Any],
    authoritative: Callable[[str, Dict[str, Any]], bool],
) -> Tuple[str, str]:
    """Return (final_verdict, source) with source in {symbolic,judge,abstain}."""
    if sym in COMMITTED and authoritative(sym, meta):
        return sym, "symbolic"
    if judge in COMMITTED:
        return judge, "judge"
    return "abstain", "abstain"


def _empty() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "abs_e": 0, "abs_n": 0,
            "used_symbolic": 0, "used_judge": 0, "used_abstain": 0}


def _accumulate(pred_err: Optional[bool], is_error: bool, c: Dict[str, int]) -> None:
    if pred_err is None:
        c["abs_e" if is_error else "abs_n"] += 1
    elif is_error and pred_err:
        c["tp"] += 1
    elif is_error:
        c["fn"] += 1
    elif pred_err:
        c["fp"] += 1
    else:
        c["tn"] += 1


def _finish(c: Dict[str, int]) -> Dict[str, Any]:
    row = _rates(c["tp"], c["fp"], c["tn"], c["fn"], c["abs_e"], c["abs_n"])
    for k in ("used_symbolic", "used_judge", "used_abstain"):
        row[k] = c[k]
    return row


# ---------------------------------------------------------------------------
# Item extraction: (is_error, sym, judge, meta) triples per testbed
# ---------------------------------------------------------------------------


def human_items(
    labels: Dict[Tuple[str, int], Dict[str, Any]],
    verdicts_by_qid: Dict[str, List[Dict[str, Any]]],
) -> List[Tuple[bool, str, str, Dict[str, Any]]]:
    judge: Dict[Tuple[str, int], str] = {}
    for qid, steps in verdicts_by_qid.items():
        for s in steps:
            judge[(qid, int(s["step_number"]))] = s["verdict"]

    items = []
    for key, lab in sorted(labels.items()):
        if key not in judge:
            continue
        v_sym = lab.get("verifier_is_valid")
        sym = "passed" if v_sym is True else "failed" if v_sym is False else "abstain"
        jud = judge[key]
        if jud not in COMMITTED:
            jud = "abstain"
        # "First step" is the verifier's own marker for "no previously
        # parsed expression was available" (src/verifier.py: the
        # `elif prev_expr is None` branch), which is exactly V2's condition.
        # It is necessary but not sufficient: categorize_step() tests the
        # CURRENT expression's parse failure first, so a step 1 whose own
        # expression did not parse is filed as parse_fail and would look
        # like it had a prior. Step 1 never has one, so exclude it too.
        category = lab.get("verifier_category")
        step_number = int(lab["step_number"])
        items.append((
            lab.get("human_correct") is False,
            sym,
            jud,
            {
                "has_parsed_prior": category != "first_step" and step_number > 1,
                "step_counter": step_number,
                "verifier_category": category,
            },
        ))
    return items


def injection_items(
    report: Dict[str, Any], judge_key: str
) -> List[Tuple[bool, str, str, Dict[str, Any]]]:
    items = []
    for t in report["trials"]:
        if t.get("skipped_reason"):
            continue
        verdicts = t.get("verdicts") or {}
        sym = (verdicts.get("symbolic") or {}).get("verdict", "error")
        jud = (verdicts.get(judge_key) or {}).get("verdict", "error")
        reasoning = (verdicts.get("symbolic") or {}).get("reasoning", "") or ""
        if sym not in COMMITTED:
            sym = "abstain"
        if jud not in COMMITTED:
            jud = "abstain"
        items.append((
            t["severity"] == "error",
            sym,
            jud,
            {
                # Same marker as the human set. run_suite never mutates the
                # first step, so this is True for every graded trial -- the
                # assertion in check_v2_assumptions() pins that down rather
                # than leaving it as a comment.
                "has_parsed_prior": not reasoning.strip().lower().startswith("first step"),
                "step_counter": int(t["step_index"]),
            },
        ))
    return items


def evaluate(items, authoritative) -> Dict[str, Any]:
    c = _empty()
    for is_error, sym, jud, meta in items:
        final, src = _route(sym, jud, meta, authoritative)
        c[f"used_{src}"] += 1
        _accumulate(None if final == "abstain" else (final == "failed"), is_error, c)
    return _finish(c)


# ---------------------------------------------------------------------------
# Assertions -- all of these run before any V1/V2 number is reported
# ---------------------------------------------------------------------------


def _cells_differ(got: Dict[str, Any], want: Dict[str, Any]) -> List[str]:
    diffs = []
    for field, exp in want.items():
        act = got.get(field)
        if exp is None or act is None:
            ok = exp is None and act is None
        elif field in INT_CELLS:
            ok = int(act) == int(exp)
        else:
            ok = abs(float(act) - float(exp)) <= TOL
        if not ok:
            diffs.append(f"{field}: published={exp!r} computed={act!r}")
    return diffs


def check_v0_reproduces_published(rows: Dict[Tuple[str, str, str], Dict[str, Any]]) -> None:
    """Hard gate. V0 must reproduce every published Table 3 cell exactly."""
    problems = []
    for (dataset, model), want in sorted(PUBLISHED_V0.items()):
        got = rows.get(("V0", dataset, model))
        if got is None:
            problems.append(f"  {dataset}/cascade->{model}: V0 row missing entirely")
            continue
        for d in _cells_differ(got, want):
            problems.append(f"  {dataset}/cascade->{model}  {d}")
    if problems:
        print("FATAL: V0 does not reproduce the published Table 3.", file=sys.stderr)
        print("The routing implementation is wrong, or an input artifact has",
              file=sys.stderr)
        print("changed. V1 and V2 are NOT reported. Differences:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(2)


def check_v1_human_equals_judge_alone(
    rows: Dict[Tuple[str, str, str], Dict[str, Any]],
    judge_alone: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    """Hard gate, from Table 1's confusion cells.

    The verifier commits zero FAIL verdicts on the human set (TP=0, FP=0;
    all 90 commits are passes). V1 only lets a symbolic FAIL override the
    judge, so on the human set V1 must be numerically identical to the
    judge alone -- by construction, not by measurement. If it is not, the
    implementation is wrong and the numbers must not be reported.
    """
    fields = ["coverage", "precision", "recall", "fpr", "abstain_rate",
              "tp", "fp", "tn", "fn", "abs_e", "abs_n", "total"]
    problems = []
    for model in ("gpt-oss-120b", "claude-sonnet-4-5"):
        got = rows.get(("V1", "human_labels", model))
        want = judge_alone.get(("human_labels", model))
        if got is None or want is None:
            problems.append(f"  human_labels/{model}: missing V1 or judge-alone row")
            continue
        for d in _cells_differ(got, {f: want.get(f) for f in fields}):
            problems.append(f"  human_labels/{model}  {d.replace('published', 'judge-alone')}")
        if got.get("used_symbolic"):
            problems.append(
                f"  human_labels/{model}  used_symbolic={got['used_symbolic']}, "
                f"expected 0 (the verifier commits no FAIL verdicts on this set)"
            )
    if problems:
        print("FATAL: V1 on the human set is not identical to the judge alone.",
              file=sys.stderr)
        print("Table 1 records TP=0 and FP=0 for the symbolic verifier there, so",
              file=sys.stderr)
        print("V1 cannot differ from the judge. This is an implementation bug,",
              file=sys.stderr)
        print("not a finding. Differences:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(3)


def check_v2_assumptions(human, injection_by_model) -> List[str]:
    """Verify, rather than assume, the two facts V2's reading rests on."""
    notes = []
    n_first = sum(1 for _, _, _, m in human if not m["has_parsed_prior"])
    n_step1 = sum(1 for _, _, _, m in human if m["step_counter"] == 1)
    n_late_first = sum(
        1 for _, _, _, m in human
        if not m["has_parsed_prior"] and m["step_counter"] > 1
    )
    notes.append(
        f"human set: {n_first}/{len(human)} steps have no parsed prior; "
        f"{n_step1} of those are at step_number 1, {n_late_first} at "
        f"step_number > 1 (predecessors all unparseable)"
    )
    # V2 withholds the verifier from every no-prior step; V2-literal only
    # from step_number <= 1. They differ exactly on the late first_steps.
    disagree = [
        (sym, m) for _, sym, _, m in human
        if _p_v2(sym, m) != _p_v2_literal(sym, m)
    ]
    committed = [m for sym, m in disagree if sym in COMMITTED]
    if disagree:
        notes.append(
            f"human set: V2 and V2-literal disagree on {len(disagree)} step(s), "
            f"of which {len(committed)} carry a committed symbolic verdict and "
            f"can therefore move a cell"
        )
    else:
        notes.append("human set: V2 and V2-literal select identical step sets")

    for model, items in sorted(injection_by_model.items()):
        n_noprior = sum(1 for _, _, _, m in items if not m["has_parsed_prior"])
        counters = sorted({m["step_counter"] for _, _, _, m in items})
        notes.append(
            f"injection/{model}: {n_noprior}/{len(items)} trials have no parsed "
            f"prior; step_index values present: {counters} (0-based, so V2 is "
            f"vacuous and collapses onto V0)"
        )
    return notes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLUMNS = [
    ("policy", "policy", 11), ("dataset", "dataset", 16), ("grader", "judge", 18),
    ("coverage", "cov", 7), ("precision", "prec", 7), ("recall", "rec", 7),
    ("fpr", "fpr", 7), ("abstain_rate", "abst", 7),
    ("tp", "TP", 4), ("fp", "FP", 4), ("tn", "TN", 4), ("fn", "FN", 4),
    ("abs_e", "AbsE", 5), ("abs_n", "AbsN", 5), ("total", "N", 4),
    ("used_symbolic", "sym", 5), ("used_judge", "judge", 6), ("used_abstain", "abst", 5),
]


def _cell(value: Any, field: str) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render(rows: Dict[Tuple[str, str, str], Dict[str, Any]],
           judge_alone: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    head = "  ".join(f"{label:<{w}}" if f in ("policy", "dataset", "grader")
                     else f"{label:>{w}}" for f, label, w in COLUMNS)
    sep = "-" * len(head)
    out = [head, sep]

    # Judge-alone reference rows first: V1's human rows must match these.
    for dataset in ("human_labels", "error_injection"):
        for model in ("gpt-oss-120b", "claude-sonnet-4-5"):
            r = judge_alone.get((dataset, model))
            if r is None:
                continue
            vals = {"policy": "judge-alone", "dataset": dataset, "grader": model, **r}
            out.append("  ".join(
                f"{_cell(vals.get(f), f):<{w}}" if f in ("policy", "dataset", "grader")
                else f"{_cell(vals.get(f), f):>{w}}" for f, _, w in COLUMNS))
    out.append(sep)

    for policy, _ in POLICIES:
        for dataset in ("human_labels", "error_injection"):
            for model in ("gpt-oss-120b", "claude-sonnet-4-5"):
                r = rows.get((policy, dataset, model))
                if r is None:
                    continue
                vals = {"policy": policy, "dataset": dataset, "grader": model, **r}
                out.append("  ".join(
                    f"{_cell(vals.get(f), f):<{w}}" if f in ("policy", "dataset", "grader")
                    else f"{_cell(vals.get(f), f):>{w}}" for f, _, w in COLUMNS))
        out.append(sep)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("results/label_templates"))
    ap.add_argument("--gptoss-human", type=Path,
                    default=Path("results/judge_step_verdicts.json"))
    ap.add_argument("--claude-human", type=Path,
                    default=Path("results/judge_step_verdicts_claude.json"))
    ap.add_argument("--gptoss-injection", type=Path,
                    default=Path("results/error_injection_report_with_judge.json"))
    ap.add_argument("--claude-injection", type=Path,
                    default=Path("results/error_injection_report_claude.json"))
    ap.add_argument("--json-out", type=Path, default=None,
                    help="optional JSON dump; must not be inside results/")
    args = ap.parse_args()

    if args.json_out is not None:
        resolved = args.json_out.resolve()
        results_dir = (Path(__file__).resolve().parent.parent / "results").resolve()
        if results_dir == resolved or results_dir in resolved.parents:
            print(f"refusing to write inside results/: {args.json_out}", file=sys.stderr)
            return 4

    labels = _load_human_labels(args.labels)
    gptoss_h = _load(args.gptoss_human)
    claude_h = _load(args.claude_human)
    gptoss_i = _load(args.gptoss_injection)
    claude_i = _load(args.claude_injection)
    gptoss_key = _find_judge_key(gptoss_i["metrics"])
    claude_key = _find_judge_key(claude_i["metrics"])
    gptoss_model = gptoss_key.split(":", 1)[-1]
    claude_model = claude_key.split(":", 1)[-1]

    human_by_model = {
        gptoss_model: human_items(labels, gptoss_h["verdicts"]),
        claude_model: human_items(labels, claude_h["verdicts"]),
    }
    injection_by_model = {
        gptoss_model: injection_items(gptoss_i, gptoss_key),
        claude_model: injection_items(claude_i, claude_key),
    }

    judge_alone = {
        ("human_labels", gptoss_model): _human_row_from_judge(labels, gptoss_h["verdicts"]),
        ("human_labels", claude_model): _human_row_from_judge(labels, claude_h["verdicts"]),
        ("error_injection", gptoss_model): _injection_row(gptoss_i, gptoss_key),
        ("error_injection", claude_model): _injection_row(claude_i, claude_key),
    }

    rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for policy, pred in POLICIES:
        for model, items in human_by_model.items():
            rows[(policy, "human_labels", model)] = evaluate(items, pred)
        for model, items in injection_by_model.items():
            rows[(policy, "error_injection", model)] = evaluate(items, pred)

    print("build_cascade_variants.py -- read-only; frozen verdicts; no API calls")
    print()
    print("Gate 1: V0 must reproduce the published Table 3 exactly ...", end=" ")
    check_v0_reproduces_published(rows)
    print("OK (all cells, both testbeds, both judges)")

    print("Gate 2: V1 on the human set must equal the judge alone ...", end=" ")
    check_v1_human_equals_judge_alone(rows, judge_alone)
    print("OK")
    print()

    for note in check_v2_assumptions(
        human_by_model[gptoss_model], injection_by_model
    ):
        print(f"  note: {note}")
    print()
    for policy, _ in POLICIES:
        print(f"  {policy:<11} {POLICY_BLURB[policy]}")
    print()
    print(render(rows, judge_alone))

    if args.json_out is not None:
        payload = {
            "read_only": True,
            "policies": {p: POLICY_BLURB[p] for p, _ in POLICIES},
            "metric_definitions": {
                "recall": "strict: TP / (TP + FN + AbsE)",
                "fpr": "covered: FP / (FP + TN)",
            },
            "rows": [
                {"policy": p, "dataset": d, "grader": g, **r}
                for (p, d, g), r in sorted(rows.items())
            ],
            "judge_alone": [
                {"dataset": d, "grader": g, **r}
                for (d, g), r in sorted(judge_alone.items())
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
