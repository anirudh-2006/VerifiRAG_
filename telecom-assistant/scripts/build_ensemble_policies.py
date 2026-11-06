#!/usr/bin/env python3
"""Three judge-ensemble policies over the two LLM judges, on both testbeds.

Read-only. Every input is a frozen artifact in ``results/``; no LLM API
calls are made and nothing is written to ``results/``. Output goes to
stdout, or to a JSON file outside ``results/`` via ``--json-out``.

The symbolic verifier plays no part here. These policies combine
gpt-oss-120b and claude-sonnet-4-5 with each other; symbolic->judge
routing is build_cascade_variants.py's job.

Policies and their abstain handling
-----------------------------------
Each judge's verdict is reduced to one of {failed, passed, abstain}, where
abstain absorbs a crash or unparseable reply, exactly as in Table 1.

UNION         flag if either judge commits FAIL.
              One FAIL + one ABSTAIN COUNTS AS A FLAG.
              Commits PASS when at least one judge commits and neither
              commits FAIL. Abstains only when both judges abstain.

INTERSECTION  flag only if both judges commit FAIL.
              One FAIL + one ABSTAIN DOES NOT COUNT AS A FLAG: the item is
              committed as PASS, not abstained. Abstains only when both
              judges abstain, so INTERSECTION covers exactly the same items
              as UNION and differs from it only in how it labels them.

AGREEMENT     commit only where both judges commit AND agree; abstain
              otherwise. ANY ABSTAIN FROM EITHER JUDGE MEANS NO AGREEMENT,
              so the item is abstained -- as is any both-committed
              disagreement.

Two consequences of the INTERSECTION rule are worth stating rather than
leaving for a reader to discover. It converts a lone committed FAIL into a
committed PASS, which becomes a false negative when the step really was an
error; and because a lone FAIL and a lone PASS are both committed, its
coverage is identical to UNION's on both testbeds. The script reports how
many items land in that lone-FAIL cell.

Gates -- all run before any ensemble number is printed
------------------------------------------------------
1. Standalone judge rows must reproduce published Table 1 exactly, every
   cell, both testbeds, both judges. Otherwise: exit 2, no ensemble rows.
2. Both judges must have a verdict record for every item, so that abstains
   are verdict-level facts rather than missing data. Otherwise: exit 3.
3. Human-set ensemble recalls must match the counts quoted in the paper's
   Analysis section (both catch 4, claude alone 6, gpt-oss alone 2,
   neither 1): UNION 12/13, INTERSECTION 4/13, AGREEMENT 4/13 with
   AbsE 8. On a mismatch the discrepancy is reported and the ensemble
   numbers are suppressed: exit 4.

Metric definitions are build_threeway_table.py's, unchanged: strict recall
TP/(TP+FN+AbsE), covered FPR FP/(FP+TN), coverage (TP+FP+TN+FN)/N.

Inputs (defaults):
  results/label_templates/Q*.json
  results/judge_step_verdicts.json
  results/judge_step_verdicts_claude.json
  results/error_injection_report_with_judge.json
  results/error_injection_report_claude.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Allow `PYTHONPATH=. python scripts/build_ensemble_policies.py`.
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
# Published Table 1 (CLAUDE.md), for the standalone-judge gate. Every cell,
# not only the rates: a policy that matched on rates but not on confusion
# cells would still be wrong. AbsN is implied by the row totals.
# ---------------------------------------------------------------------------

PUBLISHED_TABLE1 = {
    ("human_labels", "gpt-oss-120b"): {
        "coverage": 0.995, "precision": 0.667, "recall": 0.462, "fpr": 0.017,
        "abstain_rate": 0.005, "tp": 6, "fp": 3, "tn": 175, "fn": 6,
        "abs_e": 1, "abs_n": 0, "total": 191,
    },
    ("human_labels", "claude-sonnet-4-5"): {
        "coverage": 0.953, "precision": 0.370, "recall": 0.769, "fpr": 0.100,
        "abstain_rate": 0.047, "tp": 10, "fp": 17, "tn": 153, "fn": 2,
        "abs_e": 1, "abs_n": 8, "total": 191,
    },
    # Injection rows revised 2026-08-24: gold1 re-fixturing + re-collection of
    # six RateLimitError trials. See root CLAUDE.md, "Injection rows revised".
    ("error_injection", "gpt-oss-120b"): {
        "coverage": 1.000, "precision": 1.000, "recall": 0.894, "fpr": 0.000,
        "abstain_rate": 0.000, "tp": 84, "fp": 0, "tn": 24, "fn": 10,
        "abs_e": 0, "abs_n": 0, "total": 118,
    },
    ("error_injection", "claude-sonnet-4-5"): {
        "coverage": 0.958, "precision": 0.938, "recall": 0.957, "fpr": 0.286,
        "abstain_rate": 0.042, "tp": 90, "fp": 6, "tn": 15, "fn": 2,
        "abs_e": 2, "abs_n": 3, "total": 118,
    },
}

# Human-set expectations from the Analysis section's per-error counts.
EXPECTED_HUMAN = {
    "UNION": {"tp": 12, "recall": 12 / 13},
    "INTERSECTION": {"tp": 4, "recall": 4 / 13},
    "AGREEMENT": {"tp": 4, "recall": 4 / 13, "abs_e": 8},
}

INT_CELLS = {"tp", "fp", "tn", "fn", "abs_e", "abs_n", "total",
             "both_committed", "abst_one_judge", "abst_both_judges",
             "abst_disagreement", "lone_fail_items"}
TOL = 5e-4  # published rates are quoted to 3 decimals


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# A policy maps the two reduced judge verdicts to (final, abstain_reason).
# abstain_reason is None whenever the policy commits; otherwise it names why
# the policy declined, so that abstains can be split into alignment losses
# (a judge did not commit) and genuine disagreement (both did).


def _union(g: str, c: str) -> Tuple[str, Optional[str]]:
    if "failed" in (g, c):
        return "failed", None
    if "passed" in (g, c):
        return "passed", None
    return "abstain", "both_abstain"


def _intersection(g: str, c: str) -> Tuple[str, Optional[str]]:
    if g == "failed" and c == "failed":
        return "failed", None
    if g in COMMITTED or c in COMMITTED:
        return "passed", None
    return "abstain", "both_abstain"


def _agreement(g: str, c: str) -> Tuple[str, Optional[str]]:
    g_ok, c_ok = g in COMMITTED, c in COMMITTED
    if not g_ok and not c_ok:
        return "abstain", "both_abstain"
    if not g_ok or not c_ok:
        return "abstain", "one_abstain"
    if g != c:
        return "abstain", "disagreement"
    return g, None


POLICIES: List[Tuple[str, Callable[[str, str], Tuple[str, Optional[str]]]]] = [
    ("UNION", _union),
    ("INTERSECTION", _intersection),
    ("AGREEMENT", _agreement),
]

POLICY_BLURB = {
    "UNION": "flag if either judge commits FAIL (lone FAIL + abstain = flag)",
    "INTERSECTION": "flag only if both commit FAIL (lone FAIL + abstain = pass)",
    "AGREEMENT": "commit only where both commit and agree; else abstain",
}


# ---------------------------------------------------------------------------
# Item extraction: (is_error, gptoss, claude, meta) per testbed
# ---------------------------------------------------------------------------


def _reduce(v: Optional[str]) -> str:
    """Reduce a raw verdict to failed/passed/abstain. Missing -> abstain."""
    return v if v in COMMITTED else "abstain"


def _judge_map(verdicts_by_qid: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for qid, steps in verdicts_by_qid.items():
        for s in steps:
            out[(qid, int(s["step_number"]))] = s["verdict"]
    return out


def human_items(
    labels: Dict[Tuple[str, int], Dict[str, Any]],
    gptoss_verdicts: Dict[str, List[Dict[str, Any]]],
    claude_verdicts: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    g_map = _judge_map(gptoss_verdicts)
    c_map = _judge_map(claude_verdicts)
    items = []
    for key, lab in sorted(labels.items()):
        items.append({
            "id": f"{key[0]}:step{key[1]}",
            "is_error": lab.get("human_correct") is False,
            "gptoss": _reduce(g_map.get(key)),
            "claude": _reduce(c_map.get(key)),
            "gptoss_missing": key not in g_map,
            "claude_missing": key not in c_map,
        })
    return items


def _trial_key(t: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(t["gold_id"]), int(t["step_index"]), str(t["mutation_kind"]))


def injection_items(
    gptoss_report: Dict[str, Any], gptoss_key: str,
    claude_report: Dict[str, Any], claude_key: str,
) -> List[Dict[str, Any]]:
    """Join the two single-judge injection reports trial by trial.

    The two runs are separate files over the same suite, so they are joined
    on (gold_id, step_index, mutation_kind). check_injection_join() verifies
    that the join is total and that the shared fields agree before any of
    this is used.
    """
    g_by = {_trial_key(t): t for t in gptoss_report["trials"]}
    c_by = {_trial_key(t): t for t in claude_report["trials"]}
    items = []
    for key in sorted(g_by):
        gt = g_by[key]
        if gt.get("skipped_reason"):
            continue
        ct = c_by[key]
        gv = (gt.get("verdicts") or {}).get(gptoss_key) or {}
        cv = (ct.get("verdicts") or {}).get(claude_key) or {}
        items.append({
            "id": f"gold{key[0]}:step{key[1]}:{key[2]}",
            "is_error": gt["severity"] == "error",
            "gptoss": _reduce(gv.get("verdict")),
            "claude": _reduce(cv.get("verdict")),
            "gptoss_missing": "verdict" not in gv,
            "claude_missing": "verdict" not in cv,
            "family": gt.get("family"),
            "severity": gt["severity"],
        })
    return items


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(items: List[Dict[str, Any]],
             policy: Callable[[str, str], Tuple[str, Optional[str]]]) -> Dict[str, Any]:
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "abs_e": 0, "abs_n": 0,
         "abst_one_judge": 0, "abst_both_judges": 0, "abst_disagreement": 0}
    for it in items:
        final, reason = policy(it["gptoss"], it["claude"])
        is_error = it["is_error"]
        if final == "abstain":
            c["abs_e" if is_error else "abs_n"] += 1
            c[{"one_abstain": "abst_one_judge",
               "both_abstain": "abst_both_judges",
               "disagreement": "abst_disagreement"}[reason]] += 1
        elif is_error and final == "failed":
            c["tp"] += 1
        elif is_error:
            c["fn"] += 1
        elif final == "failed":
            c["fp"] += 1
        else:
            c["tn"] += 1
    row = _rates(c["tp"], c["fp"], c["tn"], c["fn"], c["abs_e"], c["abs_n"])
    for k in ("abst_one_judge", "abst_both_judges", "abst_disagreement"):
        row[k] = c[k]
    return row


def coverage_profile(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Who committed on what, independent of any policy."""
    g = sum(1 for it in items if it["gptoss"] in COMMITTED)
    c = sum(1 for it in items if it["claude"] in COMMITTED)
    both = sum(1 for it in items
               if it["gptoss"] in COMMITTED and it["claude"] in COMMITTED)
    either = sum(1 for it in items
                 if it["gptoss"] in COMMITTED or it["claude"] in COMMITTED)
    lone_fail = sum(
        1 for it in items
        if {it["gptoss"], it["claude"]} == {"failed", "abstain"}
    )
    return {
        "n": len(items),
        "gptoss_committed": g,
        "claude_committed": c,
        "both_committed": both,
        "either_committed": either,
        "neither_committed": len(items) - either,
        "gptoss_only": g - both,
        "claude_only": c - both,
        "lone_fail_items": lone_fail,
    }


# ---------------------------------------------------------------------------
# Gates
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


def check_standalone_reproduces_table1(
    judge_alone: Dict[Tuple[str, str], Dict[str, Any]]
) -> None:
    """Hard gate. No ensemble row is trustworthy if Table 1 does not come back."""
    problems = []
    for (dataset, model), want in sorted(PUBLISHED_TABLE1.items()):
        got = judge_alone.get((dataset, model))
        if got is None:
            problems.append(f"  {dataset}/{model}: standalone row missing entirely")
            continue
        for d in _cells_differ(got, want):
            problems.append(f"  {dataset}/{model}  {d}")
    if problems:
        print("FATAL: standalone judge rows do not reproduce published Table 1.",
              file=sys.stderr)
        print("An input artifact has changed or the loader is wrong. No ensemble",
              file=sys.stderr)
        print("rows are reported. Differences:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(2)


def check_no_missing_verdicts(datasets: Dict[str, List[Dict[str, Any]]]) -> None:
    """Hard gate. Every abstain must be a verdict, not a missing record.

    If a judge simply has no row for an item, the ensemble is silently
    scoring a smaller set than it reports, and the "alignment" accounting
    below would attribute a data gap to a judge decision.
    """
    problems = []
    for dataset, items in sorted(datasets.items()):
        for field, model in (("gptoss_missing", "gpt-oss-120b"),
                             ("claude_missing", "claude-sonnet-4-5")):
            missing = [it["id"] for it in items if it[field]]
            if missing:
                problems.append(
                    f"  {dataset}/{model}: {len(missing)} item(s) with no verdict "
                    f"record, e.g. {missing[:5]}"
                )
    if problems:
        print("FATAL: some items have no verdict record from a judge.",
              file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(3)


def check_injection_join(gptoss_report, claude_report) -> None:
    """Hard gate. The two single-judge injection files must be the same suite.

    They are separate runs, joined here on (gold_id, step_index,
    mutation_kind). If the trial sets, the severities, the skip decisions,
    or the shared symbolic verdicts diverge, the join is silently pairing
    different trials and every ensemble cell below is meaningless.
    """
    g_by = {_trial_key(t): t for t in gptoss_report["trials"]}
    c_by = {_trial_key(t): t for t in claude_report["trials"]}
    problems = []
    if len(g_by) != len(gptoss_report["trials"]):
        problems.append("  gpt-oss report has duplicate trial keys")
    if len(c_by) != len(claude_report["trials"]):
        problems.append("  claude report has duplicate trial keys")
    only_g, only_c = sorted(set(g_by) - set(c_by)), sorted(set(c_by) - set(g_by))
    if only_g:
        problems.append(f"  {len(only_g)} trial(s) only in gpt-oss report, e.g. {only_g[:3]}")
    if only_c:
        problems.append(f"  {len(only_c)} trial(s) only in claude report, e.g. {only_c[:3]}")
    for key in sorted(set(g_by) & set(c_by)):
        gt, ct = g_by[key], c_by[key]
        if gt["severity"] != ct["severity"]:
            problems.append(f"  {key}: severity {gt['severity']!r} vs {ct['severity']!r}")
        if bool(gt.get("skipped_reason")) != bool(ct.get("skipped_reason")):
            problems.append(f"  {key}: skipped in one report but not the other")
        gs = ((gt.get("verdicts") or {}).get("symbolic") or {}).get("verdict")
        cs = ((ct.get("verdicts") or {}).get("symbolic") or {}).get("verdict")
        if gs != cs:
            problems.append(f"  {key}: symbolic verdict {gs!r} vs {cs!r}")
    if problems:
        print("FATAL: the two injection reports do not describe the same suite.",
              file=sys.stderr)
        print("\n".join(problems[:20]), file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
        raise SystemExit(6)


def check_human_expectations(rows: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    """Hard gate against the Analysis section's per-error counts."""
    problems = []
    for policy, want in sorted(EXPECTED_HUMAN.items()):
        got = rows.get((policy, "human_labels"))
        if got is None:
            problems.append(f"  {policy}: human row missing entirely")
            continue
        for d in _cells_differ(got, want):
            problems.append(f"  {policy}/human_labels  {d.replace('published', 'expected')}")
    if problems:
        print("FATAL: human-set ensemble rows disagree with the Analysis section.",
              file=sys.stderr)
        print("Expected from its per-error counts (both 4, claude alone 6,",
              file=sys.stderr)
        print("gpt-oss alone 2, neither 1): UNION recall 12/13, INTERSECTION",
              file=sys.stderr)
        print("4/13, AGREEMENT 4/13 with AbsE 8. Reporting the discrepancy",
              file=sys.stderr)
        print("instead of the numbers. Differences:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(4)


# ---------------------------------------------------------------------------
# Descriptive breakdowns
# ---------------------------------------------------------------------------


def catch_breakdown(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Who commits FAIL on the true-error items."""
    errs = [it for it in items if it["is_error"]]
    both = sum(1 for it in errs if it["gptoss"] == "failed" and it["claude"] == "failed")
    g_only = sum(1 for it in errs if it["gptoss"] == "failed" and it["claude"] != "failed")
    c_only = sum(1 for it in errs if it["gptoss"] != "failed" and it["claude"] == "failed")
    neither = len(errs) - both - g_only - c_only
    return {"errors": len(errs), "both": both, "gptoss_only": g_only,
            "claude_only": c_only, "neither": neither}


def agreement_abstain_split(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """AGREEMENT's abstains, split by ground truth and by reason."""
    out: Dict[str, Any] = {}
    for label, subset in (("error", [it for it in items if it["is_error"]]),
                          ("non_error", [it for it in items if not it["is_error"]])):
        reasons = {"one_abstain": 0, "both_abstain": 0, "disagreement": 0}
        for it in subset:
            final, reason = _agreement(it["gptoss"], it["claude"])
            if final == "abstain":
                reasons[reason] += 1
        total = sum(reasons.values())
        out[label] = {
            "n": len(subset),
            "abstained": total,
            "abstain_rate": (total / len(subset)) if subset else None,
            **reasons,
        }
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLUMNS = [
    ("policy", "policy", 18), ("dataset", "dataset", 16),
    ("coverage", "cov", 7), ("precision", "prec", 7), ("recall", "rec", 7),
    ("fpr", "fpr", 7), ("abstain_rate", "abst", 7),
    ("tp", "TP", 4), ("fp", "FP", 4), ("tn", "TN", 4), ("fn", "FN", 4),
    ("abs_e", "AbsE", 5), ("abs_n", "AbsN", 5), ("total", "N", 4),
    ("abst_one_judge", "a1j", 5), ("abst_both_judges", "a2j", 5),
    ("abst_disagreement", "adis", 5),
]
LEFT = ("policy", "dataset")


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _line(vals: Dict[str, Any]) -> str:
    return "  ".join(
        f"{_cell(vals.get(f)):<{w}}" if f in LEFT else f"{_cell(vals.get(f)):>{w}}"
        for f, _, w in COLUMNS
    )


def render(rows: Dict[Tuple[str, str], Dict[str, Any]],
           judge_alone: Dict[Tuple[str, str], Dict[str, Any]],
           datasets: List[str]) -> str:
    head = "  ".join(f"{lab:<{w}}" if f in LEFT else f"{lab:>{w}}"
                     for f, lab, w in COLUMNS)
    sep = "-" * len(head)
    out = [head, sep]
    for dataset in datasets:
        for model in ("gpt-oss-120b", "claude-sonnet-4-5"):
            r = judge_alone.get((dataset, model))
            if r is not None:
                out.append(_line({"policy": model, "dataset": dataset, **r}))
    out.append(sep)
    for policy, _ in POLICIES:
        for dataset in datasets:
            r = rows.get((policy, dataset))
            if r is not None:
                out.append(_line({"policy": policy, "dataset": dataset, **r}))
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
            return 5

    labels = _load_human_labels(args.labels)
    gptoss_h = _load(args.gptoss_human)
    claude_h = _load(args.claude_human)
    gptoss_i = _load(args.gptoss_injection)
    claude_i = _load(args.claude_injection)
    gptoss_key = _find_judge_key(gptoss_i["metrics"])
    claude_key = _find_judge_key(claude_i["metrics"])
    gptoss_model = gptoss_key.split(":", 1)[-1]
    claude_model = claude_key.split(":", 1)[-1]

    print("build_ensemble_policies.py -- read-only; frozen verdicts; no API calls")
    print()

    print("Gate 0: the two injection reports must describe the same suite ...", end=" ")
    check_injection_join(gptoss_i, claude_i)
    print("OK")

    items = {
        "human_labels": human_items(labels, gptoss_h["verdicts"], claude_h["verdicts"]),
        "error_injection": injection_items(gptoss_i, gptoss_key, claude_i, claude_key),
    }
    datasets = ["human_labels", "error_injection"]

    judge_alone = {
        ("human_labels", gptoss_model): _human_row_from_judge(labels, gptoss_h["verdicts"]),
        ("human_labels", claude_model): _human_row_from_judge(labels, claude_h["verdicts"]),
        ("error_injection", gptoss_model): _injection_row(gptoss_i, gptoss_key),
        ("error_injection", claude_model): _injection_row(claude_i, claude_key),
    }

    print("Gate 1: standalone judge rows must reproduce Table 1 exactly ...", end=" ")
    check_standalone_reproduces_table1(judge_alone)
    print("OK (all cells, both testbeds, both judges)")

    print("Gate 2: every item must carry a verdict from both judges ...", end=" ")
    check_no_missing_verdicts(items)
    print("OK")

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for policy, fn in POLICIES:
        for dataset in datasets:
            rows[(policy, dataset)] = evaluate(items[dataset], fn)

    print("Gate 3: human-set recalls must match the Analysis section ...", end=" ")
    check_human_expectations(rows)
    print("OK (UNION 12/13, INTERSECTION 4/13, AGREEMENT 4/13 with AbsE 8)")
    print()

    # --- who commits where -------------------------------------------------
    profiles = {d: coverage_profile(items[d]) for d in datasets}
    print("Judge commitment (independent of any policy)")
    for d in datasets:
        p = profiles[d]
        print(f"  {d}: N={p['n']}  gpt-oss commits {p['gptoss_committed']}, "
              f"claude commits {p['claude_committed']}")
        print(f"    both committed      {p['both_committed']}  "
              f"<- the set AGREEMENT can even look at")
        print(f"    exactly one         {p['gptoss_only'] + p['claude_only']}  "
              f"(gpt-oss only {p['gptoss_only']}, claude only {p['claude_only']})")
        print(f"    neither             {p['neither_committed']}")
        print(f"    lone FAIL + abstain {p['lone_fail_items']}  "
              f"(UNION flags these; INTERSECTION passes them)")
    print()

    print("Abstains by cause: alignment (a judge did not commit) vs disagreement")
    print("  a1j = exactly one judge abstained; a2j = both did; adis = both")
    print("  committed but disagreed. UNION and INTERSECTION abstain only on a2j,")
    print("  so their entire abstain budget is items neither judge would grade.")
    for policy, _ in POLICIES:
        for d in datasets:
            r = rows[(policy, d)]
            align = r["abst_one_judge"] + r["abst_both_judges"]
            total = r["abs_e"] + r["abs_n"]
            print(f"  {policy:<12} {d:<16} abstains {total:>3}  "
                  f"alignment {align:>3} (a1j {r['abst_one_judge']}, "
                  f"a2j {r['abst_both_judges']})  disagreement "
                  f"{r['abst_disagreement']:>3}")
    print()

    # --- per-error catch breakdown ----------------------------------------
    print("Which judge catches which error (committed FAIL on a true error)")
    for d in datasets:
        b = catch_breakdown(items[d])
        print(f"  {d}: {b['errors']} error items -- both {b['both']}, "
              f"gpt-oss only {b['gptoss_only']}, claude only {b['claude_only']}, "
              f"neither {b['neither']}")
    print()

    # --- AGREEMENT abstain concentration ----------------------------------
    print("AGREEMENT abstains: error items vs non-error items")
    for d in datasets:
        split = agreement_abstain_split(items[d])
        e, n = split["error"], split["non_error"]
        print(f"  {d}:")
        print(f"    error items     {e['abstained']:>3}/{e['n']:<4} "
              f"= {e['abstain_rate']:.3f}   "
              f"(one abstained {e['one_abstain']}, both {e['both_abstain']}, "
              f"disagreed {e['disagreement']})")
        print(f"    non-error items {n['abstained']:>3}/{n['n']:<4} "
              f"= {n['abstain_rate']:.3f}   "
              f"(one abstained {n['one_abstain']}, both {n['both_abstain']}, "
              f"disagreed {n['disagreement']})")
        if n["abstain_rate"]:
            ratio = e["abstain_rate"] / n["abstain_rate"]
            gloss = ("AGREEMENT declines to grade where the errors are"
                     if ratio > 1 else
                     "AGREEMENT abstains LESS often on errors here than on non-errors")
            print(f"    ratio {ratio:.2f}x -- {gloss}")
    print()

    for policy, _ in POLICIES:
        print(f"  {policy:<12} {POLICY_BLURB[policy]}")
    print()
    print(render(rows, judge_alone, datasets))

    if args.json_out is not None:
        payload = {
            "read_only": True,
            "judges": [gptoss_model, claude_model],
            "policies": {p: POLICY_BLURB[p] for p, _ in POLICIES},
            "abstain_handling": {
                "UNION": "lone FAIL + abstain = flag; abstain only if both abstain",
                "INTERSECTION": "lone FAIL + abstain = pass; abstain only if both abstain",
                "AGREEMENT": "any abstain from either judge = abstain; "
                             "both-committed disagreement = abstain",
            },
            "metric_definitions": {
                "recall": "strict: TP / (TP + FN + AbsE)",
                "fpr": "covered: FP / (FP + TN)",
                "coverage": "(TP+FP+TN+FN) / N; crash/unparseable count as abstain",
            },
            "commitment_profile": profiles,
            "catch_breakdown": {d: catch_breakdown(items[d]) for d in datasets},
            "agreement_abstain_split": {
                d: agreement_abstain_split(items[d]) for d in datasets
            },
            "judge_alone": [
                {"dataset": d, "grader": g, **r} for (d, g), r in sorted(judge_alone.items())
            ],
            "rows": [
                {"policy": p, "dataset": d, **r} for (p, d), r in sorted(rows.items())
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
