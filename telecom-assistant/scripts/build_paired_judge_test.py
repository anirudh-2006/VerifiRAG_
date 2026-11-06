#!/usr/bin/env python3
"""Paired significance test for judge false positives on the injection suite.

Frozen verdicts only. This script makes no API calls and reads nothing but the
two committed injection reports.

The question: claude-sonnet-4-5 commits 6 false positives across its 21 covered
negatives while gpt-oss-120b commits 0 across 24. Is that separable from noise?

THE PAIRED TEST IS THE CORRECT ONE HERE, and the unpaired one is reported only
so the difference is visible. Both judges graded the SAME 24 negative trials
from the same fixtures with the same prompt. An unpaired 2x2 (Fisher) treats
the two judge rows as independent samples drawn from separate populations,
which they are not; it discards the pairing and reports a smaller p-value than
the evidence supports. McNemar conditions on the items, uses only the trials
where the two judges disagree, and is the standard test for two raters on one
item set. We report Fisher alongside it as a sensitivity figure, never as the
headline.

Coding, on negatives only (severity != "error"):
    1 = false positive (judge committed FAIL on a negative)
    0 = correct rejection (judge committed PASS on a negative)
    abstain / unparseable / crash -> trial leaves the paired set entirely,
    consistent with the covered-FPR convention in paper-judge/sections/
    protocol.tex ("Denominator conventions").

Output: results/paired_judge_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import mcnemar_exact_test  # noqa: E402

TrialKey = Tuple[int, int, str]

CLAUDE_KEY = "llm_judge:claude-sonnet-4-5"
GPTOSS_KEY = "llm_judge:gpt-oss-120b"

# Root CLAUDE.md, "Table 1 -- authoritative, do not re-derive from any PDF",
# error-injection suite, as revised 2026-08-24. The gate below refuses to
# compute anything unless the inputs reproduce these exactly. Do not relax a
# cell to make a new input pass; a mismatch means the reports moved and the
# published rows need re-deriving first.
PUBLISHED_INJECTION: Dict[str, Dict[str, int]] = {
    CLAUDE_KEY: {
        "tp": 90, "fp": 6, "tn": 15, "fn": 2,
        "abstain_error": 2, "abstain_eq": 3,
    },
    GPTOSS_KEY: {
        "tp": 84, "fp": 0, "tn": 24, "fn": 10,
        "abstain_error": 0, "abstain_eq": 0,
    },
}
PUBLISHED_GRADED = {"total": 118, "error": 94, "non_error": 24}


class GateFailure(RuntimeError):
    """Input counts do not reproduce the published injection rows."""


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise GateFailure(f"missing input report: {path}")
    return json.loads(path.read_text())


def _judge_key(report: Dict[str, Any], path: Path) -> str:
    keys = [k for k in report.get("metrics", {}) if k.startswith("llm_judge:")]
    if len(keys) != 1:
        raise GateFailure(
            f"{path}: expected exactly one llm_judge metrics block, found {keys}"
        )
    return keys[0]


def _recount(report: Dict[str, Any], key: str) -> Dict[str, int]:
    """Recount the confusion cells from per-trial verdicts.

    Deliberately does not read the report's own metrics block, so the gate can
    compare the two against each other.
    """
    cells = {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
             "abstain_error": 0, "abstain_eq": 0}
    graded = {"total": 0, "error": 0, "non_error": 0}
    for trial in report["trials"]:
        verdict_block = trial.get("verdicts", {}).get(key)
        if not verdict_block:
            continue  # skipped trial: no verdict from any grader
        verdict = verdict_block.get("verdict")
        is_error = trial.get("severity") == "error"
        graded["total"] += 1
        graded["error" if is_error else "non_error"] += 1
        if is_error:
            if verdict == "failed":
                cells["tp"] += 1
            elif verdict == "passed":
                cells["fn"] += 1
            else:
                cells["abstain_error"] += 1
        else:
            if verdict == "failed":
                cells["fp"] += 1
            elif verdict == "passed":
                cells["tn"] += 1
            else:
                cells["abstain_eq"] += 1
    return {**cells, "graded": graded}


def _gate(reports: Dict[str, Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
    """Refuse to proceed unless both inputs reproduce the published rows.

    Three checks per judge, and all three must pass:
      1. per-trial recount == the report's own embedded metrics block
      2. per-trial recount == the published Table 1 injection row
      3. graded trial counts == 118 total / 94 error / 24 non-error
    """
    failures: List[str] = []
    detail: Dict[str, Any] = {}

    for key, (path, report) in reports.items():
        counted = _recount(report, key)
        graded = counted.pop("graded")
        embedded = report["metrics"][key]["overall"]
        published = PUBLISHED_INJECTION[key]

        for cell, got in sorted(counted.items()):
            want = embedded.get(cell)
            if want != got:
                failures.append(
                    f"{key}: per-trial recount {cell}={got} disagrees with the "
                    f"report's own metrics block ({cell}={want}) in {path.name}"
                )
        for cell, want in sorted(published.items()):
            got = counted[cell]
            if got != want:
                failures.append(
                    f"{key}: {cell}={got} does not reproduce the published "
                    f"injection row ({cell}={want}); see root CLAUDE.md Table 1"
                )
        for field, want in sorted(PUBLISHED_GRADED.items()):
            got = graded[field]
            if got != want:
                failures.append(
                    f"{key}: graded {field}={got}, published {want}"
                )
        detail[key] = {"source": str(path), "recounted": counted, "graded": graded}

    if failures:
        raise GateFailure(
            "input counts do not reproduce the published injection rows:\n  - "
            + "\n  - ".join(failures)
        )
    return detail


def _negatives(report: Dict[str, Any], key: str) -> Dict[TrialKey, str]:
    """Every graded negative trial and this judge's verdict on it."""
    out: Dict[TrialKey, str] = {}
    for trial in report["trials"]:
        if trial.get("severity") == "error":
            continue
        block = trial.get("verdicts", {}).get(key)
        if not block:
            continue
        tkey: TrialKey = (
            int(trial["gold_id"]), int(trial["step_index"]), str(trial["mutation_kind"])
        )
        out[tkey] = block.get("verdict")
    return out


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]], by total probability.

    Sums the hypergeometric probability of every table at least as extreme as
    the observed one. Exact; no normal approximation.
    """
    n = a + b + c + d
    row1, row2, col1 = a + b, c + d, a + c
    total = comb(n, col1)

    def prob(k: int) -> float:
        return comb(row1, k) * comb(row2, col1 - k) / total

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1)
                        if prob(k) <= observed + 1e-12))


def _provenance(reports: Dict[str, Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "generated_by": "scripts/build_paired_judge_test.py",
        "inputs_are_frozen_verdicts": True,
        "api_calls_made": 0,
        "source_reports": {},
    }
    for key, (path, report) in reports.items():
        prov = report.get("provenance", {})
        block["source_reports"][key] = {
            "path": str(path),
            "last_modified": prov.get("last_modified"),
            "revisions": prov.get("revisions", []),
            "two_vintage_warning": prov.get("two_vintage_warning"),
        }
    return block


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude", type=Path,
                    default=Path("results/error_injection_report_claude.json"))
    ap.add_argument("--gptoss", type=Path,
                    default=Path("results/error_injection_report_with_judge.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/paired_judge_test.json"))
    args = ap.parse_args(argv)

    claude_report = _load(args.claude)
    gptoss_report = _load(args.gptoss)

    ck = _judge_key(claude_report, args.claude)
    gk = _judge_key(gptoss_report, args.gptoss)
    if (ck, gk) != (CLAUDE_KEY, GPTOSS_KEY):
        raise GateFailure(
            f"unexpected judge keys: got {ck!r} and {gk!r}, "
            f"expected {CLAUDE_KEY!r} and {GPTOSS_KEY!r}"
        )

    reports = {ck: (args.claude, claude_report), gk: (args.gptoss, gptoss_report)}
    gate_detail = _gate(reports)
    print("GATE PASSED: both inputs reproduce the published injection rows.")

    claude_neg = _negatives(claude_report, ck)
    gptoss_neg = _negatives(gptoss_report, gk)
    if set(claude_neg) != set(gptoss_neg):
        raise GateFailure(
            "the two judges were not shown the same negative trials; "
            f"claude-only {sorted(set(claude_neg) - set(gptoss_neg))}, "
            f"gpt-oss-only {sorted(set(gptoss_neg) - set(claude_neg))}"
        )

    committed = {"passed", "failed"}
    paired_keys = sorted(k for k in claude_neg
                         if claude_neg[k] in committed and gptoss_neg[k] in committed)

    claude_fp = [1.0 if claude_neg[k] == "failed" else 0.0 for k in paired_keys]
    gptoss_fp = [1.0 if gptoss_neg[k] == "failed" else 0.0 for k in paired_keys]
    mcnemar = mcnemar_exact_test(claude_fp, gptoss_fp)

    c_fp = sum(1 for v in claude_neg.values() if v == "failed")
    c_tn = sum(1 for v in claude_neg.values() if v == "passed")
    g_fp = sum(1 for v in gptoss_neg.values() if v == "failed")
    g_tn = sum(1 for v in gptoss_neg.values() if v == "passed")
    fisher_p = _fisher_exact_two_sided(c_fp, c_tn, g_fp, g_tn)

    discordant = [
        {
            "gold_id": k[0], "step_index": k[1], "mutation_kind": k[2],
            "family": next(t["family"] for t in claude_report["trials"]
                           if (t["gold_id"], t["step_index"], t["mutation_kind"]) == k),
            "claude": claude_neg[k], "gpt-oss": gptoss_neg[k],
        }
        for k in paired_keys
        if (claude_neg[k] == "failed") != (gptoss_neg[k] == "failed")
    ]

    payload: Dict[str, Any] = {
        "question": (
            "Is claude-sonnet-4-5's false-positive rate on injection negatives "
            "separable from gpt-oss-120b's?"
        ),
        "headline": {
            "test": "exact McNemar, two-sided",
            "p_value": mcnemar["p_value_two_sided"],
            "n_paired": mcnemar["n_pairs"],
        },
        "why_paired_is_correct": (
            "Both judges graded the same 24 negative trials from the same "
            "fixtures with the same prompt, so the two judge rows are repeated "
            "measures on one item set rather than independent samples. McNemar "
            "conditions on the items and uses only the discordant trials, which "
            "is the standard test for two raters on a shared item set. The "
            "unpaired Fisher figure below discards that pairing, treats the "
            "rows as independent, and therefore reports a smaller p-value than "
            "the design supports. Fisher is a sensitivity figure here, not the "
            "headline."
        ),
        "coding": (
            "negatives only (severity != 'error'); 1 = committed FAIL (false "
            "positive), 0 = committed PASS (correct rejection); abstain and "
            "unparseable leave the set, per the covered-FPR convention"
        ),
        "paired_mcnemar": {
            **mcnemar,
            "a_is": "claude-sonnet-4-5 false positive",
            "b_is": "gpt-oss-120b false positive",
            "discordant_trials": discordant,
        },
        "unpaired_fisher": {
            "test": "Fisher exact, two-sided",
            "p_value": fisher_p,
            "table": {
                "claude-sonnet-4-5": {"fp": c_fp, "tn": c_tn,
                                      "covered_negatives": c_fp + c_tn},
                "gpt-oss-120b": {"fp": g_fp, "tn": g_tn,
                                 "covered_negatives": g_fp + g_tn},
            },
            "caveat": "ignores that both judges graded the same items",
        },
        "counts": {
            "negatives_presented_to_each_judge": len(claude_neg),
            "claude-sonnet-4-5": {
                "covered_negatives": c_fp + c_tn, "fp": c_fp, "tn": c_tn,
                "abstained_on_negatives": len(claude_neg) - (c_fp + c_tn),
            },
            "gpt-oss-120b": {
                "covered_negatives": g_fp + g_tn, "fp": g_fp, "tn": g_tn,
                "abstained_on_negatives": len(gptoss_neg) - (g_fp + g_tn),
            },
            "negatives_both_judges_committed_on": len(paired_keys),
        },
        "gate": {
            "published_source": "root CLAUDE.md, Table 1, error-injection suite",
            "checks": [
                "per-trial recount == each report's embedded metrics block",
                "per-trial recount == published injection row",
                "graded trials == 118 total / 94 error / 24 non-error",
            ],
            "passed": True,
            "detail": gate_detail,
        },
        "provenance": _provenance(reports),
    }

    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nnegatives presented to each judge: {len(claude_neg)}")
    print(f"  claude-sonnet-4-5  FP {c_fp} / {c_fp + c_tn} covered "
          f"({len(claude_neg) - (c_fp + c_tn)} abstained)")
    print(f"  gpt-oss-120b       FP {g_fp} / {g_fp + g_tn} covered "
          f"({len(gptoss_neg) - (g_fp + g_tn)} abstained)")
    print(f"\nnegatives both judges committed on: {len(paired_keys)}")
    print(f"  discordant: b (claude FP only) = {mcnemar['b_a1_b0']}, "
          f"c (gpt-oss FP only) = {mcnemar['c_a0_b1']}")
    print(f"  exact McNemar two-sided p = {mcnemar['p_value_two_sided']:.6f}"
          f"   <- the headline")
    print(f"  unpaired Fisher two-sided p = {fisher_p:.6f}"
          f"   <- sensitivity only; discards the pairing")
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as exc:
        print(f"\nGATE FAILED\n{exc}\n", file=sys.stderr)
        print("Nothing was computed and nothing was written.", file=sys.stderr)
        raise SystemExit(2)
