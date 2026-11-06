#!/usr/bin/env python3
"""
Unit tests for src.error_injection.

Covers:
  * every mutation callable — shape of output, skip semantics, severity
    labelling consistency
  * backward-compatible MUTATORS dict exposure
  * grader wrappers (SymbolicGrader on a synthetic gold, LLMJudgeGrader
    with FakeJudge)
  * fixture viability filter
  * metric computation correctness against a hand-built trial set
  * CLI main() smoke (runs full suite with symbolic grader, writes JSON)

Run:
    ./.venv/bin/python scripts/test_error_injection.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.error_injection import (  # noqa: E402
    GOLD_FIXTURES,
    MUTATION_SPECS,
    MUTATORS,
    GraderVerdict,
    LLMJudgeGrader,
    MutationSpec,
    SymbolicGrader,
    Trial,
    compute_metrics,
    filter_viable_fixtures,
    inject_and_grade,
    inject_and_verify,
    main as ei_main,
    run_suite,
)
from src.llm_judge import FakeJudge  # noqa: E402


def assert_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_in(item, container, label: str) -> None:
    if item not in container:
        raise AssertionError(f"{label}: {item!r} not in {container!r}")


def assert_true(cond, label: str) -> None:
    if not cond:
        raise AssertionError(f"{label}: condition not true")


# ---------------------------------------------------------------------------
# Mutation-library tests
# ---------------------------------------------------------------------------


def test_mutation_specs_have_expected_families_and_severities() -> None:
    families = {s.family for s in MUTATION_SPECS}
    severities = {s.severity for s in MUTATION_SPECS}
    assert_true(
        {"algebraic", "bounds", "identifier", "equivalent", "control"}.issubset(families),
        "all five families present",
    )
    assert_true(
        {"error", "equivalent", "control"} == severities,
        f"exactly 3 severity buckets, got {severities}",
    )
    # Every mutation kind is unique.
    kinds = [s.kind for s in MUTATION_SPECS]
    assert_eq(len(kinds), len(set(kinds)), "mutation kinds unique")


def test_error_mutations_change_expression_or_skip_cleanly() -> None:
    base = "integrate(exp(-a*t)*exp(-I*omega*t), (t, 0, oo))"
    for spec in MUTATION_SPECS:
        if spec.severity != "error":
            continue
        out = spec.apply(base)
        # None is a legitimate skip (mutation not applicable on this expr);
        # otherwise the expression must actually change.
        if out is None:
            continue
        assert_true(
            out != base,
            f"error mutation {spec.kind} returned unchanged expression",
        )


def test_bounds_mutations_skip_on_expressions_without_limits() -> None:
    # An expression with no integrate()/Sum() — the bounds mutators must skip.
    s = "(omega + 1)**2"
    specs = [m for m in MUTATION_SPECS if m.family == "bounds"]
    assert_true(len(specs) >= 2, "at least two bounds mutations registered")
    for spec in specs:
        out = spec.apply(s)
        assert_eq(out, None, f"{spec.kind} must skip when no integral present")


def test_swap_integral_limits_actually_swaps() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "swap_integral_limits")
    out = spec.apply("integrate(exp(-t), (t, 0, oo))")
    assert_in("(t, oo, 0)", out, "swap produces reversed limits")


def test_drop_upper_limit_changes_upper_bound() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "drop_upper_limit")
    out = spec.apply("integrate(exp(-t), (t, 0, oo))")
    assert_in("(t, 0, 1)", out, "drop_upper truncates oo -> 1")


def test_drop_term_skips_single_term() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "drop_term")
    assert_eq(spec.apply("omega"), None, "drop_term has nothing to drop on 'omega'")
    out = spec.apply("omega**2 + 2*omega + 1")
    assert out != "omega**2 + 2*omega + 1", "drop_term actually drops"
    assert_true(out.count("+") < 2, "one + removed")


def test_exponent_bump_skips_when_no_integer_power() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "exponent_bump")
    assert_eq(spec.apply("a + b"), None, "no ** in expression")
    out = spec.apply("x**2")
    assert_eq(out, "x**3", "power bumped 2 -> 3")


def test_missing_factor_of_2_strips_or_divides() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "missing_factor_of_2")
    assert_eq(spec.apply("2 * sin(omega)"), "sin(omega)", "leading 2* stripped")
    assert_eq(spec.apply("sin(omega)"), "(sin(omega))/2", "falls back to /2")


def test_rename_variable_preserves_structure_but_changes_name() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "rename_variable")
    out = spec.apply("2*sin(omega)/omega")
    assert_true(out is not None, "renamed something")
    assert_true("omega" not in out, "omega no longer appears")
    assert_in("Omega", out, "replaced with Omega")


def test_rename_variable_skips_bound_dummy_variables() -> None:
    # `t` is the integration dummy here — renaming it uniformly would be a
    # no-op on the expression's value, so we must NOT treat it as an error
    # injection.  Also no free `omega` / `s` / etc. — so the mutator should
    # skip entirely and return None.
    spec = next(s for s in MUTATION_SPECS if s.kind == "rename_variable")
    # `t` is bound; `mu`, `nu` aren't in the rename priority list → skip.
    expr = "integrate(exp(-mu*t)*exp(-nu*t), (t, 0, oo))"
    assert_eq(spec.apply(expr), None, "skip when only bound dummies available")

    # `s` IS free (Laplace variable), so should get renamed while `t` is left
    # alone (it's the bound dummy).
    expr2 = "integrate(exp(-(a + s)*t), (t, 0, oo))"
    out = spec.apply(expr2)
    assert_true(out is not None, "picks free variable when present")
    assert_true("(t, 0, oo)" in out, "bound variable t untouched")
    assert_true("s" not in re.sub(r"\(t,.*?\)", "", out),
                f"free s renamed, got: {out!r}")


def test_equivalent_mutations_actually_rewrite() -> None:
    # These should change the string but not the algebraic meaning.
    assert_eq(
        next(s for s in MUTATION_SPECS if s.kind == "expand_power").apply("x**2"),
        "(x)*(x)",
        "expand_power: x**2 -> (x)*(x)",
    )
    out_dist = next(s for s in MUTATION_SPECS if s.kind == "distribute").apply("2*(a + b)")
    assert_in("2*a", out_dist, "distribute has k*a")
    assert_in("2*b", out_dist, "distribute has k*b")


def test_commute_operands_skips_without_top_level_operator() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "commute_operands")
    # Single function call has no top-level + or *; everything is inside parens.
    assert_eq(spec.apply("sin(a + b)"), None, "no top-level + or *")
    out = spec.apply("a + b")
    assert_eq(out, "b + a", "swaps operands")


def test_noop_returns_same_string() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "no_op")
    assert_eq(spec.apply("anything"), "anything", "control is identity")


def test_backcompat_mutators_dict_is_callable() -> None:
    # Legacy callers expected MUTATORS[kind](expr) to return a string.
    for kind, fn in MUTATORS.items():
        out = fn("(omega + 1)**2")
        assert_true(isinstance(out, str), f"legacy MUTATORS[{kind}] returns string")


# ---------------------------------------------------------------------------
# Grader wrapper tests
# ---------------------------------------------------------------------------


def _trivial_gold() -> dict:
    return {
        "concept": "trivial",
        "final_result": "x + 1",
        "steps": [
            {"step_number": 1, "description": "start", "expression": "x", "operation": "definition"},
            {"step_number": 2, "description": "add 1", "expression": "x + 1", "operation": "add"},
        ],
    }


def test_symbolic_grader_returns_failed_for_bad_step() -> None:
    g = SymbolicGrader()
    bad = _trivial_gold()
    bad["steps"][1]["expression"] = "x - 5"  # not consistent with step 1
    v = g.grade_step(bad, 1)
    assert_in(v.verdict, {"failed", "abstain"}, "symbolic flags or abstains on inconsistency")


def test_symbolic_grader_returns_passed_on_known_equivalence() -> None:
    g = SymbolicGrader()
    # Use an existing fixture — these are all verified at baseline.
    v = g.grade_step(GOLD_FIXTURES[2], 1)  # (omega+1)**2 -> omega**2+2omega+1
    assert_eq(v.verdict, "passed", "pre-verified fixture step grades passed")


def test_llm_judge_grader_maps_judge_step_verdict() -> None:
    fake = FakeJudge(
        default='{"verdict":"failed","confidence":0.9,"reasoning":"sign flipped"}'
    )
    g = LLMJudgeGrader(judge=fake)
    v = g.grade_step(_trivial_gold(), 1)
    assert_eq(v.verdict, "failed", "judge verdict propagates")
    assert_true(g.name.startswith("llm_judge"), "name includes backend tag")
    assert_true("fake-judge" in g.name, "name mentions fake model")


def test_llm_judge_grader_maps_unparseable_to_abstain() -> None:
    # When the judge emits garbage, llm_judge returns unparseable; for this
    # study we treat that the same as abstain because the judge didn't
    # commit to a verdict.
    fake = FakeJudge(default="not JSON at all")
    g = LLMJudgeGrader(judge=fake)
    v = g.grade_step(_trivial_gold(), 1)
    assert_eq(v.verdict, "abstain", "unparseable -> abstain for metric purposes")


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------


def test_inject_and_grade_records_skipped_when_mutation_not_applicable() -> None:
    spec = next(s for s in MUTATION_SPECS if s.kind == "swap_integral_limits")
    gold = _trivial_gold()  # no integrals → mutation should skip
    trial = inject_and_grade(gold, 1, spec, [SymbolicGrader()])
    assert_eq(trial.skipped_reason, "mutation_not_applicable", "skipped when N/A")
    assert_eq(trial.mutated_expr, None, "no mutated expression recorded")
    assert_eq(trial.verdicts, {}, "no graders run on skipped trials")


def test_inject_and_grade_records_verdicts_from_every_grader() -> None:
    fake = FakeJudge(default='{"verdict":"failed","confidence":0.8,"reasoning":""}')
    graders = [SymbolicGrader(), LLMJudgeGrader(judge=fake)]
    spec = next(s for s in MUTATION_SPECS if s.kind == "sign_flip")
    gold = GOLD_FIXTURES[2]  # quadratic identity, reliable
    trial = inject_and_grade(gold, 1, spec, graders)
    assert_eq(trial.skipped_reason, None, "mutation applied")
    assert_in("symbolic", trial.verdicts, "symbolic verdict present")
    assert_true(
        any(k.startswith("llm_judge") for k in trial.verdicts),
        "llm_judge verdict present",
    )


def test_filter_viable_fixtures_drops_broken_ones() -> None:
    broken = [
        GOLD_FIXTURES[2],  # quadratic identity, known-good and fast
        {
            "concept": "broken-fixture",
            "final_result": "?",
            "steps": [
                # 1 → 2 is one of the few pairs the verifier flags as
                # definitively `False` (pure numeric mismatch); most variable
                # expressions cause it to abstain instead.
                {"step_number": 1, "description": "start",
                 "expression": "1", "operation": "definition"},
                {"step_number": 2, "description": "bogus",
                 "expression": "2", "operation": "algebra"},
            ],
        },
    ]
    viable, warnings = filter_viable_fixtures(broken)
    assert_eq(len(viable), 1, "one fixture dropped")
    assert_true(
        any("broken-fixture" in w for w in warnings),
        f"broken fixture warned; warnings were: {warnings}",
    )


def test_run_suite_produces_trials_for_every_mutation_per_step() -> None:
    tiny = [GOLD_FIXTURES[2]]  # 2-step fixture, one mutable step
    trials = run_suite(golds=tiny, graders=[SymbolicGrader()])
    # 1 step × len(MUTATION_SPECS) trials.
    assert_eq(len(trials), len(MUTATION_SPECS),
              "one trial per mutation on the mutable step")


# ---------------------------------------------------------------------------
# Metric-computation tests
# ---------------------------------------------------------------------------


def _trial(severity: str, family: str, verdict: str) -> Trial:
    return Trial(
        gold_id=0,
        concept="x",
        step_index=1,
        mutation_kind="k",
        family=family,
        severity=severity,
        original_expr="a",
        mutated_expr="b" if severity != "control" else "a",
        verdicts={"g": {"verdict": verdict, "confidence": 1.0, "reasoning": ""}},
    )


def test_compute_metrics_recall_and_precision_on_clean_sets() -> None:
    # 4 error trials: 3 caught (failed), 1 missed (passed).
    # 2 equivalent trials: 1 FP (failed), 1 TN (passed).
    trials = [
        _trial("error", "algebraic", "failed"),
        _trial("error", "algebraic", "failed"),
        _trial("error", "algebraic", "failed"),
        _trial("error", "algebraic", "passed"),
        _trial("equivalent", "equivalent", "failed"),
        _trial("equivalent", "equivalent", "passed"),
    ]
    m = compute_metrics(trials, ["g"])["g"]["overall"]
    assert_eq(m["tp"], 3, "three true positives")
    assert_eq(m["fn"], 1, "one false negative")
    assert_eq(m["fp"], 1, "one false positive")
    assert_eq(m["tn"], 1, "one true negative")
    assert_eq(round(m["recall_strict"], 3), 0.75, "recall = 3/4")
    assert_eq(round(m["precision"], 3), 0.75, "precision = 3/4")
    assert_eq(round(m["fpr"], 3), 0.5, "fpr = 1/2")


def test_compute_metrics_abstains_not_counted_as_positive() -> None:
    trials = [
        _trial("error", "algebraic", "abstain"),
        _trial("error", "algebraic", "abstain"),
        _trial("equivalent", "equivalent", "abstain"),
    ]
    m = compute_metrics(trials, ["g"])["g"]["overall"]
    assert_eq(m["tp"], 0, "no TPs when only abstains")
    assert_eq(m["abstain_error"], 2, "error-side abstains tracked")
    assert_eq(m["abstain_eq"], 1, "equivalent-side abstains tracked")
    assert_eq(round(m["recall_strict"], 3), 0.0, "strict recall = 0")
    assert_eq(m["recall_committed"], None, "no committed predictions → None")


def test_compute_metrics_skipped_trials_excluded() -> None:
    t = Trial(
        gold_id=0, concept="", step_index=1, mutation_kind="k",
        family="algebraic", severity="error", original_expr="a",
        mutated_expr=None, verdicts={}, skipped_reason="mutation_not_applicable",
    )
    m = compute_metrics([t], ["g"])["g"]["overall"]
    assert_eq(m["total"], 0, "skipped trials don't inflate totals")


def test_compute_metrics_by_family_split() -> None:
    trials = [
        _trial("error", "algebraic", "failed"),
        _trial("error", "bounds", "passed"),
        _trial("equivalent", "equivalent", "passed"),
    ]
    m = compute_metrics(trials, ["g"])["g"]["by_family"]
    assert_eq(m["algebraic"]["tp"], 1, "algebraic has 1 TP")
    assert_eq(m["bounds"]["fn"], 1, "bounds has 1 FN")
    assert_eq(m["equivalent"]["tn"], 1, "equivalent has 1 TN")


def test_compute_metrics_crash_verdict_counts_as_abstain() -> None:
    # A verdict of "error" (judge crash / unparseable response) on a
    # true-error trial must count toward AbsE, not vanish from the
    # denominator entirely -- otherwise strict recall is inflated. This
    # must match build_threeway_table.py's treatment of the same verdict.
    trials = [
        _trial("error", "algebraic", "failed"),
        _trial("error", "algebraic", "error"),
        _trial("equivalent", "equivalent", "error"),
    ]
    m = compute_metrics(trials, ["g"])["g"]
    overall = m["overall"]
    assert_eq(overall["abstain_error"], 1, "crash on error trial counts as AbsE")
    assert_eq(overall["abstain_eq"], 1, "crash on equivalent trial counts as AbsN")
    assert_eq(round(overall["recall_strict"], 3), 0.5, "crash inflates the denominator, not just the numerator")
    fam = m["by_family"]["algebraic"]
    assert_eq(fam["abstain_error"], 1, "per-family AbsE also counts crashes")
    assert_eq(round(fam["recall_strict"], 3), 0.5, "per-family strict recall reflects the crash-as-abstain")


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------


def test_legacy_inject_and_verify_returns_expected_keys() -> None:
    out = inject_and_verify(GOLD_FIXTURES[2], 1, "sign_flip")
    for k in (
        "concept", "step_index", "mutation", "original_expr",
        "mutated_expr", "flagged_at_step", "flagged_anywhere",
        "verifier_message", "abstained_at_step",
    ):
        assert_in(k, out, f"legacy key {k!r} present")


def test_legacy_inject_and_verify_unknown_kind_raises() -> None:
    raised = False
    try:
        inject_and_verify(GOLD_FIXTURES[2], 1, "no_such_kind")
    except KeyError:
        raised = True
    assert_true(raised, "unknown kind raises KeyError")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_main_writes_report() -> None:
    # Keep this test cheap: monkey-patch GOLD_FIXTURES to just the two simplest
    # algebraic fixtures.  The full 9-fixture suite hits expensive SymPy
    # integrations under strict_mode and belongs in a separate perf run.
    import src.error_injection as ei
    original = ei.GOLD_FIXTURES
    ei.GOLD_FIXTURES = [ei.GOLD_FIXTURES[2], ei.GOLD_FIXTURES[3]]  # quadratic + trig
    try:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "report.json"
            rc = ei_main(["--output", str(out)])
            assert_eq(rc, 0, "main exits zero")
            assert_true(out.exists(), "report file written")
            data = json.loads(out.read_text())
            assert_in("metrics", data, "metrics key present in report")
            assert_in("trials", data, "trials key present in report")
            assert_in("config", data, "config key present in report")
            assert_in("symbolic", data["metrics"], "symbolic grader graded")
            # Sanity: at least some trials passed / failed / weren't skipped.
            assert_true(
                data["metrics"]["symbolic"]["overall"]["total"] > 0,
                "some trials evaluated",
            )
    finally:
        ei.GOLD_FIXTURES = original


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("specs: families + severities", test_mutation_specs_have_expected_families_and_severities),
    ("error mutations change or skip", test_error_mutations_change_expression_or_skip_cleanly),
    ("bounds mutations skip cleanly", test_bounds_mutations_skip_on_expressions_without_limits),
    ("swap_integral_limits works", test_swap_integral_limits_actually_swaps),
    ("drop_upper_limit works", test_drop_upper_limit_changes_upper_bound),
    ("drop_term skips single term", test_drop_term_skips_single_term),
    ("exponent_bump skips no-power", test_exponent_bump_skips_when_no_integer_power),
    ("missing_factor_of_2 behaviours", test_missing_factor_of_2_strips_or_divides),
    ("rename_variable preserves structure", test_rename_variable_preserves_structure_but_changes_name),
    ("rename_variable skips bound vars", test_rename_variable_skips_bound_dummy_variables),
    ("equivalent mutations rewrite", test_equivalent_mutations_actually_rewrite),
    ("commute_operands skips top-paren", test_commute_operands_skips_without_top_level_operator),
    ("noop is identity", test_noop_returns_same_string),
    ("legacy MUTATORS dict usable", test_backcompat_mutators_dict_is_callable),
    ("symbolic grader failed on bad", test_symbolic_grader_returns_failed_for_bad_step),
    ("symbolic grader passed on good", test_symbolic_grader_returns_passed_on_known_equivalence),
    ("llm_judge grader propagates verdict", test_llm_judge_grader_maps_judge_step_verdict),
    ("llm_judge unparseable → abstain", test_llm_judge_grader_maps_unparseable_to_abstain),
    ("inject_and_grade skip record", test_inject_and_grade_records_skipped_when_mutation_not_applicable),
    ("inject_and_grade multi-grader", test_inject_and_grade_records_verdicts_from_every_grader),
    ("filter_viable_fixtures drops broken", test_filter_viable_fixtures_drops_broken_ones),
    ("run_suite produces all trials", test_run_suite_produces_trials_for_every_mutation_per_step),
    ("metrics: precision/recall/fpr", test_compute_metrics_recall_and_precision_on_clean_sets),
    ("metrics: abstains tracked", test_compute_metrics_abstains_not_counted_as_positive),
    ("metrics: skipped excluded", test_compute_metrics_skipped_trials_excluded),
    ("metrics: by-family split", test_compute_metrics_by_family_split),
    ("metrics: crash verdict counts as abstain", test_compute_metrics_crash_verdict_counts_as_abstain),
    ("legacy inject_and_verify keys", test_legacy_inject_and_verify_returns_expected_keys),
    ("legacy inject_and_verify unknown", test_legacy_inject_and_verify_unknown_kind_raises),
    ("main() writes report", test_main_writes_report),
]


def main() -> int:
    passed = failed = 0
    print(f"Running {len(TESTS)} error_injection tests...\n")
    for name, fn in TESTS:
        try:
            fn()
            print(f"  OK  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    total = passed + failed
    print(f"\nResult: {passed}/{total} passed, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
