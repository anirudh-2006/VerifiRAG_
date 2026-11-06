#!/usr/bin/env python3
"""
Unit tests for src.llm_judge using FakeJudge.

These tests exercise every branch of the judge pipeline — prompt rendering,
response parsing (direct JSON / fenced / regex fallback / garbage),
verdict validation, confidence clamping — without touching the network.

Run:
    ./.venv/bin/python scripts/test_llm_judge.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.llm_judge import (  # noqa: E402
    FakeJudge,
    _extract_json_object,
    _parse_verdict,
    _render_expected_for_prompt,
    _safe_float,
    _VALID_FINAL_VERDICTS,
    _VALID_STEP_VERDICTS,
    judge_final_answer,
    judge_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_in(item, container, label: str) -> None:
    if item not in container:
        raise AssertionError(f"{label}: expected {item!r} in {container!r}")


def assert_close(a: float, b: float, tol: float, label: str) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{label}: |{a} - {b}| > {tol}")


# ---------------------------------------------------------------------------
# Tests — pure helpers
# ---------------------------------------------------------------------------


def test_safe_float_clamps_and_handles_garbage() -> None:
    assert_eq(_safe_float(0.5), 0.5, "safe_float mid")
    assert_eq(_safe_float(-0.2), 0.0, "safe_float negative")
    assert_eq(_safe_float(1.7), 1.0, "safe_float >1")
    assert_eq(_safe_float("0.9"), 0.9, "safe_float string number")
    assert_eq(_safe_float("abc"), 0.0, "safe_float garbage")
    assert_eq(_safe_float(None), 0.0, "safe_float None")
    assert_eq(_safe_float(float("nan")), 0.0, "safe_float NaN")


def test_extract_json_direct() -> None:
    obj = _extract_json_object('{"verdict": "match", "confidence": 0.9}')
    assert obj == {"verdict": "match", "confidence": 0.9}


def test_extract_json_fenced() -> None:
    raw = 'Sure!\n```json\n{"verdict": "mismatch", "confidence": 0.8}\n```\nThanks.'
    obj = _extract_json_object(raw)
    assert obj == {"verdict": "mismatch", "confidence": 0.8}


def test_extract_json_regex_fallback() -> None:
    raw = 'Here is my verdict: {"verdict":"abstain","confidence":0.5} hope that helps'
    obj = _extract_json_object(raw)
    assert obj == {"verdict": "abstain", "confidence": 0.5}


def test_extract_json_garbage_returns_none() -> None:
    assert _extract_json_object("hello world") is None
    assert _extract_json_object("") is None
    assert _extract_json_object("   ") is None


def test_parse_verdict_unknown_verdict_marked_unparseable() -> None:
    raw = '{"verdict":"super_correct","confidence":1.0,"reasoning":"eh"}'
    out = _parse_verdict(raw, _VALID_FINAL_VERDICTS)
    assert_eq(out["verdict"], "unparseable", "unknown verdict → unparseable")
    assert_eq(out["judge_raw"], raw, "raw preserved for debugging")
    # Reasoning should explain why, not be silently empty.
    assert out["reasoning"], "non-empty reasoning when verdict invalid"


def test_parse_verdict_garbage_input() -> None:
    out = _parse_verdict("total nonsense from the judge", _VALID_FINAL_VERDICTS)
    assert_eq(out["verdict"], "unparseable", "garbage → unparseable")
    assert_eq(out["confidence"], 0.0, "garbage confidence → 0")


def test_parse_verdict_valid_step_confidence_clamped() -> None:
    raw = '{"verdict":"passed","confidence":3.5,"reasoning":"ok"}'
    out = _parse_verdict(raw, _VALID_STEP_VERDICTS)
    assert_eq(out["verdict"], "passed", "step pass verdict")
    assert_eq(out["confidence"], 1.0, "clamped to 1")


# ---------------------------------------------------------------------------
# Tests — expected rendering
# ---------------------------------------------------------------------------


def test_render_symbolic() -> None:
    r = _render_expected_for_prompt({"type": "symbolic", "expression": "1/(s+2)"})
    assert_in("symbolic", r, "render symbolic label")
    assert_in("1/(s+2)", r, "render symbolic expression")


def test_render_inequality() -> None:
    r = _render_expected_for_prompt({"type": "inequality", "expression": "re(s) > -2"})
    assert_in("Inequality", r, "render inequality label")
    assert_in("re(s) > -2", r, "render inequality body")


def test_render_composite_all() -> None:
    spec = {
        "type": "composite",
        "aggregation": "all",
        "parts": [
            {"name": "transform", "type": "symbolic", "expression": "1/s"},
            {"name": "roc", "type": "inequality", "expression": "re(s) > 0"},
        ],
    }
    r = _render_expected_for_prompt(spec)
    assert_in("aggregation = all", r, "agg label")
    assert_in("ALL", r, "composite-all emphasis")
    assert_in("transform", r, "part 1 name")
    assert_in("1/s", r, "part 1 expression")
    assert_in("re(s) > 0", r, "part 2 expression")


def test_render_composite_any() -> None:
    spec = {
        "type": "composite",
        "aggregation": "any",
        "parts": [
            {"name": "a", "type": "symbolic", "expression": "z/(z-1)"},
            {"name": "b", "type": "symbolic", "expression": "1/(1-z**(-1))"},
        ],
    }
    r = _render_expected_for_prompt(spec)
    assert_in("aggregation = any", r, "agg label")
    assert_in("AT LEAST ONE", r, "composite-any emphasis")


def test_render_legacy_string() -> None:
    r = _render_expected_for_prompt("2*sin(omega)/omega")
    assert_in("2*sin(omega)/omega", r, "render legacy string body")


def test_render_none() -> None:
    r = _render_expected_for_prompt(None)
    assert_in("no reference", r.lower(), "none spec → no-reference note")


# ---------------------------------------------------------------------------
# Tests — judge_final_answer via FakeJudge
# ---------------------------------------------------------------------------


def test_judge_final_match_via_fake() -> None:
    fake = FakeJudge(
        responses=[
            (
                "STUDENT ANSWER:\n1/(s+2)",
                '{"verdict":"match","confidence":0.95,"reasoning":"algebraic match"}',
            ),
        ]
    )
    out = judge_final_answer(
        "1/(s+2)",
        {"type": "symbolic", "expression": "1/(s+2)"},
        "Compute the Laplace transform of exp(-2*t)*u(t)",
        judge=fake,
    )
    assert_eq(out["status"], "match", "match status")
    assert_eq(out["grader"], "llm_judge", "grader tag")
    assert_eq(out["judge_model"], "fake-judge", "model tag")
    assert_close(out["confidence"], 0.95, 1e-9, "confidence preserved")
    assert_eq(out["expected_type"], "symbolic", "expected_type surfaces")


def test_judge_final_mismatch_via_fake() -> None:
    fake = FakeJudge(
        default='{"verdict":"mismatch","confidence":0.9,"reasoning":"sign error"}'
    )
    out = judge_final_answer(
        "1/(s-2)",
        {"type": "symbolic", "expression": "1/(s+2)"},
        judge=fake,
    )
    assert_eq(out["status"], "mismatch", "mismatch status")
    assert_in("sign error", out["reasoning"], "reasoning surfaced")


def test_judge_final_unparseable_from_garbage() -> None:
    fake = FakeJudge(default="the student wrote nothing useful at all")
    out = judge_final_answer("???", {"type": "symbolic", "expression": "1"}, judge=fake)
    assert_eq(out["status"], "unparseable", "garbage response → unparseable")
    assert out["judge_raw"].startswith("the student"), "raw preserved"


def test_judge_final_abstain_from_partial_answer() -> None:
    fake = FakeJudge(
        default=(
            '{"verdict":"abstain","confidence":0.55,'
            '"reasoning":"partial derivation, unclear"}'
        )
    )
    out = judge_final_answer(
        "X(s) = something...",
        {"type": "symbolic", "expression": "1/(s+2)"},
        judge=fake,
    )
    assert_eq(out["status"], "abstain", "abstain status")


def test_judge_final_composite_prompt_contains_all_parts() -> None:
    fake = FakeJudge(
        default='{"verdict":"match","confidence":1.0,"reasoning":""}'
    )
    spec = {
        "type": "composite",
        "aggregation": "all",
        "parts": [
            {"name": "transform", "type": "symbolic", "expression": "1/s"},
            {"name": "roc", "type": "inequality", "expression": "re(s) > 0"},
        ],
    }
    judge_final_answer("1/s, Re(s) > 0", spec, "whatever", judge=fake)
    assert len(fake.calls) == 1, "judge called once"
    prompt = fake.calls[0]
    assert_in("1/s", prompt, "composite part 1 in prompt")
    assert_in("re(s) > 0", prompt, "composite part 2 in prompt")
    assert_in("aggregation = all", prompt, "aggregation mode in prompt")


def test_judge_final_invalid_verdict_is_unparseable_not_matched() -> None:
    fake = FakeJudge(
        default='{"verdict":"correct","confidence":0.9,"reasoning":"bad label"}'
    )
    out = judge_final_answer("1", {"type": "symbolic", "expression": "1"}, judge=fake)
    assert_eq(out["status"], "unparseable", "invalid verdict → unparseable")


# ---------------------------------------------------------------------------
# Tests — judge_step via FakeJudge
# ---------------------------------------------------------------------------


def test_judge_step_passed() -> None:
    fake = FakeJudge(
        default='{"verdict":"passed","confidence":0.88,"reasoning":"valid move"}'
    )
    out = judge_step(
        {"expression": "Integral(exp(-s*t)*exp(-2*t), (t, 0, oo))", "operation": "apply Laplace definition"},
        prior_steps=[{"expression": "exp(-2*t)*u(t)", "operation": "start"}],
        question="Laplace transform of exp(-2*t)*u(t)",
        judge=fake,
    )
    assert_eq(out["verdict"], "passed", "passed verdict")
    assert_eq(out["grader"], "llm_judge", "grader tag")
    assert len(fake.calls) == 1, "one call"
    prompt = fake.calls[0]
    assert_in("exp(-2*t)*u(t)", prompt, "prior step in prompt")
    assert_in("apply Laplace definition", prompt, "operation in prompt")


def test_judge_step_failed() -> None:
    fake = FakeJudge(
        default='{"verdict":"failed","confidence":0.92,"reasoning":"sign flipped"}'
    )
    out = judge_step(
        {"expression": "1/(s-2)", "operation": "integrate"},
        prior_steps=None,
        question="",
        judge=fake,
    )
    assert_eq(out["verdict"], "failed", "failed verdict")


def test_judge_step_invalid_verdict_is_unparseable() -> None:
    # "match" is a final-answer verdict, not a step verdict.
    fake = FakeJudge(default='{"verdict":"match","confidence":0.9,"reasoning":""}')
    out = judge_step(
        {"expression": "x", "operation": "y"}, judge=fake
    )
    assert_eq(out["verdict"], "unparseable", "step verdicts do not include 'match'")


def test_judge_step_empty_prior_steps_shows_none() -> None:
    fake = FakeJudge(default='{"verdict":"abstain","confidence":0.5,"reasoning":""}')
    judge_step({"expression": "e", "operation": "o"}, judge=fake)
    prompt = fake.calls[0]
    assert_in("(none)", prompt, "empty prior steps renders as '(none)'")


# ---------------------------------------------------------------------------
# Tests — FakeJudge substring routing
# ---------------------------------------------------------------------------


def test_fake_judge_routes_by_substring_order() -> None:
    fake = FakeJudge(
        responses=[
            ("FIRST_MARKER", '{"verdict":"match","confidence":1.0,"reasoning":""}'),
            ("SECOND_MARKER", '{"verdict":"mismatch","confidence":1.0,"reasoning":""}'),
        ],
        default='{"verdict":"abstain","confidence":0.5,"reasoning":""}',
    )

    # Neither marker present → default (abstain).
    out1 = judge_final_answer("plain", "plain", "plain", judge=fake)
    assert_eq(out1["status"], "abstain", "default applied when no marker matches")

    # Second marker present → second response.
    out2 = judge_final_answer(
        "xyz SECOND_MARKER xyz", "expected", "question", judge=fake
    )
    assert_eq(out2["status"], "mismatch", "second-marker response routed")

    # Both markers present → first wins (iteration order).
    out3 = judge_final_answer(
        "FIRST_MARKER and SECOND_MARKER together", "expected", "question", judge=fake
    )
    assert_eq(out3["status"], "match", "first-matching-substring wins")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("safe_float clamps + garbage", test_safe_float_clamps_and_handles_garbage),
    ("extract_json direct", test_extract_json_direct),
    ("extract_json fenced", test_extract_json_fenced),
    ("extract_json regex fallback", test_extract_json_regex_fallback),
    ("extract_json garbage → None", test_extract_json_garbage_returns_none),
    ("parse_verdict unknown → unparseable", test_parse_verdict_unknown_verdict_marked_unparseable),
    ("parse_verdict garbage input", test_parse_verdict_garbage_input),
    ("parse_verdict clamps confidence", test_parse_verdict_valid_step_confidence_clamped),
    ("render symbolic", test_render_symbolic),
    ("render inequality", test_render_inequality),
    ("render composite all", test_render_composite_all),
    ("render composite any", test_render_composite_any),
    ("render legacy string", test_render_legacy_string),
    ("render None", test_render_none),
    ("judge_final match", test_judge_final_match_via_fake),
    ("judge_final mismatch", test_judge_final_mismatch_via_fake),
    ("judge_final unparseable", test_judge_final_unparseable_from_garbage),
    ("judge_final abstain", test_judge_final_abstain_from_partial_answer),
    ("judge_final composite prompt", test_judge_final_composite_prompt_contains_all_parts),
    ("judge_final invalid verdict", test_judge_final_invalid_verdict_is_unparseable_not_matched),
    ("judge_step passed", test_judge_step_passed),
    ("judge_step failed", test_judge_step_failed),
    ("judge_step invalid verdict", test_judge_step_invalid_verdict_is_unparseable),
    ("judge_step empty prior shows (none)", test_judge_step_empty_prior_steps_shows_none),
    ("FakeJudge substring routing order", test_fake_judge_routes_by_substring_order),
]


def main() -> int:
    passed = 0
    failed = 0
    print(f"Running {len(TESTS)} llm_judge tests...\n")
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
