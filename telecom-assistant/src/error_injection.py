"""
Synthetic error-injection harness for grader evaluation.

What this measures
------------------
For a set of known-correct ("gold") derivations we inject controlled
perturbations into single steps and ask one or more graders — the symbolic
verifier and (optionally) an LLM-as-judge — to flag the mutated step.

Each trial has a declared severity:

  * "error"       : a real mathematical mistake was introduced.  A good
                    grader should return "failed".
  * "equivalent"  : the expression was rewritten in a notationally different
                    but algebraically equivalent way.  A good grader should
                    NOT return "failed" (that would be a false positive).
  * "control"     : no mutation applied.  Any "failed" verdict here is a
                    false positive on the baseline derivation itself.

From these trial outcomes we compute, per grader and per mutation family,
standard classifier metrics: precision, recall, FPR, and abstain rate.
This is the experimental primitive the paper's error-injection study
depends on — the output JSON is the sole input for the results table and
the PR curve.

What's new vs the original
--------------------------
* 14 mutation kinds in 4 families (algebraic / bounds / identifier /
  equivalent) plus a control, up from 5 uniform-severity mutators.
* Severity tagging so precision / recall / FPR are well-defined.
* Pluggable grader abstraction: `SymbolicGrader` always, `LLMJudgeGrader`
  opt-in via ``--with-judge`` (backed by any `JudgeClient`, including
  `FakeJudge` for deterministic offline runs).
* Confusion matrix + per-family metrics in the saved report.
* Backward-compatible shims: `MUTATORS`, `inject_and_verify`, and
  `run_suite` remain importable with their original semantics.

Usage
-----
  python -m src.error_injection                        # symbolic only
  python -m src.error_injection --with-judge           # + real LLM judge
  python -m src.error_injection --fixtures gold.json   # extra fixtures
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union

from src.verifier import verify_derivation


# ─────────────────────────────── fixtures ─────────────────────────────────────
# Hand-authored gold derivations we've smoke-tested against the symbolic
# verifier.  Keeping them in-tree means the harness runs without needing to
# label real LLM output first.  Every fixture's unmutated steps should
# verify cleanly (`is_valid != False`); the harness will filter out any
# fixture that doesn't and warn.

GOLD_FIXTURES: List[Dict[str, Any]] = [
    {
        "concept": "Fourier transform of exp(-a*t)*u(t)",
        "final_result": "1/(a + I*omega)",
        "steps": [
            {
                "step_number": 1,
                "description": "Definition of Fourier transform for causal signal",
                "expression": "integrate(exp(-a*t)*exp(-I*omega*t), (t, 0, oo))",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Combine exponentials",
                "expression": "integrate(exp(-(a + I*omega)*t), (t, 0, oo))",
                "operation": "algebra",
            },
            {
                "step_number": 3,
                "description": "Evaluate the integral",
                "expression": "1/(a + I*omega)",
                "operation": "integration",
            },
        ],
    },
    {
        "concept": "Time-shift property for a causal exponential",
        "final_result": "exp(-I*omega*t0)*(1/(a + I*omega))",
        "steps": [
            {
                "step_number": 1,
                "description": "Fourier transform of the shifted causal signal exp(-a*(t - t0))*u(t - t0)",
                "expression": "integrate(exp(-a*(t - t0))*exp(-I*omega*t), (t, t0, oo))",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Substitute t -> t + t0 and factor the shift out of the integral",
                "expression": "exp(-I*omega*t0)*integrate(exp(-a*t)*exp(-I*omega*t), (t, 0, oo))",
                "operation": "substitution",
            },
            {
                "step_number": 3,
                "description": "Evaluate the remaining integral",
                "expression": "exp(-I*omega*t0)*(1/(a + I*omega))",
                "operation": "integration",
            },
        ],
    },
    {
        "concept": "Pure algebra — quadratic identity",
        "final_result": "omega**2 + 2*omega + 1",
        "steps": [
            {
                "step_number": 1,
                "description": "Start with (omega + 1) squared",
                "expression": "(omega + 1)**2",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Expand",
                "expression": "omega**2 + 2*omega + 1",
                "operation": "algebra",
            },
        ],
    },
    {
        "concept": "Trig identity",
        "final_result": "2*sin(omega)*cos(omega)",
        "steps": [
            {
                "step_number": 1,
                "description": "Start with double-angle form",
                "expression": "sin(2*omega)",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Apply double-angle identity",
                "expression": "2*sin(omega)*cos(omega)",
                "operation": "identity",
            },
        ],
    },
    {
        "concept": "Euler identity in sinc",
        "final_result": "2*sin(omega)/omega",
        "steps": [
            {
                "step_number": 1,
                "description": "Start with complex-exponential difference",
                "expression": "(exp(I*omega) - exp(-I*omega))/(I*omega)",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Apply Euler identity",
                "expression": "2*I*sin(omega)/(I*omega)",
                "operation": "identity",
            },
            {
                "step_number": 3,
                "description": "Cancel I",
                "expression": "2*sin(omega)/omega",
                "operation": "algebra",
            },
        ],
    },
    {
        "concept": "Laplace transform of exp(-a*t)*u(t)",
        "final_result": "1/(s + a)",
        "steps": [
            {
                "step_number": 1,
                "description": "Apply Laplace definition",
                "expression": "integrate(exp(-a*t)*exp(-s*t), (t, 0, oo))",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Combine exponentials",
                "expression": "integrate(exp(-(a + s)*t), (t, 0, oo))",
                "operation": "algebra",
            },
            {
                "step_number": 3,
                "description": "Evaluate integral",
                "expression": "1/(s + a)",
                "operation": "integration",
            },
        ],
    },
    {
        "concept": "Partial fraction expansion",
        "final_result": "1/(2*(s - 1)) - 1/(2*(s + 1))",
        "steps": [
            {
                "step_number": 1,
                "description": "Start with rational",
                "expression": "1/(s**2 - 1)",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Factor denominator",
                "expression": "1/((s - 1)*(s + 1))",
                "operation": "factor",
            },
            {
                "step_number": 3,
                "description": "Partial-fraction decomposition",
                "expression": "1/(2*(s - 1)) - 1/(2*(s + 1))",
                "operation": "partial_fractions",
            },
        ],
    },
    {
        "concept": "Geometric series sum",
        "final_result": "1/(1 - r)",
        "steps": [
            {
                "step_number": 1,
                "description": "Series definition",
                "expression": "Sum(r**n, (n, 0, oo))",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Closed form for |r| < 1",
                "expression": "1/(1 - r)",
                "operation": "geometric_sum",
            },
        ],
    },
    {
        "concept": "Derivative of polynomial",
        "final_result": "3*x**2",
        "steps": [
            {
                "step_number": 1,
                "description": "Set up the derivative",
                "expression": "diff(x**3, x)",
                "operation": "definition",
            },
            {
                "step_number": 2,
                "description": "Apply the power rule",
                "expression": "3*x**2",
                "operation": "differentiation",
            },
        ],
    },
]


# ───────────────────────────── mutation specs ─────────────────────────────────


@dataclass
class MutationSpec:
    """Declarative description of a single mutation.

    ``apply`` returns the mutated expression string or ``None`` if the
    mutation isn't applicable to this particular expression (in which case
    the harness records a "skipped" trial rather than a false measurement).
    """

    kind: str
    family: str  # algebraic | bounds | identifier | equivalent | control
    severity: str  # "error" | "equivalent" | "control"
    description: str
    apply: Callable[[str], Optional[str]]


# ── algebraic mutations (severity = "error") ────────────────────────────────


def _mut_sign_flip(expr: str) -> Optional[str]:
    s = expr.strip()
    if s.startswith("-"):
        return s[1:]
    if s.startswith("+"):
        return "-" + s[1:]
    return f"-({s})"


def _mut_coeff_x2(expr: str) -> Optional[str]:
    return f"2*({expr})"


def _mut_coeff_half(expr: str) -> Optional[str]:
    return f"({expr})/2"


def _mut_drop_term(expr: str) -> Optional[str]:
    depth = 0
    last_add = -1
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0 and i > 0:
            last_add = i
    if last_add > 0:
        return expr[:last_add].rstrip()
    # No top-level additive term to drop — not applicable.
    return None


def _mut_exponent_bump(expr: str) -> Optional[str]:
    m = re.search(r"\*\*\s*(\d+)", expr)
    if not m:
        return None
    n = int(m.group(1))
    start, end = m.span(1)
    return expr[:start] + str(n + 1) + expr[end:]


def _mut_constant_shift(expr: str) -> Optional[str]:
    return f"({expr}) + 1"


def _mut_missing_factor_of_2(expr: str) -> Optional[str]:
    # Classic mistake: drops a factor of 2 from a leading numeric coefficient
    # if present, otherwise divides the whole expression by 2.
    m = re.match(r"\s*2\s*\*\s*(.*)", expr)
    if m:
        return m.group(1)
    return f"({expr})/2"


# ── bounds mutations (severity = "error") ───────────────────────────────────

_LIMIT_TUPLE_RE = re.compile(
    r"\(\s*([A-Za-z_]\w*)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"
)


def _first_limit_tuple(expr: str) -> Optional[re.Match]:
    """Return the first ``(var, a, b)`` triple that looks like integration
    or summation bounds.  We require the triple to appear inside an
    ``integrate(...)`` or ``Sum(...)`` call to avoid matching generic tuples.
    """
    if not any(k in expr for k in ("integrate(", "Integral(", "Sum(", "sum(")):
        return None
    return _LIMIT_TUPLE_RE.search(expr)


def _mut_swap_integral_limits(expr: str) -> Optional[str]:
    m = _first_limit_tuple(expr)
    if not m:
        return None
    var, a, b = m.group(1), m.group(2), m.group(3)
    swapped = f"({var}, {b}, {a})"
    return expr[: m.start()] + swapped + expr[m.end() :]


def _mut_drop_upper_limit(expr: str) -> Optional[str]:
    m = _first_limit_tuple(expr)
    if not m:
        return None
    var, a, b = m.group(1), m.group(2), m.group(3)
    # Change upper limit from its true value to a nearby but wrong one.
    new_b = "1" if b.strip() != "1" else "2"
    replaced = f"({var}, {a}, {new_b})"
    return expr[: m.start()] + replaced + expr[m.end() :]


# ── identifier mutations (severity = "error") ───────────────────────────────


def _is_bound_variable(expr: str, name: str) -> bool:
    """True if ``name`` appears as the first element of an ``(x, a, b)`` tuple
    inside an integrate / Integral / Sum / sum / diff call.  Renaming a bound
    variable is a no-op on the expression's value, so such mutations would
    produce a mis-labeled trial (injected "error" that's actually equivalent).
    """
    if not re.search(r"(integrate|Integral|Sum|sum|diff)\s*\(", expr):
        return False
    pattern = rf"\(\s*{re.escape(name)}\s*,"
    return bool(re.search(pattern, expr))


def _mut_rename_variable(expr: str) -> Optional[str]:
    # Rename the first well-known FREE signal variable uniformly inside this
    # step.  Because the prior step still uses the original name, the two
    # steps become algebraically inconsistent — a real error a grader should
    # catch.  We skip any candidate that appears as a bound/dummy variable
    # in this step (e.g. the ``t`` in ``integrate(..., (t, 0, oo))``); that
    # rename would be algebraically trivial and mis-classify the trial.
    candidates = (
        ("omega", "Omega"),
        ("s", "p"),
        ("z", "zeta"),
        ("r", "rho"),
        ("a", "alpha"),
        ("x", "xi"),
    )
    for old, new in candidates:
        pattern = rf"\b{re.escape(old)}\b"
        if re.search(pattern, expr) and not _is_bound_variable(expr, old):
            return re.sub(pattern, new, expr)
    return None


# ── equivalent-notation mutations (severity = "equivalent") ─────────────────
# A good grader should NOT flag these.  If it does, that's a false positive
# on over-strict grading — exactly the kind of issue the paper needs to
# report.


def _mut_commute_operands(expr: str) -> Optional[str]:
    """Swap the operands of the first top-level ``+`` or ``*``.

    Only applies if the operator is outside parentheses; otherwise the
    mutation would leak into a different sub-expression.
    """
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+*" and 0 < i < len(expr) - 1:
            left = expr[:i].rstrip()
            right = expr[i + 1 :].lstrip()
            # Skip unary plus (start of string handled above) and ``**``.
            if ch == "*" and (expr[i - 1] == "*" or expr[i + 1 : i + 2] == "*"):
                continue
            return f"{right} {ch} {left}"
    return None


def _mut_expand_power(expr: str) -> Optional[str]:
    """Rewrite the first ``X**2`` as ``(X)*(X)`` — algebraically equivalent."""
    m = re.search(r"([A-Za-z_]\w*|\([^()]+\))\s*\*\*\s*2\b", expr)
    if not m:
        return None
    base = m.group(1)
    # Strip outer parens on base if they're already there.
    paren_base = base if base.startswith("(") else f"({base})"
    replaced = f"{paren_base}*{paren_base}"
    return expr[: m.start()] + replaced + expr[m.end() :]


def _mut_distribute(expr: str) -> Optional[str]:
    """Rewrite the first ``k*(a + b)`` as ``k*a + k*b``."""
    m = re.search(
        r"(\d+|[A-Za-z_]\w*)\s*\*\s*\(\s*([^()+\-]+?)\s*([+\-])\s*([^()]+?)\s*\)",
        expr,
    )
    if not m:
        return None
    k, a, op, b = m.group(1), m.group(2), m.group(3), m.group(4)
    replaced = f"{k}*{a} {op} {k}*{b}"
    return expr[: m.start()] + replaced + expr[m.end() :]


# ── control (severity = "control") ──────────────────────────────────────────


def _mut_noop(expr: str) -> Optional[str]:
    return expr


MUTATION_SPECS: List[MutationSpec] = [
    # algebraic
    MutationSpec("sign_flip", "algebraic", "error", "negate leading sign", _mut_sign_flip),
    MutationSpec("coeff_x2", "algebraic", "error", "multiply by 2", _mut_coeff_x2),
    MutationSpec("coeff_half", "algebraic", "error", "divide by 2", _mut_coeff_half),
    MutationSpec("drop_term", "algebraic", "error", "drop last additive term", _mut_drop_term),
    MutationSpec("exponent_bump", "algebraic", "error", "increment first integer exponent", _mut_exponent_bump),
    MutationSpec("constant_shift", "algebraic", "error", "add 1", _mut_constant_shift),
    MutationSpec("missing_factor_of_2", "algebraic", "error", "drop leading factor of 2", _mut_missing_factor_of_2),
    # bounds
    MutationSpec("swap_integral_limits", "bounds", "error", "swap a<->b in first integral/sum", _mut_swap_integral_limits),
    MutationSpec("drop_upper_limit", "bounds", "error", "truncate upper limit of first integral/sum", _mut_drop_upper_limit),
    # identifier
    MutationSpec("rename_variable", "identifier", "error", "rename a key symbol uniformly in this step", _mut_rename_variable),
    # equivalent (should NOT be flagged)
    MutationSpec("commute_operands", "equivalent", "equivalent", "swap operands of first top-level +/*", _mut_commute_operands),
    MutationSpec("expand_power", "equivalent", "equivalent", "rewrite X**2 as X*X", _mut_expand_power),
    MutationSpec("distribute", "equivalent", "equivalent", "expand k*(a+b) into k*a+k*b", _mut_distribute),
    # control
    MutationSpec("no_op", "control", "control", "identity (no mutation)", _mut_noop),
]


# Backward-compatible view: original ``MUTATORS`` dict exposing only the
# error-severity algebraic mutators that the old API documented.
MUTATORS: Dict[str, Callable[[str], str]] = {
    spec.kind: (lambda e, s=spec: s.apply(e) or e)
    for spec in MUTATION_SPECS
    if spec.severity == "error" and spec.family == "algebraic" and spec.kind != "coeff_half"
}


# ─────────────────────────────── graders ──────────────────────────────────────


@dataclass
class GraderVerdict:
    verdict: str  # "passed" | "failed" | "abstain" | "error"
    confidence: Optional[float] = None
    reasoning: str = ""


class Grader(Protocol):
    name: str

    def grade_step(
        self, derivation: Dict[str, Any], step_index: int
    ) -> GraderVerdict: ...


@dataclass
class SymbolicGrader:
    """Wraps :func:`src.verifier.verify_derivation` and picks the verdict
    for a single step index.  We use ``strict_mode=True`` because gold
    fixtures are known-correct by construction — any numerical mismatch
    between consecutive steps is therefore a real error, not an ambiguity.
    """

    name: str = "symbolic"

    def grade_step(
        self, derivation: Dict[str, Any], step_index: int
    ) -> GraderVerdict:
        results = verify_derivation(derivation, strict_mode=True)
        if step_index >= len(results):
            return GraderVerdict("abstain", reasoning="no verifier result for index")
        r = results[step_index]
        if r.is_valid is True:
            v = "passed"
        elif r.is_valid is False:
            v = "failed"
        else:
            v = "abstain"
        return GraderVerdict(verdict=v, reasoning=r.error_message or "")


@dataclass
class LLMJudgeGrader:
    """Wraps :func:`src.llm_judge.judge_step` for the same step-level contract.

    `judge` is injected so tests can use :class:`FakeJudge` without touching
    the network.  The grader name is ``llm_judge_<model>`` so multiple
    judges (e.g. different backends) can coexist in a single run.
    """

    judge: Any
    name: str = "llm_judge"
    sleep_after_s: float = 0.0

    def __post_init__(self) -> None:
        model = getattr(self.judge, "model", "unknown")
        if not self.name.endswith(model):
            self.name = f"llm_judge:{model}"

    def grade_step(
        self, derivation: Dict[str, Any], step_index: int
    ) -> GraderVerdict:
        from src.llm_judge import judge_step

        step = derivation["steps"][step_index]
        prior = derivation["steps"][:step_index]
        out = judge_step(
            step,
            prior_steps=prior,
            question=derivation.get("concept", ""),
            judge=self.judge,
        )
        verdict = out.get("verdict", "abstain")
        if verdict == "unparseable":
            verdict = "abstain"  # treat parse failure as abstain for this study
        if self.sleep_after_s > 0:
            time.sleep(self.sleep_after_s)
        return GraderVerdict(
            verdict=verdict,
            confidence=out.get("confidence"),
            reasoning=out.get("reasoning", "")[:400],
        )


# ─────────────────────────────── trials ───────────────────────────────────────


@dataclass
class Trial:
    """One (fixture, step, mutation) combination with all graders' verdicts.

    ``skipped_reason`` is non-None when the mutation wasn't applicable to
    the original expression (e.g. ``drop_term`` on a single-term step).
    Skipped trials are excluded from metric computation.
    """

    gold_id: int
    concept: str
    step_index: int
    mutation_kind: str
    family: str
    severity: str
    original_expr: str
    mutated_expr: Optional[str]
    verdicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skipped_reason: Optional[str] = None
    baseline_undecidable: bool = False
    """True when this trial's fixture had an undecidable *unmutated* step.

    Set by ``run_suite`` from ``filter_viable_fixtures``'s findings. It says
    nothing about this trial's own verdict: it marks that the verifier could
    not decide the fixture's baseline, so a flag on this trial cannot be
    cleanly attributed to the mutation. Downstream analysis can filter or
    stratify on it without re-running the verifier.
    """


def inject_and_grade(
    gold: Dict[str, Any],
    step_index: int,
    spec: MutationSpec,
    graders: List[Grader],
    gold_id: int = 0,
) -> Trial:
    """Apply ``spec`` to ``gold.steps[step_index]`` and grade with each grader.

    Mutations that return ``None`` produce a skipped trial so we can report
    "N/A for this expression" in the final table without polluting metrics.
    """
    mutated = copy.deepcopy(gold)
    step = mutated["steps"][step_index]
    original_expr = step["expression"]
    new_expr = spec.apply(original_expr)
    if new_expr is None:
        return Trial(
            gold_id=gold_id,
            concept=gold.get("concept", ""),
            step_index=step_index,
            mutation_kind=spec.kind,
            family=spec.family,
            severity=spec.severity,
            original_expr=original_expr,
            mutated_expr=None,
            skipped_reason="mutation_not_applicable",
        )
    step["expression"] = new_expr
    verdicts: Dict[str, Dict[str, Any]] = {}
    for g in graders:
        try:
            v = g.grade_step(mutated, step_index)
            verdicts[g.name] = {
                "verdict": v.verdict,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            }
        except Exception as e:
            verdicts[g.name] = {
                "verdict": "error",
                "confidence": None,
                "reasoning": f"{type(e).__name__}: {e}",
            }
    return Trial(
        gold_id=gold_id,
        concept=gold.get("concept", ""),
        step_index=step_index,
        mutation_kind=spec.kind,
        family=spec.family,
        severity=spec.severity,
        original_expr=original_expr,
        mutated_expr=new_expr,
        verdicts=verdicts,
    )


def filter_viable_fixtures(
    golds: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Union[str, Dict[str, Any]]]]:
    """Screen fixtures on their *unmutated* steps, so that a flag on a mutated
    trial is attributable to the mutation rather than to a bad baseline.

    Three baseline outcomes, deliberately not handled alike:

    * ``is_valid is False`` — the verifier *refutes* the step. The fixture is a
      confound and is **dropped**, as before.
    * ``is_valid is None`` at a step with ``equivalence_checked is True`` — the
      verifier attempted the comparison and returned no answer. This is the
      case that used to pass the gate silently, because ``None is not False``.
      It is now recorded and propagated, and the fixture is **retained**.
    * ``is_valid is None`` at a step with ``equivalence_checked is False`` — a
      genuine first step, or a step whose expression did not parse. Nothing was
      compared, so the non-answer carries no information about the fixture and
      **must not warn**.

    Retained rather than dropped on the undecidable case because abstention is
    this verifier's normal operating mode on this testbed (it abstains on 64.4%
    of graded injection trials). A drop rule would make corpus membership a
    function of verifier coverage and would bias the suite toward the steps
    symbolic tooling already handles, reading a fact about the verifier as a
    fact about the fixture. See notes/fixture_integrity_notes.md §6b.

    Returns ``(viable, warnings)``. Warnings stay plain strings for the drop
    and error cases; the undecidable case is a structured dict so downstream
    code can query it rather than parse prose. In that dict ``gold_id`` is the
    index into ``viable`` — the ``gold_id`` the fixture's trials will carry —
    and ``input_index`` is its position in ``golds``; the two diverge as soon
    as anything ahead of it is dropped. ``step_indices`` are positions in the
    fixture's ``steps`` list, matching ``Trial.step_index``.
    """
    viable: List[Dict[str, Any]] = []
    warnings: List[Union[str, Dict[str, Any]]] = []
    for i, g in enumerate(golds):
        try:
            results = verify_derivation(g, strict_mode=True)
            bad_steps = [r.step_number for r in results if r.is_valid is False]
            # Positional, not r.step_number: these line up with the
            # Trial.step_index values run_suite emits.
            undecidable = [
                pos
                for pos, r in enumerate(results)
                if r.is_valid is None and r.equivalence_checked
            ]
        except Exception as e:
            warnings.append(f"Fixture {i} ({g.get('concept','?')}) errored: {e}")
            continue
        if bad_steps:
            warnings.append(
                f"Fixture {i} ({g.get('concept','?')}) flagged at baseline: "
                f"steps {bad_steps} — dropping from suite"
            )
            continue
        viable.append(g)
        if undecidable:
            warnings.append(
                {
                    "kind": "baseline_undecidable",
                    "gold_id": len(viable) - 1,
                    "input_index": i,
                    "concept": g.get("concept", "?"),
                    "step_indices": undecidable,
                    "detail": (
                        "verifier attempted equivalence and returned no verdict"
                    ),
                }
            )
    return viable, warnings


def run_suite(
    golds: Optional[List[Dict[str, Any]]] = None,
    specs: Optional[List[MutationSpec]] = None,
    graders: Optional[List[Grader]] = None,
    include_controls: bool = True,
    undecidable_gold_ids: Optional[Set[int]] = None,
) -> List[Trial]:
    """Produce one trial per (fixture, step ≥ 1, mutation) combination.

    First steps are skipped because they are definitions without a
    preceding step to compare against.

    ``undecidable_gold_ids`` are indices into ``golds`` whose baseline the
    verifier could not decide, as reported by ``filter_viable_fixtures``. Every
    trial from such a fixture is marked ``baseline_undecidable``.
    """
    golds = golds if golds is not None else GOLD_FIXTURES
    specs = specs if specs is not None else [
        s for s in MUTATION_SPECS if include_controls or s.severity != "control"
    ]
    graders = graders if graders is not None else [SymbolicGrader()]
    undecidable = undecidable_gold_ids or set()
    trials: List[Trial] = []
    for gold_id, gold in enumerate(golds):
        steps = gold.get("steps", [])
        for step_index in range(1, len(steps)):
            for spec in specs:
                trial = inject_and_grade(gold, step_index, spec, graders, gold_id)
                trial.baseline_undecidable = gold_id in undecidable
                trials.append(trial)
    return trials


# ─────────────────────────────── metrics ──────────────────────────────────────


def _empty_counts() -> Dict[str, int]:
    return {"passed": 0, "failed": 0, "abstain": 0, "error": 0}


def compute_metrics(
    trials: List[Trial], grader_names: List[str]
) -> Dict[str, Any]:
    """Summarize trial outcomes into per-grader, per-family classifier metrics.

    For a grader:
      - On error / control-type trials:
          TP = mutation had severity in {"error"} and verdict == "failed"
          FN = severity == "error" and verdict == "passed"
          AbsE = severity == "error" and verdict in {"abstain", "error"}
          ("error" is a crash/unparseable response, treated as an abstain
           on a true-error trial, matching build_threeway_table.py.)
      - On equivalent / control trials:
          FP = severity in {"equivalent", "control"} and verdict == "failed"
          TN = severity in {"equivalent", "control"} and verdict == "passed"
          AbsN = severity in {"equivalent", "control"} and verdict in
          {"abstain", "error"} (tracked separately — neither a false
           positive nor a confirmation).

    We report precision, recall, FPR, abstain rate; also a confusion matrix
    keyed by (severity, verdict) for inspection.

    Denominators, which must match ``scripts/build_threeway_table.py`` (the
    two implementations are independent and are required to agree):

      precision      = TP / (TP + FP)            committed predictions only
      recall_strict  = TP / (TP + FN + AbsE)     abstain-on-error is a miss
      recall_committed = TP / (TP + FN)          abstains excluded
      fpr            = FP / (FP + TN)            covered negatives only;
                                                 AbsN is NOT in the denominator
      abstain_rate   = (AbsE + AbsN) / total

    Rates whose denominator is zero are reported as ``None``, not 0.0.
    """
    out: Dict[str, Any] = {}
    for name in grader_names:
        per_family: Dict[str, Dict[str, int]] = {}
        confusion: Dict[str, Dict[str, int]] = {}
        counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "abstain_error": 0, "abstain_eq": 0, "total": 0}
        for t in trials:
            if t.skipped_reason:
                continue
            v = (t.verdicts.get(name) or {}).get("verdict", "error")
            counts["total"] += 1
            bucket = confusion.setdefault(t.severity, _empty_counts())
            bucket[v] = bucket.get(v, 0) + 1
            fam = per_family.setdefault(
                t.family,
                {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "abstain_error": 0, "abstain_eq": 0, "total": 0},
            )
            fam["total"] += 1

            if t.severity == "error":
                if v == "failed":
                    counts["tp"] += 1
                    fam["tp"] += 1
                elif v == "passed":
                    counts["fn"] += 1
                    fam["fn"] += 1
                elif v in ("abstain", "error"):
                    # A crash/unparseable verdict ("error") counts as abstain
                    # on a true-error trial, matching the paper's stated
                    # metric definition (AbsE = abstain or crash) and
                    # build_threeway_table.py's overall-metric computation.
                    counts["abstain_error"] += 1
                    fam["abstain_error"] += 1
            elif t.severity in {"equivalent", "control"}:
                if v == "failed":
                    counts["fp"] += 1
                    fam["fp"] += 1
                elif v == "passed":
                    counts["tn"] += 1
                    fam["tn"] += 1
                elif v in ("abstain", "error"):
                    # Same crash-as-abstain treatment as the error-severity
                    # branch above, for consistency with build_threeway_table.py.
                    counts["abstain_eq"] += 1
                    fam["abstain_eq"] += 1

        def _rates(c: Dict[str, int]) -> Dict[str, Any]:
            tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
            errors_total = tp + fn + c.get("abstain_error", 0)
            neg_committed = fp + tn
            # "Strict" recall counts abstains on error trials as misses.
            recall_strict = tp / errors_total if errors_total else None
            # "Ever-commits" recall excludes abstains from the denominator.
            recall_committed = tp / (tp + fn) if (tp + fn) else None
            precision = tp / (tp + fp) if (tp + fp) else None
            # Covered FPR: committed negatives only. AbsN is neither a false
            # positive nor a confirmation, so it leaves the denominator
            # entirely -- unlike AbsE, which is a miss and stays in
            # recall_strict's. With no committed negatives the rate is
            # undefined (None), not 0.0.
            fpr = fp / neg_committed if neg_committed else None
            abstain_rate = (
                (c.get("abstain_error", 0) + c.get("abstain_eq", 0)) / c["total"]
                if c["total"]
                else None
            )
            return {
                "precision": precision,
                "recall_strict": recall_strict,
                "recall_committed": recall_committed,
                "fpr": fpr,
                "abstain_rate": abstain_rate,
                **c,
            }

        out[name] = {
            "overall": _rates(counts),
            "by_family": {f: _rates(c) for f, c in per_family.items()},
            "confusion": confusion,
        }
    return out


# ─────────────────────────────── reporting ────────────────────────────────────


def _fmt_pct(x: Optional[float]) -> str:
    return f"{100 * x:5.1f}%" if isinstance(x, float) else "   n/a"


def print_report(trials: List[Trial], metrics: Dict[str, Any]) -> None:
    n_total = len(trials)
    n_skipped = sum(1 for t in trials if t.skipped_reason)
    print("=" * 78)
    print("SYNTHETIC ERROR INJECTION — GRADER SENSITIVITY")
    print("=" * 78)
    print(f"Trials total        : {n_total}")
    print(f"Trials skipped      : {n_skipped}  (mutation not applicable)")
    print(f"Trials evaluated    : {n_total - n_skipped}")
    print()
    for name, m in metrics.items():
        overall = m["overall"]
        print(f"── {name} ──────────────────────────────────────────────────")
        print(
            f"  overall:  precision={_fmt_pct(overall['precision'])}  "
            f"recall(strict)={_fmt_pct(overall['recall_strict'])}  "
            f"FPR={_fmt_pct(overall['fpr'])}  "
            f"abstain={_fmt_pct(overall['abstain_rate'])}"
        )
        print(f"  counts:   tp={overall['tp']:3d}  fn={overall['fn']:3d}  "
              f"fp={overall['fp']:3d}  tn={overall['tn']:3d}  "
              f"abs(err)={overall['abstain_error']:3d}  "
              f"abs(eq)={overall['abstain_eq']:3d}")
        print("  by family:")
        print(f"    {'family':<12} {'prec':>7} {'rec(s)':>7} {'FPR':>7} "
              f"{'abst':>7} {'total':>6}")
        for fam, c in m["by_family"].items():
            print(
                f"    {fam:<12} {_fmt_pct(c['precision']):>7} "
                f"{_fmt_pct(c['recall_strict']):>7} {_fmt_pct(c['fpr']):>7} "
                f"{_fmt_pct(c['abstain_rate']):>7} {c['total']:>6}"
            )
        print()
    print("=" * 78)


# ──────────────────────────── backward-compat ────────────────────────────────
# Thin shim: older callers (and README examples) expect a flat dict result.


def inject_and_verify(
    gold: Dict[str, Any], step_index: int, kind: str
) -> Dict[str, Any]:
    """Legacy single-grader entrypoint.

    Kept so external scripts / docs that import this name continue to work.
    Uses the symbolic grader only; new code should prefer
    :func:`inject_and_grade` with explicit grader list.
    """
    spec = next((s for s in MUTATION_SPECS if s.kind == kind), None)
    if spec is None:
        raise KeyError(f"unknown mutation kind: {kind!r}")
    trial = inject_and_grade(gold, step_index, spec, [SymbolicGrader()])
    v = (trial.verdicts.get("symbolic") or {}).get("verdict", "error")
    return {
        "concept": gold.get("concept", ""),
        "step_index": step_index,
        "mutation": kind,
        "original_expr": trial.original_expr,
        "mutated_expr": trial.mutated_expr,
        "flagged_at_step": v == "failed",
        "flagged_anywhere": v == "failed",
        "verifier_message": (trial.verdicts.get("symbolic") or {}).get("reasoning", ""),
        "abstained_at_step": v == "abstain",
    }


# ──────────────────────────────── CLI ─────────────────────────────────────────


def _build_default_graders(
    with_judge: bool, judge_sleep_s: float = 0.0
) -> List[Grader]:
    graders: List[Grader] = [SymbolicGrader()]
    if with_judge:
        from src.llm_judge import make_judge_from_env

        graders.append(
            LLMJudgeGrader(
                judge=make_judge_from_env(), sleep_after_s=judge_sleep_s
            )
        )
    return graders


def _trial_to_dict(t: Trial) -> Dict[str, Any]:
    d = asdict(t)
    # Emit baseline_undecidable only when it is set. Absent means False, and
    # omitting it keeps unaffected trials byte-identical to the frozen reports,
    # which the regrade/splice workflow diffs against.
    if not d.get("baseline_undecidable"):
        d.pop("baseline_undecidable", None)
    return d


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help="Path to a JSON file with extra gold derivations (list or single dict).",
    )
    ap.add_argument(
        "--with-judge",
        action="store_true",
        help="Also run the LLM-as-judge grader (uses Cerebras endpoint).",
    )
    ap.add_argument(
        "--judge-sleep-s",
        type=float,
        default=1.0,
        help="Sleep after each LLM judge call when --with-judge (rate limiting). "
        "Use 0 to disable.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="results/error_injection_report.json",
        help="Where to write the machine-readable report.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    random.seed(args.seed)

    golds = list(GOLD_FIXTURES)
    if args.fixtures:
        with open(args.fixtures) as f:
            extra = json.load(f)
        if isinstance(extra, dict):
            extra = [extra]
        golds.extend(extra)

    viable, warnings = filter_viable_fixtures(golds)
    for w in warnings:
        if isinstance(w, dict) and w.get("kind") == "baseline_undecidable":
            print(
                f"[warn] Fixture {w['gold_id']} ({w['concept']}) undecidable at "
                f"baseline: step_indices {w['step_indices']} — keeping it in the "
                f"suite and flagging its trials"
            )
        else:
            print(f"[warn] {w}")
    undecidable_gold_ids = {
        w["gold_id"]
        for w in warnings
        if isinstance(w, dict) and w.get("kind") == "baseline_undecidable"
    }
    if not viable:
        print("No viable fixtures after filtering; aborting.")
        return 1
    print(f"Running with {len(viable)} viable fixtures "
          f"({len(golds) - len(viable)} filtered out).")

    judge_sleep = args.judge_sleep_s if args.with_judge else 0.0
    graders = _build_default_graders(args.with_judge, judge_sleep_s=judge_sleep)
    trials = run_suite(
        golds=viable,
        graders=graders,
        include_controls=True,
        undecidable_gold_ids=undecidable_gold_ids,
    )
    metrics = compute_metrics(trials, [g.name for g in graders])

    print_report(trials, metrics)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "config": {
            "seed": args.seed,
            "with_judge": args.with_judge,
            "judge_sleep_s": judge_sleep,
            "n_fixtures": len(viable),
            "mutation_specs": [
                {k: getattr(s, k) for k in ("kind", "family", "severity", "description")}
                for s in MUTATION_SPECS
            ],
            "graders": [g.name for g in graders],
        },
        "metrics": metrics,
        "trials": [_trial_to_dict(t) for t in trials],
        "warnings": warnings,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved report to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
