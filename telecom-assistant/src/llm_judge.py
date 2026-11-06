"""
LLM-as-judge grader for derivation-style RAG answers.

Two entry points:
  - judge_final_answer(generated, expected_spec, question, *, judge=None)
  - judge_step(step, prior_steps, question, *, judge=None)

Both return a dict whose primary keys mirror src.verifier.check_final_answer_match
(for the final-answer case) or the verification step-result shape (for the step
case), plus judge-specific fields::

    {
      "status"     : "match" | "mismatch" | "abstain" | "unparseable",
      "generated"  : str,
      "expected"   : str | None,
      "expected_type": str,
      # judge-specific:
      "confidence" : float in [0.0, 1.0],
      "reasoning"  : str,
      "judge_raw"  : str,            # raw LLM response for debugging
      "judge_model": str,            # which model produced the verdict
      "grader"     : "llm_judge",    # tag so downstream can distinguish graders
    }

The backend is pluggable via the JudgeClient protocol.  The default
`OpenAIJudgeClient` hits the Cerebras endpoint (same one used by src/rag_chain.py).
A `FakeJudge` backend returns canned responses for deterministic tests that
never touch the network.

This module is intentionally decoupled from the symbolic grader: the two are
meant to be compared, not chained.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.load_env import load_project_dotenv

load_project_dotenv()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FINAL_ANSWER_RUBRIC = """You are a strict but fair grader for
signals-and-systems derivations. You will be given a question, a reference
(expected) answer, and a student-generated answer. Decide whether the
student's final answer is mathematically equivalent to the reference.

Rules:
- Two expressions are equivalent if they differ only by algebraic
  simplification, notation (e.g. X(jw) vs X(j*omega), j vs I),
  ordering of commutative operands, or equivalent function calls.
- Accept any of the listed reference alternatives (when multiple are given).
- If the reference specifies parts with aggregation = "all", ALL parts must be
  present in the student answer to call it a MATCH.
- If the reference specifies parts with aggregation = "any", matching ANY one
  part suffices for MATCH.
- If the student answer is empty, pseudocode, or clearly unparseable math,
  use "unparseable" (not "mismatch").
- Use "abstain" only if the student answer is partial and you genuinely
  cannot decide.  Do NOT use "abstain" as a default when one form is clearly
  equivalent to another; commit to "match" or "mismatch" in that case.

Output ONLY a JSON object in this exact shape, no prose before or after:
{
  "verdict":    "match" | "mismatch" | "abstain" | "unparseable",
  "confidence": <float in [0, 1]>,
  "reasoning":  "<one or two sentence justification>"
}
"""

FINAL_ANSWER_TEMPLATE = """{rubric}

QUESTION:
{question}

REFERENCE EXPECTED ANSWER:
{expected_rendered}

STUDENT ANSWER:
{generated}
"""

STEP_RUBRIC = """You are a strict grader for a single step of a math
derivation. You will be given the question being solved, the preceding steps
(as context), and the current step's symbolic expression with the operation
connecting it to the previous step.

Decide whether the current step is mathematically valid given the previous
step — that is, does the stated operation, applied to the previous
expression, produce this expression?

Rules:
- "passed": the step is a valid algebraic, calculus, or definitional move.
- "failed": the step contains a math error (wrong manipulation, wrong limits,
  wrong sign, dropped term, etc.).
- "abstain": the step is too vague, uses undefined symbols, or you cannot
  tell without more context. Prefer "abstain" over guessing.

Output ONLY a JSON object in this exact shape, no prose before or after:
{
  "verdict":    "passed" | "failed" | "abstain",
  "confidence": <float in [0, 1]>,
  "reasoning":  "<one or two sentence justification>"
}
"""

STEP_TEMPLATE = """{rubric}

QUESTION:
{question}

PRIOR STEPS (most recent last):
{prior_steps}

CURRENT STEP:
  expression: {expression}
  operation connecting to previous: {operation}
"""


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class JudgeClient(Protocol):
    """Minimal backend interface the judge needs."""

    model: str

    def score(self, prompt: str) -> str:
        """Return the raw model output for the given prompt."""
        ...


@dataclass
class OpenAIJudgeClient:
    """Default backend — uses a ChatOpenAI-compatible endpoint (Cerebras).

    Kept deliberately thin: no LangChain memory, no chains, just one
    completion call at temperature=0 for determinism across runs.
    """

    # Judge backend is fully swappable via environment variables so a second
    # judge (different model/provider) can be slotted in without code changes:
    #   JUDGE_MODEL        e.g. "gpt-oss-120b", "gpt-4o-mini", "claude-3-5-haiku"
    #   JUDGE_BASE_URL     OpenAI-compatible endpoint
    #   JUDGE_API_KEY_ENV  name of the env var holding that provider's API key
    #   JUDGE_MAX_TOKENS   token budget (reasoning models need headroom)
    #
    # NOTE: the original judge model (qwen-3-235b-a22b-instruct-2507) was
    # deprecated by the provider. The default gpt-oss-120b is an open-weight
    # reasoning model whose internal reasoning tokens count against max_tokens,
    # so the budget is raised to leave room for the JSON verdict.
    model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-oss-120b")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "JUDGE_BASE_URL", "https://api.cerebras.ai/v1"
        )
    )
    api_key_env: str = field(
        default_factory=lambda: os.getenv("JUDGE_API_KEY_ENV", "CEREBRAS_API_KEY")
    )
    temperature: float = 0.0
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("JUDGE_MAX_TOKENS", "2048"))
    )
    # Prevents hung `invoke` when the provider never returns (stuck TCP).
    request_timeout_s: float = 180.0

    def __post_init__(self) -> None:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set; cannot instantiate judge client"
            )
        self._llm = ChatOpenAI(
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout_s,
        )

    def score(self, prompt: str) -> str:
        resp = self._llm.invoke(prompt)
        return getattr(resp, "content", str(resp))


@dataclass
class AnthropicJudgeClient:
    """Second-judge backend — native Anthropic (Claude) via langchain_anthropic.

    Anthropic's API is not natively OpenAI-shaped, so this uses a proper
    ChatAnthropic client rather than the beta OpenAI-compat layer. Same
    ``score(prompt) -> str`` contract as :class:`OpenAIJudgeClient`, so it is a
    drop-in judge for both judge_vs_human and the error-injection harness.

    Swappable via env vars:
        ANTHROPIC_JUDGE_MODEL   e.g. "claude-sonnet-4-5", "claude-3-5-haiku-latest"
        ANTHROPIC_API_KEY_ENV   name of the env var holding the key
        JUDGE_MAX_TOKENS        token budget for the response
    """

    model: str = field(
        default_factory=lambda: os.getenv(
            "ANTHROPIC_JUDGE_MODEL", "claude-sonnet-4-5"
        )
    )
    api_key_env: str = field(
        default_factory=lambda: os.getenv(
            "ANTHROPIC_API_KEY_ENV", "ANTHROPIC_API_KEY"
        )
    )
    temperature: float = 0.0
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("JUDGE_MAX_TOKENS", "2048"))
    )
    request_timeout_s: float = 180.0

    def __post_init__(self) -> None:
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set; cannot instantiate judge client"
            )
        self._llm = ChatAnthropic(
            model=self.model,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout_s,
        )

    def score(self, prompt: str) -> str:
        resp = self._llm.invoke(prompt)
        content = getattr(resp, "content", resp)
        # Anthropic may return a list of content blocks; join any text parts.
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content)


def make_judge_from_env() -> "JudgeClient":
    """Instantiate the judge backend selected by the JUDGE_BACKEND env var.

    JUDGE_BACKEND=openai  (default) -> OpenAIJudgeClient (Cerebras/OpenAI-compat)
    JUDGE_BACKEND=anthropic          -> AnthropicJudgeClient (native Claude)

    This keeps both judge scripts backend-agnostic: a second judge is a matter
    of exporting env vars, no code edits.
    """
    backend = os.getenv("JUDGE_BACKEND", "openai").strip().lower()
    if backend in ("anthropic", "claude"):
        return AnthropicJudgeClient()
    return OpenAIJudgeClient()


@dataclass
class FakeJudge:
    """Stub backend for tests. Returns canned responses keyed by substring
    match over the prompt.

    Iteration order of `responses` is preserved; the first matching substring
    wins.  If none match, `default` is returned (defaults to an "abstain"
    verdict so unhandled branches don't silently pretend to pass).

    Usage:
        fake = FakeJudge(responses=[
            ("1/(s+2)", '{"verdict":"match","confidence":0.95,"reasoning":""}'),
        ])
        judge_final_answer(gen, spec, q, judge=fake)
    """

    responses: List[Tuple[str, str]] = field(default_factory=list)
    default: str = (
        '{"verdict": "abstain", "confidence": 0.5, "reasoning": "fake-default"}'
    )
    model: str = "fake-judge"
    calls: List[str] = field(default_factory=list)

    def score(self, prompt: str) -> str:
        self.calls.append(prompt)
        for substring, response in self.responses:
            if substring in prompt:
                return response
        return self.default


# ---------------------------------------------------------------------------
# Expected-answer rendering
# ---------------------------------------------------------------------------


def _render_expected_for_prompt(expected_spec: Any) -> str:
    """Flatten an expected_answer spec into a readable reference string.

    The judge is given a natural-language rendering rather than JSON because
    our backend reasons better over prose than over raw schemas, and because
    the judge doesn't need to know the internal shape — it just needs a
    clear description of what counts as correct.
    """
    if expected_spec is None:
        return "(no reference answer provided)"
    if isinstance(expected_spec, str):
        return f"Expression: {expected_spec}"
    if not isinstance(expected_spec, dict):
        return str(expected_spec)

    t = expected_spec.get("type")
    if t in {"symbolic", "sympy"}:
        return f"Expression (symbolic): {expected_spec.get('expression', '')}"
    if t == "inequality":
        return f"Inequality: {expected_spec.get('expression', '')}"
    if t == "text":
        return f"Text answer: {expected_spec.get('expression', '')}"
    if t == "composite":
        agg = (expected_spec.get("aggregation") or "all").lower()
        parts = expected_spec.get("parts", []) or []
        header = (
            f"Composite answer (aggregation = {agg}). "
            f"{'ALL' if agg == 'all' else 'AT LEAST ONE'} of the following parts "
            "must be present for a match:"
        )
        lines = [header]
        for i, p in enumerate(parts, start=1):
            pt = p.get("type", "?")
            name = p.get("name", f"part_{i}")
            expr = p.get("expression", "")
            lines.append(f"  - {name} ({pt}): {expr}")
        return "\n".join(lines)
    return json.dumps(expected_spec)


# ---------------------------------------------------------------------------
# Judge-response parsing
# ---------------------------------------------------------------------------

_VALID_FINAL_VERDICTS = {"match", "mismatch", "abstain", "unparseable"}
_VALID_STEP_VERDICTS = {"passed", "failed", "abstain"}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM response.

    Tries three strategies in order:
      1. direct json.loads on the whole string,
      2. code fences (```json ... ``` or plain ``` ... ```),
      3. regex on the first balanced-looking {...} run.
    """
    if not raw:
        return None
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    for fence in ("```json", "```JSON", "```"):
        if fence in stripped:
            start = stripped.find(fence) + len(fence)
            end = stripped.find("```", start)
            if end > start:
                try:
                    return json.loads(stripped[start:end].strip())
                except Exception:
                    pass
    m = _JSON_OBJECT_RE.search(stripped)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _parse_verdict(raw: str, valid_set: set) -> Dict[str, Any]:
    """Parse the judge's raw response into a normalized dict.

    Always returns a dict with verdict / confidence / reasoning / judge_raw.
    Clamps confidence into [0, 1].  Marks out-of-range verdicts as
    "unparseable" but preserves the raw response so downstream code can
    surface why the judge went off-rails.
    """
    obj = _extract_json_object(raw)
    if not isinstance(obj, dict):
        return {
            "verdict": "unparseable",
            "confidence": 0.0,
            "reasoning": "(could not parse judge response as JSON)",
            "judge_raw": raw,
        }
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in valid_set:
        return {
            "verdict": "unparseable",
            "confidence": _safe_float(obj.get("confidence", 0.0)),
            "reasoning": (
                str(obj.get("reasoning", ""))[:500]
                or f"(verdict {verdict!r} not in {sorted(valid_set)})"
            ),
            "judge_raw": raw,
        }
    return {
        "verdict": verdict,
        "confidence": _safe_float(obj.get("confidence", 0.0)),
        "reasoning": str(obj.get("reasoning", ""))[:500],
        "judge_raw": raw,
    }


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_judge() -> JudgeClient:
    return OpenAIJudgeClient()


def judge_final_answer(
    generated: str,
    expected_spec: Any,
    question: str = "",
    *,
    judge: Optional[JudgeClient] = None,
) -> Dict[str, Any]:
    """Grade a final answer with an LLM.

    Returns a dict whose key surface mirrors
    `src.verifier.check_final_answer_match`, plus judge-specific fields
    (confidence, reasoning, judge_raw, judge_model, grader).
    """
    judge = judge or _default_judge()
    expected_rendered = _render_expected_for_prompt(expected_spec)
    prompt = FINAL_ANSWER_TEMPLATE.format(
        rubric=FINAL_ANSWER_RUBRIC,
        question=(question or "").strip() or "(not provided)",
        expected_rendered=expected_rendered,
        generated=generated or "",
    )
    raw = judge.score(prompt)
    parsed = _parse_verdict(raw, _VALID_FINAL_VERDICTS)

    expected_str: Optional[str] = None
    expected_type: str = "legacy_string"
    if isinstance(expected_spec, str):
        expected_str = expected_spec
    elif isinstance(expected_spec, dict):
        expected_str = expected_spec.get("expression") or json.dumps(expected_spec)
        expected_type = str(expected_spec.get("type", "legacy_string"))

    return {
        "status": parsed["verdict"],
        "generated": generated,
        "expected": expected_str,
        "expected_type": expected_type,
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"],
        "judge_raw": parsed["judge_raw"],
        "judge_model": getattr(judge, "model", "unknown"),
        "grader": "llm_judge",
    }


def judge_step(
    step: Dict[str, Any],
    prior_steps: Optional[List[Dict[str, Any]]] = None,
    question: str = "",
    *,
    judge: Optional[JudgeClient] = None,
) -> Dict[str, Any]:
    """Grade a single derivation step with an LLM.

    `step` must have keys `expression` and `operation`.  `prior_steps` is a
    list of earlier step dicts (same shape) — passed so the judge can check
    that the stated operation produces this expression from the previous one.
    """
    judge = judge or _default_judge()
    prior_rendered = (
        "\n".join(
            f"  {i + 1}. {s.get('expression', '')}  [{s.get('operation', '')}]"
            for i, s in enumerate(prior_steps or [])
        )
        or "(none)"
    )
    prompt = STEP_TEMPLATE.format(
        rubric=STEP_RUBRIC,
        question=(question or "").strip() or "(not provided)",
        prior_steps=prior_rendered,
        expression=step.get("expression", ""),
        operation=step.get("operation", ""),
    )
    raw = judge.score(prompt)
    parsed = _parse_verdict(raw, _VALID_STEP_VERDICTS)
    return {
        "verdict": parsed["verdict"],
        "expression": step.get("expression", ""),
        "operation": step.get("operation", ""),
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"],
        "judge_raw": parsed["judge_raw"],
        "judge_model": getattr(judge, "model", "unknown"),
        "grader": "llm_judge",
    }


__all__ = [
    "JudgeClient",
    "OpenAIJudgeClient",
    "FakeJudge",
    "judge_final_answer",
    "judge_step",
]
