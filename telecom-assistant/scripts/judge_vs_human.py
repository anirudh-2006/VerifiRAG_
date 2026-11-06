"""
Score the LLM-as-judge against the same human-labeled steps the symbolic
verifier was scored on, and emit a side-by-side grader-vs-human report.

The judge sees one step at a time (with the prior steps as context, the same
way the human labeller saw them in `results/label_templates/Q*.json`) and
must commit to "passed" / "failed" / "abstain".  We map those to the same
True/False/None the symbolic verifier emits, then run the existing
`src.label_scorer.score()` against both, so the two graders are graded on
the *same* covered/abstain split and confusion-matrix definition.

Usage
-----
  # Real Cerebras-backed judge (network required):
  PYTHONPATH=. python scripts/judge_vs_human.py

  # Deterministic FakeJudge backend, useful for plumbing tests:
  PYTHONPATH=. python scripts/judge_vs_human.py --fake

  # Smoke test only the first N steps (across all queries):
  PYTHONPATH=. python scripts/judge_vs_human.py --limit 5

  # Resume a previously interrupted real run (reuses output verdicts):
  PYTHONPATH=. python scripts/judge_vs_human.py --resume

  # Keep going even if a step keeps rate-limiting after retries:
  PYTHONPATH=. python scripts/judge_vs_human.py --continue-on-rate-limit

Outputs
-------
  results/judge_step_verdicts.json
      per-step judge verdicts + reasoning + both confusion matrices.
  results/judge_cache/<sha>.json
      one cached judge response per unique prompt+model.

Cache key is sha256(judge.model + prompt). Changing the model invalidates
the cache automatically; tweaking the rubric or step text invalidates only
the affected entries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.label_scorer import load_labels, score, print_report  # noqa: E402
from src.llm_judge import (  # noqa: E402
    STEP_RUBRIC,
    STEP_TEMPLATE,
    FakeJudge,
    JudgeClient,
    OpenAIJudgeClient,
    judge_step,
    make_judge_from_env,
)

DEFAULT_LABEL_DIR = "results/label_templates"
DEFAULT_OUT = "results/judge_step_verdicts.json"
DEFAULT_CACHE_DIR = "results/judge_cache"


def _is_rate_limit_error(msg: str) -> bool:
    lower = msg.lower()
    return (
        "429" in msg
        or "ratelimit" in msg
        or "queue_exceeded" in msg
        or "rate limit" in lower
        or "too_many_requests" in lower
    )


def _is_transient_connection_error(msg: str) -> bool:
    """Matches proxy hiccups, TLS resets, provider outages wrapped as
    APIConnectionError, etc. — worth exponential backoff like 429."""
    lower = msg.lower()
    return (
        "apiconnectionerror" in lower
        or "connection error" in lower
        or "proxyerror" in lower
        or "proxy error" in lower
        or "403 forbidden" in lower
        or "connecterror" in lower
        or "connection reset" in lower
        or "remote end closed" in lower
        or "timed out" in lower
        or "timeout" in lower
        or "ssl" in lower and "error" in lower
        or "502" in msg
        or "503" in msg
        or "504" in msg
    )


def _is_recoverable_api_error(msg: str) -> bool:
    return _is_rate_limit_error(msg) or _is_transient_connection_error(msg)


def _step_to_judge_dict(template_step: Dict[str, Any]) -> Dict[str, Any]:
    """Map a label-template step into the dict shape `judge_step()` expects.

    The label templates carry `expression_given` and `description`; the judge
    prompt expects `expression` and `operation`. We treat the human-written
    `description` as the operation (it's what the LLM said it was doing).
    """
    return {
        "expression": template_step.get("expression_given", ""),
        "operation":  template_step.get("description", ""),
    }


def _verdict_to_is_valid(verdict: str) -> Optional[bool]:
    if verdict == "passed":
        return True
    if verdict == "failed":
        return False
    return None  # abstain or unparseable


def _build_step_prompt(
    step: Dict[str, Any],
    prior: List[Dict[str, Any]],
    question: str,
) -> str:
    """Reproduce the prompt `judge_step` would build, so we can hash it
    for caching without invoking the LLM."""
    prior_rendered = (
        "\n".join(
            f"  {i + 1}. {p.get('expression', '')}  [{p.get('operation', '')}]"
            for i, p in enumerate(prior)
        )
        or "(none)"
    )
    return STEP_TEMPLATE.format(
        rubric=STEP_RUBRIC,
        question=(question or "").strip() or "(not provided)",
        prior_steps=prior_rendered,
        expression=step.get("expression", ""),
        operation=step.get("operation", ""),
    )


def _hash_prompt(prompt: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x1f")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def _load_cached(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(cache_dir, key)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cached(cache_dir: str, key: str, data: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, key), "w") as f:
        json.dump(data, f, indent=2)


def _judge_step_with_retry(
    current: Dict[str, Any],
    prior: List[Dict[str, Any]],
    question: str,
    judge: JudgeClient,
    *,
    max_attempts: int = 5,
    base_wait_s: float = 30.0,
    cap_wait_s: float = 300.0,
) -> Dict[str, Any]:
    """Wrap `judge_step` with the same exponential-backoff retry the
    evaluator uses for 429s. Non-rate-limit errors are re-raised
    immediately so genuine bugs aren't silently retried away."""
    for attempt in range(max_attempts):
        try:
            return judge_step(current, prior, question, judge=judge)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if _is_recoverable_api_error(msg) and attempt < max_attempts - 1:
                wait = min(base_wait_s * (2 ** attempt), cap_wait_s)
                why = "rate-limited" if _is_rate_limit_error(msg) else "transient API/network error"
                print(f"    {why}; sleeping {wait:.0f}s before retry {attempt + 2}/{max_attempts}")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def grade_all_with_judge(
    labeled: List[Dict[str, Any]],
    judge: JudgeClient,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    sleep_s: float = 0.5,
    limit: Optional[int] = None,
    max_attempts: int = 5,
    base_wait_s: float = 30.0,
    max_wait_s: float = 300.0,
    continue_on_rate_limit: bool = False,
    resume_verdicts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    checkpoint_path: Optional[str] = None,
    log: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """For each labeled step, run the judge and return verdicts keyed by qid.

    Each query's verdict list preserves step order. Cached responses are
    surfaced via the per-step `cached: true` flag so users can tell which
    calls were live.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    existing_by_qid_and_step: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    if resume_verdicts:
        for qid, vs in resume_verdicts.items():
            out[qid] = list(vs)
            existing_by_qid_and_step[qid] = {
                v.get("step_number"): v for v in vs if v.get("step_number") is not None
            }

    counter = 0
    for q in labeled:
        qid = q.get("id", "?")
        question = q.get("question", "")
        steps = q.get("steps", []) or []
        verdicts: List[Dict[str, Any]] = out.get(qid, [])
        prior: List[Dict[str, Any]] = []
        for s in steps:
            if limit is not None and counter >= limit:
                break
            counter += 1

            current = _step_to_judge_dict(s)
            step_number = s.get("step_number")

            existing = existing_by_qid_and_step.get(qid, {}).get(step_number)
            if existing is not None:
                if log:
                    conf = existing.get("confidence", 0.0) or 0.0
                    print(
                        f"  [resume] {qid} step {step_number:>2}: "
                        f"{existing.get('verdict'):<11} ({conf:.2f})"
                    )
                prior.append(current)
                continue

            prompt = _build_step_prompt(current, prior, question)
            key = _hash_prompt(prompt, getattr(judge, "model", "unknown"))

            cached = _load_cached(cache_dir, key) if use_cache else None
            if cached is not None:
                result = cached
                from_cache = True
            else:
                try:
                    result = _judge_step_with_retry(
                        current,
                        prior,
                        question,
                        judge,
                        max_attempts=max_attempts,
                        base_wait_s=base_wait_s,
                        cap_wait_s=max_wait_s,
                    )
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if continue_on_rate_limit and _is_recoverable_api_error(msg):
                        reason = (
                            "rate-limited after max retries"
                            if _is_rate_limit_error(msg)
                            else "transient API/network error after max retries"
                        )
                        result = {
                            "verdict": "abstain",
                            "confidence": 0.0,
                            "reasoning": reason,
                            "judge_raw": msg[:1000],
                            "judge_model": getattr(judge, "model", "unknown"),
                            "grader": "llm_judge",
                        }
                        print(f"    step {qid}:{step_number} marked abstain ({reason})")
                    else:
                        raise
                from_cache = False
                if use_cache:
                    _save_cached(cache_dir, key, result)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            entry = {
                "step_number": step_number,
                "expression":  current["expression"],
                "operation":   current["operation"],
                "verdict":     result.get("verdict"),
                "confidence":  result.get("confidence"),
                "reasoning":   result.get("reasoning"),
                "judge_raw":   result.get("judge_raw"),
                "cached":      from_cache,
            }
            verdicts.append(entry)
            existing_by_qid_and_step.setdefault(qid, {})[step_number] = entry
            if log:
                tag = "[cache]" if from_cache else "[live] "
                conf = result.get("confidence", 0.0) or 0.0
                print(
                    f"  {tag} {qid} step {s.get('step_number'):>2}: "
                    f"{result.get('verdict'):<11} ({conf:.2f})"
                )

            if checkpoint_path:
                try:
                    snap = dict(out)
                    snap[qid] = verdicts
                    with open(checkpoint_path, "w") as cf:
                        json.dump({"verdicts": snap, "partial": True}, cf, indent=2)
                except OSError:
                    pass

            prior.append(current)
        out[qid] = verdicts
        if limit is not None and counter >= limit:
            break
    return out


def merge_judge_verdicts(
    labeled: List[Dict[str, Any]],
    judge_verdicts: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Return a deep-ish copy of `labeled` with each step augmented with
    `judge_is_valid` + `judge_verdict` + `judge_confidence`.

    The original `verifier_is_valid` is preserved untouched, so the same
    list can be scored against either grader by passing the right
    `valid_key` to `score()`.
    """
    augmented: List[Dict[str, Any]] = []
    for q in labeled:
        qid = q.get("id", "?")
        verdict_by_step = {
            v["step_number"]: v
            for v in judge_verdicts.get(qid, [])
            if v.get("step_number") is not None
        }
        new_q = {**q, "steps": []}
        for s in q.get("steps", []):
            v = verdict_by_step.get(s.get("step_number"))
            new_s = dict(s)
            if v is not None:
                new_s["judge_is_valid"]   = _verdict_to_is_valid(v["verdict"])
                new_s["judge_verdict"]    = v["verdict"]
                new_s["judge_confidence"] = v["confidence"]
            else:
                new_s["judge_is_valid"]   = None
                new_s["judge_verdict"]    = "absent"
                new_s["judge_confidence"] = 0.0
            new_q["steps"].append(new_s)
        augmented.append(new_q)
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score LLM-as-judge against human labels.",
    )
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between live judge calls.")
    parser.add_argument("--fake", action="store_true",
                        help="Use FakeJudge backend (no network).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N steps total (smoke testing).")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max attempts per step on judge call failures.")
    parser.add_argument("--base-wait-s", type=float, default=30.0,
                        help="Initial wait in seconds for exponential backoff.")
    parser.add_argument("--max-wait-s", type=float, default=300.0,
                        help="Cap wait in seconds for exponential backoff.")
    parser.add_argument("--continue-on-rate-limit", action="store_true",
                        help="Mark abstain if retries exhaust on 429 or transient "
                        "network/proxy/connection errors (not only rate limits).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --out or --checkpoint if present (tries both).")
    parser.add_argument(
        "--checkpoint",
        default="results/judge_step_verdicts.partial.json",
        help="Write partial verdicts after each step (for interrupted runs).",
    )
    args = parser.parse_args()

    labeled = load_labels(args.label_dir)
    if not labeled:
        print(f"No filled label files found in {args.label_dir}.")
        return
    n_steps = sum(len(q.get("steps", [])) for q in labeled)
    print(f"Loaded {len(labeled)} labeled queries ({n_steps} steps).")

    if args.fake:
        judge: JudgeClient = FakeJudge(default=(
            '{"verdict": "abstain", "confidence": 0.5, '
            '"reasoning": "fake-default"}'
        ))
        print("Using FakeJudge backend (no network).")
    else:
        judge = make_judge_from_env()
        print(f"Using judge backend: {type(judge).__name__} model: {judge.model}")

    resume_verdicts: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if args.resume:
        for resume_path in (args.out, args.checkpoint):
            if not os.path.isfile(resume_path):
                continue
            try:
                with open(resume_path) as f:
                    existing = json.load(f)
                rv = existing.get("verdicts", {})
                if isinstance(rv, dict) and rv:
                    resume_verdicts = rv
                    recovered = sum(
                        len(vs) for vs in rv.values() if isinstance(vs, list)
                    )
                    print(
                        f"Resuming from {resume_path}: found {recovered} "
                        "saved step verdicts."
                    )
                    break
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not load resume file {resume_path}: {e}")

    print()
    print("Grading steps with judge...")
    verdicts = grade_all_with_judge(
        labeled, judge,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        sleep_s=args.sleep,
        limit=args.limit,
        max_attempts=max(1, int(args.max_attempts)),
        base_wait_s=max(0.0, float(args.base_wait_s)),
        max_wait_s=max(0.0, float(args.max_wait_s)),
        continue_on_rate_limit=args.continue_on_rate_limit,
        resume_verdicts=resume_verdicts,
        checkpoint_path=args.checkpoint,
    )

    augmented = merge_judge_verdicts(labeled, verdicts)

    print()
    verifier_report = score(
        labeled,
        valid_key="verifier_is_valid",
        grader_label="verifier",
    )
    print_report(verifier_report)

    print()
    judge_report = score(
        augmented,
        valid_key="judge_is_valid",
        grader_label="judge",
    )
    print_report(judge_report)

    print()
    print("=" * 72)
    print("HEAD-TO-HEAD (covered steps where BOTH graders committed)")
    print("=" * 72)
    head_to_head = _head_to_head(augmented)
    for k, v in head_to_head.items():
        print(f"  {k:<35} {v}")

    out = {
        "config": {
            "label_dir": args.label_dir,
            "judge_model": getattr(judge, "model", "unknown"),
            "fake": args.fake,
            "limit": args.limit,
            "max_attempts": args.max_attempts,
            "base_wait_s": args.base_wait_s,
            "max_wait_s": args.max_wait_s,
            "continue_on_rate_limit": args.continue_on_rate_limit,
            "resume": args.resume,
        },
        "verdicts": verdicts,
        "scores": {
            "verifier_vs_human": verifier_report,
            "judge_vs_human":    judge_report,
            "head_to_head":      head_to_head,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved combined report to: {args.out}")


def _head_to_head(augmented: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts on the subset of steps where both verifier and judge committed
    to True or False. Missed-by-verifier-but-caught-by-judge is the headline
    cell — that's what motivates the judge in the paper."""
    both_correct = both_wrong_human = 0
    only_judge_caught = only_verifier_caught = 0
    judge_caught_verifier_missed = 0
    judge_missed_verifier_caught = 0
    judge_only_committed = verifier_only_committed = 0
    neither_committed = 0
    for q in augmented:
        for s in q.get("steps", []):
            v = s.get("verifier_is_valid")
            j = s.get("judge_is_valid")
            h = s.get("human_correct")
            if h is None:
                continue
            v_committed = v is not None
            j_committed = j is not None
            if not v_committed and not j_committed:
                neither_committed += 1
                continue
            if v_committed and not j_committed:
                verifier_only_committed += 1
                continue
            if j_committed and not v_committed:
                judge_only_committed += 1
                if h is False and j is False:
                    judge_caught_verifier_missed += 1
                continue
            # both committed
            v_correct_call = (v is True and h is True) or (v is False and h is False)
            j_correct_call = (j is True and h is True) or (j is False and h is False)
            if v_correct_call and j_correct_call:
                both_correct += 1
            if not v_correct_call and not j_correct_call:
                both_wrong_human += 1
            if h is False:
                if v is True and j is False:
                    only_judge_caught += 1
                if v is False and j is True:
                    only_verifier_caught += 1
            if h is False and v is True and j is False:
                judge_caught_verifier_missed += 1
            if h is False and v is False and j is True:
                judge_missed_verifier_caught += 1
    return {
        "both_committed_both_correct":      both_correct,
        "both_committed_both_wrong":        both_wrong_human,
        "judge_caught_error_verifier_missed":   judge_caught_verifier_missed,
        "verifier_caught_error_judge_missed":   judge_missed_verifier_caught,
        "verifier_only_committed":          verifier_only_committed,
        "judge_only_committed":             judge_only_committed,
        "neither_committed":                neither_committed,
    }


if __name__ == "__main__":
    main()
