"""
Paired statistical comparison utilities for evaluation results.

This module provides:
  1) robust loading + pairing of per-query rows from eval JSON files
  2) metric extraction for common evaluator metrics
  3) paired bootstrap confidence intervals on mean delta
  4) paired permutation/sign-flip significance test

Design goals:
  - no heavyweight deps (pure stdlib)
  - deterministic via seed
  - reusable for both:
      * plain-vs-graph ablations
      * symbolic-vs-judge comparisons
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool
    getter: Callable[[dict], Optional[float]]


def _get_nested(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _final_status(row: dict) -> str:
    return (
        _get_nested(row, "pipeline.metrics.final_answer_match.status", "?")
        or "?"
    )


def _final_match_binary(row: dict) -> Optional[float]:
    return 1.0 if _final_status(row) == "match" else 0.0


def _final_match_ordinal(row: dict) -> Optional[float]:
    order = {
        "match": 3.0,
        "mismatch": 2.0,
        "unparseable": 1.0,
        "no_expected": 0.0,
        "unsupported_expected_type": 0.0,
        "?": 0.0,
    }
    return order.get(_final_status(row), 0.0)


def _metric_from_path(path: str) -> Callable[[dict], Optional[float]]:
    def _getter(row: dict) -> Optional[float]:
        return _safe_float(_get_nested(row, path))

    return _getter


BUILTIN_METRICS: Dict[str, MetricSpec] = {
    "final_match_binary": MetricSpec(
        name="final_match_binary",
        higher_is_better=True,
        getter=_final_match_binary,
    ),
    "final_match_ordinal": MetricSpec(
        name="final_match_ordinal",
        higher_is_better=True,
        getter=_final_match_ordinal,
    ),
    "parse_rate_pct": MetricSpec(
        name="parse_rate_pct",
        higher_is_better=True,
        getter=_metric_from_path("pipeline.metrics.parse_rate_pct"),
    ),
    "equiv_rate_pct": MetricSpec(
        name="equiv_rate_pct",
        higher_is_better=True,
        getter=_metric_from_path("pipeline.metrics.equiv_rate_pct"),
    ),
    "checked_passed": MetricSpec(
        name="checked_passed",
        higher_is_better=True,
        getter=_metric_from_path("pipeline.metrics.checked_passed"),
    ),
    "checked_failed": MetricSpec(
        name="checked_failed",
        higher_is_better=False,
        getter=_metric_from_path("pipeline.metrics.checked_failed"),
    ),
    "legacy_pass_rate_pct": MetricSpec(
        name="legacy_pass_rate_pct",
        higher_is_better=True,
        getter=_metric_from_path("pipeline.metrics.legacy_pass_rate_pct"),
    ),
}


def metric_spec(metric_name: str, metric_path: Optional[str] = None) -> MetricSpec:
    """
    Resolve a metric.

    - If metric_path is given, build a custom metric from dot path and ignore
      metric_name (except for the display name).
    - Else metric_name must exist in BUILTIN_METRICS.
    """
    if metric_path:
        return MetricSpec(
            name=metric_name or metric_path,
            higher_is_better=True,
            getter=_metric_from_path(metric_path),
        )
    if metric_name not in BUILTIN_METRICS:
        known = ", ".join(sorted(BUILTIN_METRICS.keys()))
        raise KeyError(f"Unknown metric '{metric_name}'. Known: {known}")
    return BUILTIN_METRICS[metric_name]


def load_results_rows(path: Path) -> List[dict]:
    """
    Load an evaluation JSON file (schema-v2 envelope or bare list) and return
    per-query rows.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: expected schema-v2 envelope or list of rows")


def split_rows_by_mode(
    rows: Sequence[dict],
    mode_path: str,
    arm_a: str,
    arm_b: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Split one result list into two arms using a mode field path.
    """
    a: List[dict] = []
    b: List[dict] = []
    for r in rows:
        m = _get_nested(r, mode_path)
        if m == arm_a:
            a.append(r)
        elif m == arm_b:
            b.append(r)
    return a, b


def pair_metric_values(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    metric: MetricSpec,
    id_key: str = "id",
) -> Dict[str, Any]:
    """
    Pair rows by query ID and extract comparable metric values.

    Returns dict with:
      paired_ids, values_a, values_b, deltas, skipped
    where delta is (b - a).
    """
    by_a = {str(r.get(id_key)): r for r in rows_a if r.get(id_key) is not None}
    by_b = {str(r.get(id_key)): r for r in rows_b if r.get(id_key) is not None}
    ids = sorted(set(by_a.keys()) & set(by_b.keys()))

    values_a: List[float] = []
    values_b: List[float] = []
    deltas: List[float] = []
    paired_ids: List[str] = []
    skipped: List[dict] = []

    for qid in ids:
        va = metric.getter(by_a[qid])
        vb = metric.getter(by_b[qid])
        if va is None or vb is None:
            skipped.append({"id": qid, "reason": "metric_missing_or_non_numeric"})
            continue
        paired_ids.append(qid)
        values_a.append(float(va))
        values_b.append(float(vb))
        deltas.append(float(vb) - float(va))

    return {
        "paired_ids": paired_ids,
        "values_a": values_a,
        "values_b": values_b,
        "deltas": deltas,
        "skipped": skipped,
    }


def _quantile(xs: Sequence[float], q: float) -> float:
    """
    Linear interpolation quantile in [0, 1].
    """
    if not xs:
        raise ValueError("quantile on empty list")
    if q <= 0:
        return min(xs)
    if q >= 1:
        return max(xs)
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def paired_bootstrap_mean_delta(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Paired bootstrap on mean delta = mean(b - a).
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have same length")
    n = len(values_a)
    if n == 0:
        raise ValueError("need at least one pair")

    deltas = [b - a for a, b in zip(values_a, values_b)]
    observed = sum(deltas) / n

    rng = random.Random(seed)
    reps: List[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            s += deltas[j]
        reps.append(s / n)

    alpha = 1 - ci
    lo = _quantile(reps, alpha / 2)
    hi = _quantile(reps, 1 - alpha / 2)
    return {
        "n_pairs": n,
        "mean_delta": observed,
        "ci_level": ci,
        "ci_low": lo,
        "ci_high": hi,
    }


def paired_permutation_test_mean_delta(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Two-sided paired permutation test (random sign flip on paired deltas).

    Null hypothesis: E[b - a] = 0.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have same length")
    n = len(values_a)
    if n == 0:
        raise ValueError("need at least one pair")

    deltas = [b - a for a, b in zip(values_a, values_b)]
    observed = sum(deltas) / n
    obs_abs = abs(observed)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        s = 0.0
        for d in deltas:
            if rng.random() < 0.5:
                s += d
            else:
                s -= d
        stat = abs(s / n)
        if stat >= obs_abs:
            extreme += 1

    # add-one smoothing keeps p in (0,1], avoids zero p-values from finite MC
    p = (extreme + 1) / (n_perm + 1)
    return {
        "n_pairs": n,
        "mean_delta": observed,
        "p_value_two_sided": p,
        "n_permutations": n_perm,
    }


def mcnemar_exact_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, Any]:
    """Exact two-sided McNemar test for paired binary (0/1) outcomes.

    Discordant counts:
      b = (# a=1, b=0),  c = (# a=0, b=1).
    Under H0, b ~ Binomial(b+c, 1/2). Uses the exact binomial mid-style
    two-sided p-value: 2 * sum_{i=0}^{min(b,c)} C(n,i) / 2^n, capped at 1.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have same length")
    n = len(values_a)
    if n == 0:
        raise ValueError("need at least one pair")

    b = c = ties_11 = ties_00 = 0
    for a, bb in zip(values_a, values_b):
        a_i = 1 if float(a) >= 0.5 else 0
        b_i = 1 if float(bb) >= 0.5 else 0
        if a_i == 1 and b_i == 0:
            b += 1
        elif a_i == 0 and b_i == 1:
            c += 1
        elif a_i == 1 and b_i == 1:
            ties_11 += 1
        else:
            ties_00 += 1

    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b, c)
        # sum_{i=0}^{k} C(n_disc, i) / 2^{n_disc}
        # compute iteratively to avoid huge factorials
        term = 1.0  # C(n,0)
        total = term
        for i in range(1, k + 1):
            term *= (n_disc - i + 1) / i
            total += term
        p = min(1.0, 2.0 * total / (2.0 ** n_disc))

    return {
        "n_pairs": n,
        "b_a1_b0": b,
        "c_a0_b1": c,
        "n_discordant": n_disc,
        "ties_11": ties_11,
        "ties_00": ties_00,
        "p_value_two_sided": p,
        "test": "mcnemar_exact",
    }


def holm_bonferroni(
    named_pvalues: Sequence[Tuple[str, float]],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Holm–Bonferroni step-down adjustment over a family of tests."""
    m = len(named_pvalues)
    ordered = sorted(
        [(name, float(p), i) for i, (name, p) in enumerate(named_pvalues)],
        key=lambda t: t[1],
    )
    adj: Dict[int, float] = {}
    running = 0.0
    for rank0, (name, p, orig_i) in enumerate(ordered):
        # rank is 1-based
        raw = (m - rank0) * p
        running = max(running, raw)
        adj[orig_i] = min(1.0, running)
    out = []
    for i, (name, p) in enumerate(named_pvalues):
        out.append({
            "name": name,
            "p_raw": float(p),
            "p_holm": adj[i],
            "reject_at_alpha": adj[i] < alpha,
            "alpha": alpha,
            "family_size": m,
        })
    return out


def cohen_kappa(pairs: Sequence[Tuple[Any, Any]]) -> Optional[float]:
    """Cohen's kappa for a list of (rater1_label, rater2_label) pairs.

    Generic two-rater agreement statistic; used both for judge-vs-human /
    judge-vs-judge agreement (reported in the paper) and, if run, for
    human-vs-human inter-annotator agreement.
    """
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({l for p in pairs for l in p}, key=lambda x: str(x))
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    m = [[0] * k for _ in range(k)]
    for a, b in pairs:
        m[idx[a]][idx[b]] += 1

    po = sum(m[i][i] for i in range(k)) / n
    row = [sum(m[i]) for i in range(k)]
    col = [sum(m[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    if pe == 1.0:
        # Perfect chance agreement (only one label used by both) -> undefined;
        # report 1.0 if observed is also perfect, else 0.0.
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def landis_koch_label(kappa: Optional[float]) -> str:
    """Qualitative bucket for a kappa value (Landis & Koch 1977 convention)."""
    if kappa is None:
        return "n/a"
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def compare_paired(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    metric: MetricSpec,
    *,
    id_key: str = "id",
    n_boot: int = 10000,
    n_perm: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    arm_a_name: str = "A",
    arm_b_name: str = "B",
) -> Dict[str, Any]:
    """
    End-to-end paired comparison report.
    """
    paired = pair_metric_values(rows_a, rows_b, metric=metric, id_key=id_key)
    values_a = paired["values_a"]
    values_b = paired["values_b"]
    deltas = paired["deltas"]
    n = len(deltas)
    if n == 0:
        raise ValueError("No paired rows with numeric metric values")

    wins_b = sum(1 for d in deltas if d > 0)
    wins_a = sum(1 for d in deltas if d < 0)
    ties = n - wins_b - wins_a

    boot = paired_bootstrap_mean_delta(
        values_a, values_b, n_boot=n_boot, ci=ci, seed=seed
    )
    perm = paired_permutation_test_mean_delta(
        values_a, values_b, n_perm=n_perm, seed=seed + 1
    )

    # Effect direction text normalized to "better" by metric direction.
    mean_delta = boot["mean_delta"]
    if metric.higher_is_better:
        better_arm = arm_b_name if mean_delta > 0 else arm_a_name if mean_delta < 0 else "tie"
    else:
        # lower is better => negative delta favors B
        better_arm = arm_b_name if mean_delta < 0 else arm_a_name if mean_delta > 0 else "tie"

    return {
        "metric": {
            "name": metric.name,
            "higher_is_better": metric.higher_is_better,
            "delta_definition": f"{arm_b_name} - {arm_a_name}",
        },
        "arms": {"a": arm_a_name, "b": arm_b_name},
        "paired_counts": {
            "n_pairs": n,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "skipped_pairs": len(paired["skipped"]),
        },
        "bootstrap": boot,
        "permutation": perm,
        "mean_values": {
            arm_a_name: sum(values_a) / n,
            arm_b_name: sum(values_b) / n,
        },
        "better_arm_by_mean_delta": better_arm,
        "paired_ids": paired["paired_ids"],
        "skipped": paired["skipped"],
    }
