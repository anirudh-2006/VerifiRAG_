#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.stats import (  # noqa: E402
    compare_paired,
    holm_bonferroni,
    load_results_rows,
    mcnemar_exact_test,
    metric_spec,
    pair_metric_values,
    split_rows_by_mode,
)


def _row(qid: str, mode: str, final: str, parse: float = 0.0) -> dict:
    return {
        "id": qid,
        "ablation": {"pipeline_retrieval_mode": mode},
        "pipeline": {
            "metrics": {
                "final_answer_match": {"status": final},
                "parse_rate_pct": parse,
                "checked_failed": 0,
            }
        },
    }


def test_split_rows_by_mode() -> None:
    rows = [
        _row("Q1", "plain", "mismatch"),
        _row("Q1", "graph", "match"),
        _row("Q2", "plain", "match"),
        _row("Q2", "graph", "match"),
    ]
    a, b = split_rows_by_mode(rows, "ablation.pipeline_retrieval_mode", "plain", "graph")
    assert len(a) == 2 and len(b) == 2


def test_pair_metric_values_final_match_binary() -> None:
    rows_a = [_row("Q1", "a", "mismatch"), _row("Q2", "a", "match")]
    rows_b = [_row("Q1", "b", "match"), _row("Q2", "b", "match")]
    spec = metric_spec("final_match_binary")
    paired = pair_metric_values(rows_a, rows_b, metric=spec)
    assert paired["paired_ids"] == ["Q1", "Q2"]
    assert paired["values_a"] == [0.0, 1.0]
    assert paired["values_b"] == [1.0, 1.0]
    assert paired["deltas"] == [1.0, 0.0]


def test_compare_paired_positive_delta_significant() -> None:
    rows_a = [_row(f"Q{i}", "a", "mismatch") for i in range(1, 21)]
    rows_b = [_row(f"Q{i}", "b", "match") for i in range(1, 21)]
    spec = metric_spec("final_match_binary")
    out = compare_paired(
        rows_a,
        rows_b,
        metric=spec,
        n_boot=2000,
        n_perm=2000,
        seed=7,
        arm_a_name="sym",
        arm_b_name="judge",
    )
    assert out["paired_counts"]["n_pairs"] == 20
    assert out["bootstrap"]["mean_delta"] > 0.95
    assert out["permutation"]["p_value_two_sided"] < 0.02
    assert out["better_arm_by_mean_delta"] == "judge"


def test_compare_paired_negative_delta_with_lower_better_metric() -> None:
    # checked_failed: lower is better. B has consistently fewer failures.
    rows_a = []
    rows_b = []
    for i in range(1, 11):
        ra = _row(f"Q{i}", "a", "match")
        rb = _row(f"Q{i}", "b", "match")
        ra["pipeline"]["metrics"]["checked_failed"] = 4
        rb["pipeline"]["metrics"]["checked_failed"] = 1
        rows_a.append(ra)
        rows_b.append(rb)

    spec = metric_spec("checked_failed")
    out = compare_paired(rows_a, rows_b, metric=spec, n_boot=1000, n_perm=1000, seed=3, arm_a_name="A", arm_b_name="B")
    assert out["bootstrap"]["mean_delta"] < 0.0
    assert out["better_arm_by_mean_delta"] == "B"


def test_load_results_rows_supports_schema_v2_and_list() -> None:
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "env.json"
        p2 = Path(td) / "list.json"
        rows = [_row("Q1", "plain", "match")]
        p1.write_text(json.dumps({"schema_version": 2, "results": rows}))
        p2.write_text(json.dumps(rows))
        assert len(load_results_rows(p1)) == 1
        assert len(load_results_rows(p2)) == 1


def test_mcnemar_exact_all_discordant() -> None:
    a = [0] * 10
    b = [1] * 10
    r = mcnemar_exact_test(a, b)
    assert r["c_a0_b1"] == 10 and r["b_a1_b0"] == 0
    assert r["p_value_two_sided"] == 2 * (0.5 ** 10)  # two-sided: k=0 and k=10


def test_mcnemar_exact_no_discordant() -> None:
    r = mcnemar_exact_test([1, 0, 1], [1, 0, 1])
    assert r["n_discordant"] == 0
    assert r["p_value_two_sided"] == 1.0


def test_holm_bonferroni_family() -> None:
    h = holm_bonferroni([("a", 0.001), ("b", 0.04), ("c", 0.03)], alpha=0.05)
    by = {x["name"]: x for x in h}
    assert by["a"]["reject_at_alpha"] is True
    assert by["a"]["p_holm"] == 0.003
    assert by["b"]["reject_at_alpha"] is False
    assert by["c"]["reject_at_alpha"] is False


def main() -> int:
    tests = [
        test_split_rows_by_mode,
        test_pair_metric_values_final_match_binary,
        test_compare_paired_positive_delta_significant,
        test_compare_paired_negative_delta_with_lower_better_metric,
        test_load_results_rows_supports_schema_v2_and_list,
        test_mcnemar_exact_all_discordant,
        test_mcnemar_exact_no_discordant,
        test_holm_bonferroni_family,
    ]
    ok = 0
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
        ok += 1
    print(f"\n{ok}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
