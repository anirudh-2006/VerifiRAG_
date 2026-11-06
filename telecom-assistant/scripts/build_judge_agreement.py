#!/usr/bin/env python3
"""Judge–judge and judge–human agreement (Cohen's κ).

Binary coding on committed verdicts only:
  passed → correct, failed → error.
Abstain / unparseable / crash are excluded from that pair (not forced into a
class). Human labels use human_correct True/False.

Reports:
  κ(gpt-oss, claude) on steps where BOTH judges committed
  κ(gpt-oss, human)  on steps where gpt-oss committed
  κ(claude, human)   on steps where claude committed

Plus overlap of caught human errors (do they fail the same way?).

Output: results/judge_agreement.json / .md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import cohen_kappa as _cohen_kappa  # noqa: E402
from src.stats import landis_koch_label as _landis_koch  # noqa: E402
from build_threeway_table import _load, _load_human_labels  # noqa: E402


def _judge_map(path: Path) -> Dict[Tuple[str, int], str]:
    d = _load(path)
    out: Dict[Tuple[str, int], str] = {}
    for qid, steps in d["verdicts"].items():
        for s in steps:
            out[(qid, int(s["step_number"]))] = s["verdict"]
    return out


def _human_binary(label: Dict[str, Any]) -> str:
    """Human label as a binary class, with the third case handled explicitly.

    The other ten sites in this repo read ``human_correct is False`` to mean
    "error", which sends a missing or None label to the NON-error class. This
    file used to read ``is True`` and send it to the ERROR class instead, so
    the same tri-state had two opposite defaults across the codebase. Neither
    default is reachable on the frozen corpus (``main`` now asserts that), but
    they cannot both be right, so the None case is refused rather than
    silently bucketed. This is the shape of the gold1 gate bug: a non-answer
    read as an answer.
    """
    h = label.get("human_correct")
    if h is None:
        raise ValueError(
            "human_correct is missing or None; it must be True or False. "
            "A tri-state label cannot be coerced into a binary class here."
        )
    return "correct" if h is True else "error"


def _to_binary(verdict: str) -> Optional[str]:
    if verdict == "passed":
        return "correct"
    if verdict == "failed":
        return "error"
    return None  # abstain / unparseable / crash


def _agreement_block(
    pairs: List[Tuple[str, str]], name: str
) -> Dict[str, Any]:
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    kappa = _cohen_kappa(pairs)
    cm = Counter(pairs)
    return {
        "name": name,
        "n": n,
        "pct_agreement": agree / n if n else None,
        "kappa": kappa,
        "landis_koch": _landis_koch(kappa),
        "confusion": {f"{a}|{b}": v for (a, b), v in sorted(cm.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("results/label_templates"))
    ap.add_argument("--gptoss", type=Path,
                    default=Path("results/judge_step_verdicts.json"))
    ap.add_argument("--claude", type=Path,
                    default=Path("results/judge_step_verdicts_claude.json"))
    ap.add_argument("--out", type=Path, default=Path("results/judge_agreement.json"))
    ap.add_argument("--out-md", type=Path, default=Path("results/judge_agreement.md"))
    args = ap.parse_args()

    labels = _load_human_labels(args.labels)
    gpt = _judge_map(args.gptoss)
    claude = _judge_map(args.claude)
    keys = sorted(set(labels) & set(gpt) & set(claude))

    # Fail loudly rather than let a missing or None human label be silently
    # bucketed. Added 2026-08-30 with _human_binary; see its docstring.
    bad = [k for k in keys if labels[k].get("human_correct") is None]
    if bad:
        raise SystemExit(
            "human_correct missing or None on %d step(s), e.g. %s. "
            "Every label must be True or False before agreement is computed."
            % (len(bad), bad[:5])
        )

    # Judge–judge: both committed.
    jj_pairs: List[Tuple[str, str]] = []
    for k in keys:
        a, b = _to_binary(gpt[k]), _to_binary(claude[k])
        if a is not None and b is not None:
            jj_pairs.append((a, b))

    # Judge–human: judge committed.
    def jh_pairs(judge: Dict[Tuple[str, int], str]) -> List[Tuple[str, str]]:
        pairs = []
        for k in keys:
            j = _to_binary(judge[k])
            if j is None:
                continue
            human = _human_binary(labels[k])
            pairs.append((j, human))
        return pairs

    gh = jh_pairs(gpt)
    ch = jh_pairs(claude)

    blocks = [
        _agreement_block(jj_pairs, "gpt-oss vs claude"),
        _agreement_block(gh, "gpt-oss vs human"),
        _agreement_block(ch, "claude vs human"),
    ]

    # Among the 13 human errors: who caught what?
    error_keys = [k for k in keys if labels[k].get("human_correct") is False]
    both_catch = gpt_only = claude_only = neither = 0
    per_error = []
    for k in error_keys:
        g_catch = gpt[k] == "failed"
        c_catch = claude[k] == "failed"
        if g_catch and c_catch:
            both_catch += 1
            tag = "both"
        elif g_catch:
            gpt_only += 1
            tag = "gpt-oss_only"
        elif c_catch:
            claude_only += 1
            tag = "claude_only"
        else:
            neither += 1
            tag = "neither"
        per_error.append({
            "id": f"{k[0]}.{k[1]}",
            "error_type": labels[k].get("error_type"),
            "gpt-oss": gpt[k],
            "claude": claude[k],
            "catch": tag,
        })

    payload = {
        "coding": "binary correct/error on committed verdicts; abstain excluded",
        "n_steps": len(keys),
        "n_human_errors": len(error_keys),
        "agreement": {b["name"]: b for b in blocks},
        "error_catch_overlap": {
            "both": both_catch,
            "gpt-oss_only": gpt_only,
            "claude_only": claude_only,
            "neither": neither,
            "per_error": per_error,
        },
    }
    args.out.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Judge agreement (Cohen's κ)",
        "",
        "Binary coding on committed steps (passed=correct, failed=error); "
        "abstain/unparseable excluded.",
        "",
        "| comparison | n | % agree | κ | Landis–Koch |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for b in blocks:
        lines.append(
            f"| {b['name']} | {b['n']} | {100*b['pct_agreement']:.1f}% | "
            f"{b['kappa']:.3f} | {b['landis_koch']} |"
        )
    lines += [
        "",
        "## Human-error catch overlap (n=13)",
        f"- both catch: {both_catch}",
        f"- gpt-oss only: {gpt_only}",
        f"- claude only: {claude_only}",
        f"- neither: {neither}",
        "",
    ]
    md = "\n".join(lines)
    args.out_md.write_text(md)
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.out_md}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
