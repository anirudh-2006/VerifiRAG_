print("Starting evaluator...", flush=True)

# src/evaluator.py
import json
import time
import os
import copy
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.concept_graph import build_telecom_graph, identify_concepts, get_prerequisites
from src.rag_chain import ask_structured, build_rag_chain
from src.run_metadata import (
    load_eval_run_config,
    make_eval_envelope,
    new_run_block,
    project_root,
)
from src.verifier import (
    verify_derivation,
    categorize_step,
    check_final_answer_match,
)

RESULTS_DIR = "results"
LABEL_TEMPLATES_DIR = "results/label_templates"
DEFAULT_RETRIEVAL_CFG = {
    "plain_similarity_k": 4,
    "graph_prerequisite_keyword_k": 2,
    "graph_prerequisite_depth": 2,
    "graph_enabled": True,
    "graph_max_prereqs": None,
    "graph_max_extra_chunks": None,
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def resolve_expected_answer_spec(query_item: dict) -> dict:
    """
    Return a structured expected-answer spec.

    New format:
      query_item["expected_answer"] = {"type": "...", "expression"/"value": "..."}

    Legacy format:
      query_item["expected_final_result"] = "..."
    """
    if isinstance(query_item.get("expected_answer"), dict):
        return query_item["expected_answer"]
    legacy = query_item.get("expected_final_result", "")
    return {"type": "symbolic", "expression": legacy}


def expected_answer_to_text(expected_spec: dict) -> str:
    if (expected_spec.get("type") or "").lower() == "composite":
        parts = expected_spec.get("parts") or []
        rendered = []
        for p in parts:
            name = p.get("name", "part")
            value = p.get("expression") or p.get("value") or p.get("expected") or ""
            rendered.append(f"{name}: {value}")
        return "; ".join(rendered)
    return (
        expected_spec.get("expression")
        or expected_spec.get("value")
        or expected_spec.get("expected")
        or ""
    )

def get_retrieval_config(eval_config: Optional[dict]) -> dict:
    cfg = dict(DEFAULT_RETRIEVAL_CFG)
    if isinstance(eval_config, dict):
        retrieval = eval_config.get("retrieval", {})
        if isinstance(retrieval, dict):
            cfg.update(retrieval)
    return cfg


def get_shared_resources(eval_config: Optional[dict] = None):
    """Load embeddings, vectorstore, and graph once — reuse across all queries."""
    print("Loading shared resources...")
    embedding_model = "all-MiniLM-L6-v2"
    vectorstore_dir = "vectorstore"
    if isinstance(eval_config, dict):
        embedding_model = (
            eval_config.get("embedding", {}).get("langchain_hf_model_name")
            or embedding_model
        )
        vectorstore_dir = (
            eval_config.get("ingest", {}).get("vectorstore_dir")
            or vectorstore_dir
        )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings
    )
    G = build_telecom_graph()
    print("Resources loaded.\n")
    return embeddings, vectorstore, G

def plain_rag_retrieve(query: str, vectorstore, retrieval_cfg: Optional[dict] = None) -> list:
    """Retrieve chunks using plain similarity search — NO concept graph."""
    k = int((retrieval_cfg or {}).get("plain_similarity_k", 4))
    return vectorstore.similarity_search(query, k=k)

def graph_rag_retrieve(query: str, vectorstore, G, retrieval_cfg: Optional[dict] = None) -> list:
    """Retrieve chunks using concept graph augmented retrieval."""
    cfg = dict(DEFAULT_RETRIEVAL_CFG)
    if isinstance(retrieval_cfg, dict):
        cfg.update(retrieval_cfg)

    base_k = int(cfg.get("plain_similarity_k", 4))
    results = vectorstore.similarity_search(query, k=base_k)
    if not bool(cfg.get("graph_enabled", True)):
        return results, [], []

    matched = identify_concepts(query, G)
    ordered_prereqs = []
    seen_prereqs = set()
    depth = int(cfg.get("graph_prerequisite_depth", 2))
    for concept in matched:
        prereqs = get_prerequisites(G, concept, depth=depth)
        for p in prereqs:
            if p not in seen_prereqs:
                seen_prereqs.add(p)
                ordered_prereqs.append(p)
    max_prereqs = cfg.get("graph_max_prereqs")
    if isinstance(max_prereqs, int) and max_prereqs >= 0:
        ordered_prereqs = ordered_prereqs[:max_prereqs]

    seen = set(doc.page_content for doc in results)

    per_prereq_k = int(cfg.get("graph_prerequisite_keyword_k", 2))
    max_extra_chunks = cfg.get("graph_max_extra_chunks")
    extra_chunks = 0
    for prereq in ordered_prereqs:
        keywords = G.nodes[prereq].get("keywords", [])
        if keywords:
            for doc in vectorstore.similarity_search(keywords[0], k=per_prereq_k):
                if doc.page_content not in seen:
                    results.append(doc)
                    seen.add(doc.page_content)
                    extra_chunks += 1
                    if isinstance(max_extra_chunks, int) and max_extra_chunks >= 0:
                        if extra_chunks >= max_extra_chunks:
                            break
        if isinstance(max_extra_chunks, int) and max_extra_chunks >= 0:
            if extra_chunks >= max_extra_chunks:
                break

    return results, matched, ordered_prereqs

# ─── single query evaluation ──────────────────────────────────────────────────

def evaluate_single_query(query_item: dict, vectorstore, G, eval_config: Optional[dict] = None) -> dict:
    """
    Run one query through both systems and record all metrics.
    Returns a dict of results for this query.
    """
    qid      = query_item["id"]
    question = query_item["question"]
    print(f"\n{'─'*60}")
    print(f"[{qid}] {question}")
    print(f"{'─'*60}")

    result = {
        "id":         qid,
        "question":   question,
        "difficulty": query_item["difficulty"],
        "timestamp":  datetime.now().isoformat(),
    }

    # ── System A: plain RAG ──────────────────────────────────────────────────
    retrieval_cfg = get_retrieval_config(eval_config)
    eval_opts = (eval_config or {}).get("evaluation", {}) if isinstance(eval_config, dict) else {}
    pipeline_mode = os.environ.get(
        "TELECOM_PIPELINE_RETRIEVAL_MODE",
        eval_opts.get("pipeline_retrieval_mode", "graph"),
    ).lower()
    print("  Running System A (plain RAG)...")
    t0 = time.time()
    plain_chunks = plain_rag_retrieve(question, vectorstore, retrieval_cfg)
    plain_time = time.time() - t0

    result["system_a"] = {
        "chunks_retrieved":  len(plain_chunks),
        "retrieval_time_s":  round(plain_time, 2),
        "chunk_sources":     list({
            c.metadata.get("source","?").split("/")[-1]
            for c in plain_chunks
        }),
    }
    print(f"  System A: {len(plain_chunks)} chunks in {plain_time:.2f}s")

    # ── System B: RAG + concept graph ────────────────────────────────────────
    print("  Running System B (RAG + concept graph)...")
    t0 = time.time()
    graph_chunks, matched_concepts, prereqs = graph_rag_retrieve(
        question, vectorstore, G, retrieval_cfg
    )
    graph_time = time.time() - t0

    result["system_b"] = {
        "chunks_retrieved":    len(graph_chunks),
        "retrieval_time_s":    round(graph_time, 2),
        "matched_concepts":    matched_concepts,
        "prerequisites_added": prereqs,
        "extra_chunks":        len(graph_chunks) - len(plain_chunks),
        "chunk_sources":       list({
            c.metadata.get("source","?").split("/")[-1]
            for c in graph_chunks
        }),
    }
    print(f"  System B: {len(graph_chunks)} chunks "
          f"(+{result['system_b']['extra_chunks']} from graph) "
          f"in {graph_time:.2f}s")
    print(f"  Concepts matched: {matched_concepts}")
    print(f"  Prerequisites:    {prereqs}")

    # ── Full pipeline: structured answer + verification ───────────────────────
    print(f"  Running full pipeline (structured answer + verification, mode={pipeline_mode})...")
    t0 = time.time()
    structured = None
    max_attempts = int(eval_opts.get("structured_output_max_attempts", 5))
    rate_limit_base_wait_s = int(eval_opts.get("rate_limit_base_wait_s", 30))
    for attempt in range(max_attempts):
        try:
            structured = ask_structured(
                question,
                vectorstore=vectorstore,
                G=G,
                retrieval_mode=pipeline_mode,
                retrieval_config=retrieval_cfg,
            )
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = ('429' in msg) or ('RateLimit' in msg) or ('queue_exceeded' in msg)
            if is_rate_limit and attempt < (max_attempts - 1):
                # Exponential backoff: 30s, 60s, 120s, 240s, ... (capped at 300s).
                wait = min(rate_limit_base_wait_s * (2 ** attempt), 300)
                print(f"  Rate limited — waiting {wait}s before retry {attempt+2}/{max_attempts}...")
                time.sleep(wait)
            else:
                raise
    answer_time = time.time() - t0

    if structured is None:
        print("  ✗ Structured output failed")
        result["pipeline"] = {"error": "structured output failed"}
        return result

    steps       = structured.get("steps", [])
    final_result = structured.get("final_result", "")
    print(f"  Generated {len(steps)} steps in {answer_time:.2f}s")
    print(f"  Final result: {final_result}")

    # ── Verification ──────────────────────────────────────────────────────────
    print("  Running verification...")
    t0 = time.time()
    verifier_fast_mode = bool(eval_opts.get("verifier_fast_mode", True))
    ver_results = verify_derivation(structured, fast_mode=verifier_fast_mode)
    ver_time = time.time() - t0

    # ── Separated metrics ─────────────────────────────────────────────────────
    # Buckets are mutually exclusive and together cover every step. We report
    # rates over well-defined denominators instead of a single conflated Pass%.
    categories = [categorize_step(r) for r in ver_results]
    counts = {c: categories.count(c) for c in set(categories)}
    total = len(ver_results)

    parsed_count = sum(1 for r in ver_results if r.expression_parsed is not None)
    parse_rate = round(parsed_count / total * 100, 1) if total > 0 else 0.0

    # Sequential equivalence denominator: only steps that were actually checked.
    # (Exclude first_step, parse_fail, natural_lang, roc, integral_skip.)
    checked_total = sum(1 for r in ver_results if r.equivalence_checked)
    checked_passed = sum(
        1 for r in ver_results if r.equivalence_checked and r.is_valid is True
    )
    checked_failed = sum(
        1 for r in ver_results if r.equivalence_checked and r.is_valid is False
    )
    equiv_rate = (
        round(checked_passed / checked_total * 100, 1)
        if checked_total > 0 else None
    )

    # Final answer match against expected answer target (structured or legacy).
    expected_answer = resolve_expected_answer_spec(query_item)
    expected_final_text = expected_answer_to_text(expected_answer)
    final_match = check_final_answer_match(final_result, expected_answer)

    # Legacy passed/failed/unclear for backwards compatibility with older viewers.
    passed_legacy  = sum(1 for r in ver_results if r.is_valid is True)
    failed_legacy  = sum(1 for r in ver_results if r.is_valid is False)
    unclear_legacy = sum(1 for r in ver_results if r.is_valid is None)
    legacy_pass_rate = (
        round(passed_legacy / total * 100, 1) if total > 0 else 0.0
    )

    print(f"  Verification: parse_rate={parse_rate}%  "
          f"equiv_rate={equiv_rate}%  "
          f"final_match={final_match['status']}  "
          f"confirmed_errors={checked_failed}  — {ver_time:.2f}s")

    # Per-step dump for human labelling.
    per_step = []
    for r, cat in zip(ver_results, categories):
        per_step.append({
            "step_number":         r.step_number,
            "description":         r.description,
            "expression_given":    r.expression_given,
            "expression_parsed":   str(r.expression_parsed)
                                    if r.expression_parsed is not None else None,
            "previous_parsed":     str(r.previous_parsed)
                                    if r.previous_parsed is not None else None,
            "equivalence_checked": r.equivalence_checked,
            "is_valid":            r.is_valid,
            "category":            cat,
            "error_message":       r.error_message,
        })

    result["pipeline"] = {
        "steps_generated":    total,
        "answer_time_s":      round(answer_time, 2),
        "retrieval_mode":     pipeline_mode,
        "final_result":       final_result,
        "expected_result":    expected_final_text,  # legacy flat field
        "expected_answer":    expected_answer,      # new structured field
        "metrics": {
            "parse_rate_pct":          parse_rate,
            "parsed_count":            parsed_count,
            "total_steps":             total,
            "checked_total":           checked_total,
            "checked_passed":          checked_passed,
            "checked_failed":          checked_failed,
            "equiv_rate_pct":          equiv_rate,
            "final_answer_match":      final_match,
            "category_counts":         counts,
            "legacy_pass_rate_pct":    legacy_pass_rate,
        },
        "verification": {
            "passed":          passed_legacy,
            "failed":          failed_legacy,
            "unclear":         unclear_legacy,
            "total":           total,
            "pass_rate_pct":   legacy_pass_rate,
            "time_s":          round(ver_time, 2),
            "errors_found":    [
                {"step": r.step_number, "message": r.error_message}
                for r in ver_results if r.is_valid is False
            ],
            "per_step":        per_step,
        },
    }

    _write_label_template(qid, question, expected_final_text, structured, per_step)

    return result


def _write_label_template(qid, question, expected_final, structured, per_step):
    """
    Write a labelling template for this query so a human can tick correctness
    step-by-step. If a filled template already exists, do not overwrite it.
    """
    os.makedirs(LABEL_TEMPLATES_DIR, exist_ok=True)
    path = os.path.join(LABEL_TEMPLATES_DIR, f"{qid}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
            if existing.get("_filled"):
                return  # never clobber a human-reviewed file
        except Exception:
            pass

    template = {
        "id":                qid,
        "question":          question,
        "expected_final":    expected_final,
        "generated_final":   structured.get("final_result", ""),
        "_filled":           False,
        "_instructions": (
            "For each step set 'human_correct' to true/false. "
            "Set 'error_type' to one of: algebra, sign, missing_term, "
            "substitution, definition, none. "
            "Set '_filled' to true when done. "
            "Set 'final_answer_correct' to true/false after comparing "
            "'generated_final' to 'expected_final'."
        ),
        "final_answer_correct": None,
        "steps": [
            {
                "step_number":       s["step_number"],
                "description":       s["description"],
                "expression_given":  s["expression_given"],
                "verifier_category": s["category"],
                "verifier_is_valid": s["is_valid"],
                "human_correct":     None,
                "error_type":        None,
                "notes":             "",
            }
            for s in per_step
        ],
    }
    with open(path, "w") as f:
        json.dump(template, f, indent=2)

# ─── full evaluation run ──────────────────────────────────────────────────────

def run_evaluation(queries: list, output_file: str = None):
    """Run all queries and save results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{RESULTS_DIR}/eval_{timestamp}.json"

    root = project_root()
    cfg_path = os.environ.get("TELECOM_EVAL_CONFIG")
    eval_config = load_eval_run_config(root, cfg_path)
    if eval_config.get("_error"):
        print(f"Warning: {eval_config['_error']}", flush=True)
    run_block = new_run_block(root=root, eval_config=eval_config)

    retrieval_cfg = get_retrieval_config(eval_config)
    embeddings, vectorstore, G = get_shared_resources(eval_config)

    all_results = []
    failed_queries = []

    print(f"\nRunning evaluation on {len(queries)} queries...")
    print(f"Results will be saved to: {output_file}")
    print(f"Retrieval config: {retrieval_cfg}")
    print(
        f"Reproducibility: schema v2 envelope (metadata + config snapshot). "
        f"Override config with TELECOM_EVAL_CONFIG=/path/to.json\n",
        flush=True,
    )

    for i, query_item in enumerate(queries):
        print(f"\nProgress: {i+1}/{len(queries)}")
        try:
            result = evaluate_single_query(query_item, vectorstore, G, eval_config=eval_config)
            all_results.append(result)

            # save after every query so you don't lose progress
            envelope = make_eval_envelope(run_block, all_results)
            with open(output_file, "w") as f:
                json.dump(envelope, f, indent=2)
            sleep_s = int(eval_config.get("evaluation", {}).get("sleep_between_queries_s", 45))
            time.sleep(sleep_s)

        except Exception as e:
            print(f"  ✗ Query {query_item['id']} failed: {e}")
            failed_queries.append(query_item["id"])

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(all_results)
    print(f"\nFull results saved to: {output_file}")
    if failed_queries:
        print(f"Failed queries: {failed_queries}")

    return all_results


def run_ablation(queries: list, output_file: str = None):
    """
    Run each query with two retrieval modes (`plain`, `graph`) so we can
    directly compare retrieval impact while keeping the rest fixed.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{RESULTS_DIR}/eval_ablation_{timestamp}.json"

    root = project_root()
    cfg_path = os.environ.get("TELECOM_EVAL_CONFIG")
    base_config = load_eval_run_config(root, cfg_path)
    if base_config.get("_error"):
        print(f"Warning: {base_config['_error']}", flush=True)

    run_block = new_run_block(root=root, eval_config=base_config)
    embeddings, vectorstore, G = get_shared_resources(base_config)
    all_results = []

    print(f"\nRunning retrieval ablation on {len(queries)} queries...")
    print(f"Output: {output_file}\n")
    eval_cfg = base_config.get("evaluation", {})
    sleep_s = int(eval_cfg.get("sleep_between_queries_s", 45))
    # Give the LLM provider a moment to cool down if another eval run just
    # finished. Prevents a 429 on the very first ablation query that would
    # otherwise inflate the plain-mode time delta.
    initial_cooldown_s = int(eval_cfg.get("initial_cooldown_s", 0))
    if initial_cooldown_s > 0:
        print(f"Initial cooldown: sleeping {initial_cooldown_s}s before first query...")
        time.sleep(initial_cooldown_s)
    total_pairs = 2 * len(queries)
    pair_idx = 0
    failed: list[dict] = []
    for mode in ("plain", "graph"):
        print(f"\n=== Ablation mode: {mode} ===")
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["pipeline_retrieval_mode"] = mode
        for i, query_item in enumerate(queries):
            pair_idx += 1
            print(f"\nProgress ({mode}): {i+1}/{len(queries)}")
            try:
                result = evaluate_single_query(query_item, vectorstore, G, eval_config=cfg)
                result["ablation"] = {"pipeline_retrieval_mode": mode}
                all_results.append(result)
                envelope = make_eval_envelope(run_block, all_results)
                with open(output_file, "w") as f:
                    json.dump(envelope, f, indent=2)
            except Exception as e:
                print(f"  ✗ Query {query_item['id']} ({mode}) failed: {e}")
                failed.append({"id": query_item["id"], "mode": mode, "error": str(e)})
            # Keep LLM-call cadence identical across modes so time deltas are
            # comparable and we don't trip the provider's rate limiter. Skip
            # sleep on the very last call to save wall-clock time.
            if sleep_s > 0 and pair_idx < total_pairs:
                time.sleep(sleep_s)

    print_summary(all_results)
    print_ablation_summary(all_results)
    print(f"\nAblation results saved to: {output_file}")
    if failed:
        print(f"Failed queries ({len(failed)}): "
              + ", ".join(f"{f['id']}/{f['mode']}" for f in failed))
    return all_results


def _final_match_score(status: str) -> int:
    """
    Ordinal score for final-answer status so we can compute directionality.
    Higher is better.
    """
    order = {
        "match": 3,
        "mismatch": 2,
        "unparseable": 1,
        "no_expected": 0,
        "unsupported_expected_type": 0,
        "?": 0,
    }
    return order.get(status or "?", 0)


def print_ablation_summary(results: list):
    """
    Print a compact plain-vs-graph comparison for ablation runs.
    Expects each row to include:
      result["id"]
      result["ablation"]["pipeline_retrieval_mode"] in {"plain","graph"}
      result["pipeline"]["metrics"]
    """
    grouped = {}
    for r in results:
        mode = (r.get("ablation", {}) or {}).get("pipeline_retrieval_mode")
        if mode not in {"plain", "graph"}:
            continue
        grouped.setdefault(r["id"], {})[mode] = r

    if not grouped:
        return

    print(f"\n{'='*110}")
    print("ABLATION DELTA SUMMARY (graph vs plain)")
    print(f"{'='*110}")
    print(
        f"{'ID':<5} {'Final':<18} {'ParseΔ(pp)':<11} {'EquivΔ(pp)':<11} "
        f"{'ErrΔ':<7} {'TimeΔ(s)':<9} {'Verdict':<10}"
    )
    print(f"{'─'*110}")

    wins = losses = ties = 0
    missing_pairs = 0
    parse_deltas = []
    equiv_deltas = []
    err_deltas = []
    time_deltas = []

    for qid in sorted(grouped.keys()):
        pair = grouped[qid]
        plain = pair.get("plain")
        graph = pair.get("graph")
        if plain is None or graph is None:
            missing_pairs += 1
            continue

        pm = plain.get("pipeline", {}).get("metrics", {})
        gm = graph.get("pipeline", {}).get("metrics", {})

        p_final = (pm.get("final_answer_match", {}) or {}).get("status", "?")
        g_final = (gm.get("final_answer_match", {}) or {}).get("status", "?")

        p_parse = pm.get("parse_rate_pct", 0.0) or 0.0
        g_parse = gm.get("parse_rate_pct", 0.0) or 0.0
        parse_delta = round(g_parse - p_parse, 1)

        p_equiv = pm.get("equiv_rate_pct")
        g_equiv = gm.get("equiv_rate_pct")
        if p_equiv is None or g_equiv is None:
            equiv_delta = None
        else:
            equiv_delta = round(float(g_equiv) - float(p_equiv), 1)

        p_err = int(pm.get("checked_failed", 0) or 0)
        g_err = int(gm.get("checked_failed", 0) or 0)
        err_delta = g_err - p_err

        p_time = float(plain.get("pipeline", {}).get("answer_time_s", 0.0) or 0.0)
        g_time = float(graph.get("pipeline", {}).get("answer_time_s", 0.0) or 0.0)
        time_delta = round(g_time - p_time, 2)

        p_score = _final_match_score(p_final)
        g_score = _final_match_score(g_final)
        if g_score > p_score:
            verdict = "graph_win"
            wins += 1
        elif g_score < p_score:
            verdict = "plain_win"
            losses += 1
        else:
            verdict = "tie"
            ties += 1

        parse_deltas.append(parse_delta)
        if equiv_delta is not None:
            equiv_deltas.append(equiv_delta)
        err_deltas.append(err_delta)
        time_deltas.append(time_delta)

        equiv_str = f"{equiv_delta:+.1f}" if equiv_delta is not None else "n/a"
        print(
            f"{qid:<5} {f'{p_final}->{g_final}':<18} {parse_delta:+.1f}{'':<4} "
            f"{equiv_str:<11} {err_delta:+d}{'':<4} {time_delta:+.2f}{'':<3} {verdict:<10}"
        )

    print(f"{'─'*110}")
    if parse_deltas:
        avg_parse_delta = round(sum(parse_deltas) / len(parse_deltas), 2)
        avg_err_delta = round(sum(err_deltas) / len(err_deltas), 2)
        avg_time_delta = round(sum(time_deltas) / len(time_deltas), 2)
        avg_equiv_delta = (
            round(sum(equiv_deltas) / len(equiv_deltas), 2)
            if equiv_deltas else None
        )
        avg_equiv_str = f"{avg_equiv_delta:+.2f}" if avg_equiv_delta is not None else "n/a"
        print(
            f"Avg   {'':<18} {avg_parse_delta:+.2f}{'':<3} {avg_equiv_str:<11} "
            f"{avg_err_delta:+.2f}{'':<1} {avg_time_delta:+.2f}{'':<2} "
            f"wins={wins}, losses={losses}, ties={ties}"
        )
    if missing_pairs:
        print(f"Note: skipped {missing_pairs} query IDs without both plain+graph rows.")
    print(f"{'='*110}\n")

def rerun_failed(query_ids: list):
    """Re-run only specific query IDs."""
    from data.eval_queries import EVAL_QUERIES
    subset = [q for q in EVAL_QUERIES if q["id"] in query_ids]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RESULTS_DIR}/eval_rerun_{timestamp}.json"
    run_evaluation(subset, output_file)

def print_summary(results: list):
    """Print a readable summary table of all results with separated metrics."""
    print(f"\n{'='*98}")
    print("EVALUATION SUMMARY")
    print(f"{'='*98}")
    print(f"{'ID':<5} {'Difficulty':<12} {'Steps':<6} "
          f"{'Parse%':<8} {'Equiv%':<8} {'Checked':<8} "
          f"{'Errors':<8} {'FinalMatch':<12}")
    print(f"{'─'*98}")

    parse_rates = []
    equiv_rates = []
    final_match_counts = {"match": 0, "mismatch": 0,
                          "unparseable": 0, "no_expected": 0}
    confirmed_errors = 0

    for r in results:
        pipe = r.get("pipeline", {})
        if "error" in pipe:
            print(f"{r['id']:<5} {r['difficulty']:<12} — PIPELINE FAILED")
            continue

        m = pipe.get("metrics", {})
        steps     = m.get("total_steps", 0)
        parse_pct = m.get("parse_rate_pct", 0)
        equiv_pct = m.get("equiv_rate_pct")
        checked   = m.get("checked_total", 0)
        errors    = m.get("checked_failed", 0)
        final     = m.get("final_answer_match", {}).get("status", "?")

        parse_rates.append(parse_pct)
        if equiv_pct is not None:
            equiv_rates.append(equiv_pct)
        final_match_counts[final] = final_match_counts.get(final, 0) + 1
        confirmed_errors += errors

        equiv_str = f"{equiv_pct}" if equiv_pct is not None else "n/a"
        print(f"{r['id']:<5} {r['difficulty']:<12} {steps:<6} "
              f"{parse_pct:<8} {equiv_str:<8} {checked:<8} "
              f"{errors:<8} {final:<12}")

    print(f"{'─'*98}")
    if parse_rates:
        avg_parse = round(sum(parse_rates) / len(parse_rates), 1)
    else:
        avg_parse = 0
    if equiv_rates:
        avg_equiv = round(sum(equiv_rates) / len(equiv_rates), 1)
    else:
        avg_equiv = None
    avg_equiv_str = f"{avg_equiv}" if avg_equiv is not None else "n/a"
    print(f"{'Avg':<5} {'':<12} {'':<6} "
          f"{avg_parse:<8} {avg_equiv_str:<8} {'':<8} "
          f"{confirmed_errors:<8}")
    print(f"\nFinal-answer match breakdown: {final_match_counts}")
    print(f"Confirmed math errors (verifier-flagged):  {confirmed_errors}")
    print(f"{'='*98}\n")
    print(
        "Interpretation:\n"
        "  Parse%  : fraction of step-expressions that parsed into SymPy\n"
        "  Equiv%  : of steps we actually checked, how many matched the prior step\n"
        "  Errors  : steps the verifier confirmed as mathematically wrong\n"
        "  Final   : match / mismatch / unparseable / no_expected\n"
    )


if __name__ == "__main__":
    import sys
    from data.eval_queries import EVAL_QUERIES

    # CLI:
    #   python -m src.evaluator all         (default)
    #   python -m src.evaluator rerun Q03 Q08 ...
    #   python -m src.evaluator ablate
    #   python -m src.evaluator ablate Q03 Q08
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == "rerun":
        rerun_failed(args[1:])
    elif len(args) >= 1 and args[0] == "ablate":
        if len(args) > 1:
            subset = [q for q in EVAL_QUERIES if q["id"] in args[1:]]
        else:
            subset = EVAL_QUERIES
        run_ablation(subset)
    else:
        run_evaluation(EVAL_QUERIES)


