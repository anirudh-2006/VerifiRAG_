"""Phase 3: retrieval-augmented answers with an LLM."""

# src/rag_chain.py
# src/rag_chain.py  — full updated version
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from src.concept_graph import build_telecom_graph, identify_concepts, get_prerequisites

VECTORSTORE_DIR = "vectorstore"

SYSTEM_PROMPT = """You are an expert tutor for telecom and electrical engineering students.

The following context is extracted from course materials. Use it as your primary reference.
You may fill in standard mathematical steps implied by the context but not explicitly shown.
If the context contains a result or formula, use it as a starting point and show the full derivation.
Never invent formulas that contradict the context.

When answering derivation or math questions:
1. State what you are deriving and define all variables first
2. Show every step on its own line
3. Briefly explain what you did at each step in plain English
4. Clearly label the final result

Context from course materials:
{context}

Student question: {question}
"""

STRUCTURED_PROMPT = """You are an expert tutor for telecom and electrical engineering students.

Use the provided context as your primary reference.
You may fill in standard mathematical steps implied by the context.
Never invent formulas that contradict the context.

Return your answer as a JSON object with this EXACT structure — no extra text, no markdown fences:
{{
  "concept": "name of what is being derived",
  "variables": {{"variable_name": "what it represents"}},
  "steps": [
    {{
      "step_number": 1,
      "description": "plain English explanation of what this step does",
      "expression": "the mathematical expression at this step as a Python SymPy string",
      "operation": "what operation connects previous step to this one"
    }}
  ],
  "final_result": "the final expression(s) as a Python SymPy string"
}}

Rules for the "final_result" field:
- If the question asks for ONE thing (a transform, a formula, a value), write ONE expression.
- If the question asks for MULTIPLE things (e.g. "compute the transform AND state the ROC",
  or "give the transform pair"), write ALL of them as a single comma-separated string
  with the parts in the order asked. Use top-level commas only; never put the separator
  inside parentheses. Do NOT label the parts.
  Example (transform + ROC):  "1/(s+2), re(s) > -2"
  Example (pair):              "exp(-a*t)*u(t), 1/(s+a)"
- Do NOT prefix with an assignment (write "1/(s+2)", not "X(s) = 1/(s+2)").

Rules for the "expression" field (STRICT — these strings must parse with sympy.parsing.sympy_parser.parse_expr):
- Pure mathematical expression only. NO Python keywords (if/else/for/in/where).
- NO pseudo-code like `.exists`, `.contains`, `ROC`, `X(s)|s=jw`, `name='dtft'`.
- NO assignment on the left — write the expression itself.
- Use ** for exponents (not ^). Use * for all multiplication.
- Use these function names: sin, cos, exp, integrate, Sum, Piecewise, sqrt, log, Abs.
  (Do NOT prefix with sympy. — write `exp(x)`, not `sympy.exp(x)`.)
- Use these symbol names only: t, omega, s, z, T, n, k, m, a, r, sigma, Omega, I.
- For complex unit use I (capital I).
- For piecewise functions use Piecewise((val1, cond1), (val2, cond2), ...).
  Example: Piecewise((1, Abs(t) < T/2), (0, True))
- For evaluation at a point, use subs via rewriting.
  Instead of: "X(s)|s=jomega", write: "X(I*omega)" or expand the substituted form.

GOOD examples (do this):
  "2*sin(omega)/omega"
  "exp(-s*t0)*X"
  "Sum(x_n * exp(-I*Omega*n), (n, -oo, oo))"
  "Piecewise((1, Abs(t) < T/2), (0, True))"
  "integrate(exp(-s*t), (t, 0, oo))"

BAD examples (never do this):
  "X(s)|s=jomega"                  (use X(I*omega) instead)
  "0 if k%2==0 else 2/(pi*k)"      (use Piecewise)
  "sympy.And(X.exists, Re(s) > 0)" (not verifiable — omit or put in description)
  "for -T/4 < t < T/4"             (use Piecewise)
  "X_dtft = Sum(...)"              (drop the LHS assignment)

Context:
{context}

Student question: {question}

Output under 12 steps total. Be concise. Return ONLY the JSON object, no markdown fences."""

# add this function to src/rag_chain.py

import json
from typing import Any, Optional, Dict

DEFAULT_RETRIEVAL_CONFIG = {
    "plain_similarity_k": 4,
    "graph_prerequisite_keyword_k": 2,
    "graph_prerequisite_depth": 2,
    "graph_enabled": True,
    "graph_max_prereqs": None,
    "graph_max_extra_chunks": None,
}


def _merge_retrieval_config(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_RETRIEVAL_CONFIG)
    if isinstance(overrides, dict):
        cfg.update(overrides)
    return cfg


def ask_structured(
    question: str,
    vectorstore=None,
    G=None,
    retrieval_mode: str = "graph",
    retrieval_config: Optional[Dict[str, Any]] = None,
):
    """Returns both the readable answer and structured JSON for verification.

    vectorstore and G can be passed in to avoid re-initialising on every call
    (e.g. from the batch evaluator).
    """
    if vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings
        )
    if G is None:
        G = build_telecom_graph()
    r_cfg = _merge_retrieval_config(retrieval_config)
    mode = (retrieval_mode or "graph").lower()

    llm = ChatOpenAI(
        model="qwen-3-235b-a22b-instruct-2507",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
        temperature=0,
        max_tokens=2500,
    )

    def graph_enhanced_retrieve(query: str) -> str:
        base_k = int(r_cfg.get("plain_similarity_k", 4))
        results = vectorstore.similarity_search(query, k=base_k)

        if mode == "plain" or not bool(r_cfg.get("graph_enabled", True)):
            return "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source','?').split('/')[-1]}, "
                f"Page {doc.metadata.get('page','?')}]\n{doc.page_content}"
                for doc in results
            )

        matched_concepts = identify_concepts(query, G)
        ordered_prereqs = []
        seen_prereqs = set()
        depth = int(r_cfg.get("graph_prerequisite_depth", 2))
        max_prereqs = r_cfg.get("graph_max_prereqs")
        for concept in matched_concepts:
            prereqs = get_prerequisites(G, concept, depth=depth)
            for p in prereqs:
                if p not in seen_prereqs:
                    seen_prereqs.add(p)
                    ordered_prereqs.append(p)
        if isinstance(max_prereqs, int) and max_prereqs >= 0:
            ordered_prereqs = ordered_prereqs[:max_prereqs]

        seen = set(doc.page_content for doc in results)
        extra_chunks = 0
        per_prereq_k = int(r_cfg.get("graph_prerequisite_keyword_k", 2))
        max_extra_chunks = r_cfg.get("graph_max_extra_chunks")
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
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source','?').split('/')[-1]}, "
            f"Page {doc.metadata.get('page','?')}]\n{doc.page_content}"
            for doc in results
        )

    prompt = ChatPromptTemplate.from_template(STRUCTURED_PROMPT)
    chain = (
        {"context": RunnableLambda(graph_enhanced_retrieve), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke(question)

    # strip markdown fences if the model added them anyway
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]

    try:
        structured = json.loads(clean.strip())
        return structured
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        print(f"Raw output was:\n{raw}")
        return None

def build_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    G = build_telecom_graph()

    llm = ChatOpenAI(
        model="qwen-3-235b-a22b-instruct-2507",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
        temperature=0,
        max_tokens=1500,
    )

    def graph_enhanced_retrieve(query: str) -> str:
        """
        1. Identify which concepts the query is about
        2. Find their prerequisites via the concept graph
        3. Retrieve chunks for query + all prerequisite keywords
        4. Return everything as one combined context string
        """
        # Step 1: find matched concepts and their prerequisites
        matched_concepts = identify_concepts(query, G)
        all_prereqs = set()
        for concept in matched_concepts:
            prereqs = get_prerequisites(G, concept, depth=2)
            all_prereqs.update(prereqs)

        print(f"\n[Graph] Query matched concepts: {matched_concepts}")
        print(f"[Graph] Adding prerequisite context for: {list(all_prereqs)}")

        # Step 2: retrieve chunks for the original query
        results = vectorstore.similarity_search(query, k=4)
        seen_contents = set(doc.page_content for doc in results)

        # Step 3: retrieve chunks for each prerequisite concept
        for prereq in all_prereqs:
            prereq_keywords = G.nodes[prereq].get("keywords", [])
            if prereq_keywords:
                # use the first keyword as the search query
                prereq_query = prereq_keywords[0]
                prereq_results = vectorstore.similarity_search(prereq_query, k=2)
                for doc in prereq_results:
                    if doc.page_content not in seen_contents:
                        results.append(doc)
                        seen_contents.add(doc.page_content)

        print(f"[Graph] Total chunks retrieved: {len(results)}")

        # Step 4: format all chunks into one context string
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source','?').split('/')[-1]}, "
            f"Page {doc.metadata.get('page','?')}]\n{doc.page_content}"
            for doc in results
        )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {"context": RunnableLambda(graph_enhanced_retrieve), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask(question: str):
    chain = build_rag_chain()
    print(f"\nQuestion: {question}\n")
    print("Answer:\n")
    response = chain.invoke(question)  # invoke instead of stream
    print(response)

if __name__ == "__main__":
    ask("What is the relationship between the Laplace transform and the Fourier transform?")