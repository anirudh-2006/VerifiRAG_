# main.py
import streamlit as st
import json
from dotenv import load_dotenv
load_dotenv()

from src.rag_chain import ask_structured, ask
from src.verifier import verify_derivation, StepResult

st.set_page_config(
    page_title="Telecom Learning Assistant",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Learning Assistant")
st.caption("RAG + Concept Graph + Symbolic Verification for Signals & Systems")

# sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Answer mode",
        ["Verified derivation", "Quick answer"],
        help="Verified mode runs symbolic checking on each step. Quick mode is faster."
    )
    show_raw_json = st.checkbox("Show raw structured output", value=False)
    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "Derive the Fourier transform of a rectangular pulse",
        "What is the relationship between Fourier and Laplace transforms?",
        "Explain the Z-transform and its region of convergence",
        "Derive the convolution theorem",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# handle prefill from sidebar buttons
default_input = st.session_state.pop("prefill", "")

question = st.chat_input(
    "Ask a signals & systems question...",
    key="chat_input"
) or default_input

if question:
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if mode == "Quick answer":
            # plain RAG answer without verification
            from src.rag_chain import build_rag_chain
            chain = build_rag_chain()
            with st.spinner("Retrieving and generating..."):
                response = chain.invoke(question)
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        else:
            # full verified pipeline
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("Derivation")
                with st.spinner("Running RAG + concept graph retrieval..."):
                    structured = ask_structured(question)

                if structured is None:
                    st.error("Could not generate structured output. Try rephrasing.")
                else:
                    # display each step cleanly
                    steps = structured.get("steps", [])
                    concept = structured.get("concept", "")
                    st.markdown(f"**Concept:** {concept}")
                    st.divider()

                    for step in steps:
                        st.markdown(
                            f"**Step {step['step_number']}** — "
                            f"{step['description']}"
                        )
                        st.code(step['expression'], language="python")

                    st.divider()
                    st.markdown(
                        f"**Final result:** `{structured.get('final_result', '')}`"
                    )

                    if show_raw_json:
                        with st.expander("Raw structured output"):
                            st.json(structured)

            with col2:
                st.subheader("Verification")
                if structured:
                    with st.spinner("Running symbolic verification..."):
                        results = verify_derivation(structured, fast_mode=True)

                    passed  = sum(1 for r in results if r.is_valid is True)
                    failed  = sum(1 for r in results if r.is_valid is False)
                    unclear = sum(1 for r in results if r.is_valid is None)
                    total   = len(results)

                    # summary metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("✓ Passed",  passed)
                    m2.metric("✗ Failed",  failed,
                              delta=f"-{failed}" if failed else None,
                              delta_color="inverse")
                    m3.metric("? Review",  unclear)

                    st.divider()

                    # per-step results
                    for r in results:
                        if r.is_valid is True:
                            icon, color = "✓", "green"
                        elif r.is_valid is False:
                            icon, color = "✗", "red"
                        else:
                            icon, color = "?", "orange"

                        with st.expander(
                            f":{color}[{icon}] Step {r.step_number}: "
                            f"{r.description[:50]}..."
                            if len(r.description) > 50
                            else f":{color}[{icon}] Step {r.step_number}: {r.description}"
                        ):
                            st.code(r.expression_given, language="python")
                            if r.error_message:
                                if r.is_valid is False:
                                    st.error(r.error_message)
                                elif r.is_valid is None:
                                    st.warning(r.error_message)

                    # overall verdict
                    st.divider()
                    if failed > 0:
                        st.error(
                            f"⚠️ {failed} step(s) contain errors. "
                            f"Review flagged steps before trusting this derivation."
                        )
                    elif unclear > 0:
                        st.warning(
                            f"✓ No errors detected. "
                            f"{unclear} step(s) need manual review "
                            f"(semantic transformations)."
                        )
                    else:
                        st.success("✓ All steps verified successfully.")

                    full_response = (
                        f"**{concept}**\n\n" +
                        "\n\n".join(
                            f"**Step {s['step_number']}:** {s['description']}\n"
                            f"`{s['expression']}`"
                            for s in steps
                        ) +
                        f"\n\n**Final result:** `{structured.get('final_result', '')}`"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )