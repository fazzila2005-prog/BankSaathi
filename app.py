"""BankSaathi: a bank customer-rights assistant built on a Nugen aligned model.

The model is aligned on RBI consumer banking rules (fraud liability, Ombudsman,
failed-transaction compensation, KYC, BSBD accounts, Charter of Customer Rights).

Run:
    export NUGEN_API_KEY="..."
    streamlit run app.py

The model ID is read from state.json (written by nugen_pipeline.py) or can be
pasted in the sidebar. Turn on "Compare with base model" to send the same
question to a second model side by side, which is a quick way to show what the
alignment changed.
"""

import json
import os
from pathlib import Path

import streamlit as st

from nugen_client import chat
from rag import RAG_INSTRUCTIONS, Retriever, build_context

st.set_page_config(page_title="BankSaathi", page_icon="🏦", layout="wide")

state = json.loads(Path("state.json").read_text(encoding="utf-8")) if Path("state.json").exists() else {}

with st.sidebar:
    st.header("Settings")
    if not os.environ.get("NUGEN_API_KEY"):
        key = st.text_input("Nugen API key", type="password")
        if key:
            os.environ["NUGEN_API_KEY"] = key
    model_id = st.text_input("Aligned model ID", value=state.get("model_id", ""))
    system_prompt = st.text_area(
        "System prompt",
        "You are BankSaathi, a helpful assistant that explains Reserve Bank of India rules "
        "on bank customers' rights in simple language. Give the exact numbers (days, rupee "
        "limits) from the RBI rules, say what the customer should do next, and if you are "
        "not sure, say so and suggest contacting the bank or the RBI helpline 14448. "
        "You give general information, not legal advice.",
    )
    use_rag = st.toggle("Look up RBI rules (RAG)", value=True,
                        help="Finds the matching RBI document and gives it to the model with the question.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    compare = st.toggle("Compare with base model")
    base_model = st.text_input("Base model ID", "meta-llama/Llama-3.2-3B-Instruct") if compare else None
    if st.button("Clear chat"):
        st.session_state.history = []

@st.cache_resource
def get_retriever():
    return Retriever("docs")


st.title("🏦 BankSaathi")
st.caption("Know your rights as a bank customer in India. Powered by a Nugen model aligned on RBI rules. "
           "General information only, not legal advice.")

EXAMPLES = [
    "Someone used my card online without permission. What should I do and will I lose money?",
    "ATM debited my account but gave no cash. When do I get it back?",
    "My bank ignored my complaint for 5 weeks. Can I go to the RBI Ombudsman?",
    "Do I need to visit the branch to update my KYC?",
]

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about fraud, failed payments, complaints, KYC...")
if not st.session_state.history and not question:
    st.markdown("**Try asking:**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}"):
            question = ex
if question:
    if not (os.environ.get("NUGEN_API_KEY") and model_id):
        st.error("Add your API key and aligned model ID in the sidebar first.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    system = system_prompt
    sources = []
    if use_rag:
        sources = get_retriever().search_docs(question)
        system += "\n\n" + RAG_INSTRUCTIONS + build_context(sources)
    messages = [{"role": "system", "content": system}] + st.session_state.history

    with st.chat_message("assistant"):
        if compare:
            left, right = st.columns(2)
            left.markdown("**Aligned model**")
            answer = left.write_stream(chat(model_id, messages, temperature=temperature))
            right.markdown("**Base model**")
            try:
                right.write_stream(chat(base_model, messages, temperature=temperature))
            except Exception as e:  # base model may not be served for inference
                right.warning(f"Base model call failed: {e}")
        else:
            answer = st.write_stream(chat(model_id, messages, temperature=temperature))
        if sources:
            with st.expander("📄 Sources used"):
                for d in sources:
                    st.markdown(f"- **{d['doc']}** (best match: *{d['heading']}*)")

    st.session_state.history.append({"role": "assistant", "content": answer})
