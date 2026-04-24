"""
Academic City RAG Chatbot - UI (Final Deliverable)
Student: [Your Name] | Index: [Your Index Number]
"""

import os
import sys
from pathlib import Path
import json

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import importlib.util

# Set paths
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

# -------------------------------------------------------------------
# Dynamically load modules from our parts to power the UI
# -------------------------------------------------------------------
@st.cache_resource
def load_rag_pipeline():
    # Load custom DomainAwareRetriever from Part G
    spec_g = importlib.util.spec_from_file_location("innovation", str(ROOT / "part_g" / "innovation.py"))
    mod_g = importlib.util.module_from_spec(spec_g)
    spec_g.loader.exec_module(mod_g)
    DomainAwareRetriever = mod_g.DomainAwareRetriever

    # Load Context Window Manager & Prompts from Part C
    spec_c1 = importlib.util.spec_from_file_location("prompts", str(ROOT / "part_c" / "01_prompt_templates.py"))
    mod_c1 = importlib.util.module_from_spec(spec_c1)
    spec_c1.loader.exec_module(mod_c1)
    T3_HALLUCINATION_GUARD = mod_c1.T3_HALLUCINATION_GUARD

    spec_c2 = importlib.util.spec_from_file_location("cwm", str(ROOT / "part_c" / "02_context_window_manager.py"))
    mod_c2 = importlib.util.module_from_spec(spec_c2)
    spec_c2.loader.exec_module(mod_c2)
    ContextWindowManager = mod_c2.ContextWindowManager

    # Load Pipeline LLM Caller from Part E (since it has call_llm isolated)
    spec_e = importlib.util.spec_from_file_location("eval", str(ROOT / "part_e" / "evaluation.py"))
    mod_e = importlib.util.module_from_spec(spec_e)
    spec_e.loader.exec_module(mod_e)
    call_llm = mod_e.call_llm

    # Load artifacts
    EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
    index = faiss.read_index(str(EMBED_DIR / "faiss.index"))
    with open(EMBED_DIR / "metadata.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Instantiate
    retriever = DomainAwareRetriever(index, meta, model, alpha=0.7)
    cwm = ContextWindowManager(strategy="ranking", max_chars=3000, min_score=0.20, top_k=5)

    return retriever, cwm, T3_HALLUCINATION_GUARD, call_llm


# -------------------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------------------
st.set_page_config(page_title="Academic City AI", page_icon="🎓", layout="wide")

st.title("🎓 Academic City RAG Chatbot")
st.markdown("*A highly grounded AI assistant fetching real data from National Budgets and Election Results.*")

# Handle missing API Key gracefully
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in `.env` file. Please add it to talk to the LLM.")
    st.stop()

retriever, cwm, T3_Prompt, call_llm = load_rag_pipeline()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Pipeline Settings")
    st.markdown("**Part D & G Pipeline Options**")
    
    alpha = st.slider("Hybrid Search Vector Weight (α)", 0.0, 1.0, 0.7, 0.1, help="1.0 = Pure Vector, 0.0 = Pure BM25 Keyword")
    retriever.alpha = alpha

    use_domain_router = st.checkbox("Enable Domain-Aware Router (Part G)", value=True, help="Boosts chunks that match query intent (Election vs Budget)")
    retriever.domain_multiplier = 1.6 if use_domain_router else 1.0

    st.markdown("---")
    st.markdown("Retrieves exact cosine similarity using **FAISS IndexFlatIP** and **BM25**.")
    st.markdown("Uses **Groq llama-3.3-70b-versatile**.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg:
            with st.expander("🔍 View Retrieved Context"):
                for c in msg["chunks"]:
                    boost_tag = "🚀 Boosted (Domain Router)" if c.get("innovation_boost") else ""
                    st.markdown(f"**Source**: `{c['source']}` | **Score**: `{c['hybrid_score']:.3f}` {boost_tag}")
                    st.text(c['text'])

# Input
if prompt := st.chat_input("Ask about the 2025 budget or past election results..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and respond
    with st.chat_message("assistant"):
        with st.spinner("Searching corpus..."):
            # 1. Retrieve
            chunks = retriever.search(prompt, k=5)
            
            # 2. Context Window Management
            context, meta = cwm.build_context(chunks)

            # 3. Prompting
            msgs = T3_Prompt.to_messages(context, prompt)

        with st.spinner("AI is thinking..."):
            # 4. Generate
            llm_response = call_llm(msgs, max_tokens=512)
            final_text = llm_response.get("text", "Error talking to Groq.")

        st.markdown(final_text)

        with st.expander("🔍 View Retrieved Context & Scores"):
            for c in chunks:
                boost_tag = "🚀 Boosted by Domain Router" if c.get("innovation_boost") else ""
                st.markdown(f"**Source**: `{c['source']}` | **Score**: `{c['hybrid_score']:.3f}` {boost_tag}")
                st.text(c['text'])
                st.divider()

    # Append assistant response with chunks
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_text,
        "chunks": chunks
    })
