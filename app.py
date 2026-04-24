"""
GovLens AI - UI (Final Deliverable)
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

    spec_c1 = importlib.util.spec_from_file_location("prompts", str(ROOT / "part_c" / "01_prompt_templates.py"))
    mod_c1 = importlib.util.module_from_spec(spec_c1)
    spec_c1.loader.exec_module(mod_c1)
    T3_HALLUCINATION_GUARD = mod_c1.T3_HALLUCINATION_GUARD

    spec_c2 = importlib.util.spec_from_file_location("cwm", str(ROOT / "part_c" / "02_context_window_manager.py"))
    mod_c2 = importlib.util.module_from_spec(spec_c2)
    spec_c2.loader.exec_module(mod_c2)
    ContextWindowManager = mod_c2.ContextWindowManager

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

    retriever = DomainAwareRetriever(index, meta, model, alpha=0.7)
    cwm = ContextWindowManager(strategy="ranking", max_chars=3000, min_score=0.20, top_k=5)

    return retriever, cwm, T3_HALLUCINATION_GUARD, call_llm


# -------------------------------------------------------------------
# Streamlit Interface & Custom CSS (Gold & Blue Theme)
# -------------------------------------------------------------------
favicon_path = str(ROOT / "logo.png") if os.path.exists(str(ROOT / "logo.png")) else "🏛️"
st.set_page_config(page_title="GovLens AI", page_icon=favicon_path, layout="wide")

st.markdown("""
<style>
    /* Main Background Pattern - very light blue grey */
    .stApp {
        background-color: #f4f7f9;
        color: #0c2340;
    }
    
    /* Elegant Header Container */
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 20px 0;
        border-bottom: 1px solid #d0dae5;
        margin-bottom: 40px;
    }
    .header-icon {
        background-color: #e5eef7;
        color: #0c2340;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        border: 2px solid #0c2340;
    }
    .header-title {
        color: #0c2340;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .header-subtitle {
        color: #637b96;
        font-size: 15px;
        margin: 0;
        padding: 0;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 50px;
    }
    .hero-icon {
        font-size: 50px;
        color: #0c2340;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 24px;
        font-weight: 600;
        color: #0c2340;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        color: #637b96;
        font-size: 16px;
    }

    /* Cards */
    .stCard {
        background-color: #ffffff;
        border: 1px solid #d0dae5;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        display: flex;
        align-items: center;
        font-weight: 500;
        color: #0c2340;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stCard:hover {
        border-color: #d4af37; /* Gold border on hover */
        box-shadow: 0 6px 12px -2px rgba(212, 175, 55, 0.3);
        transform: translateY(-2px);
    }

    /* Chat Messages styling */
    .stChatMessage {
        background-color: transparent !important;
        color: #0c2340 !important; /* Force text color to be visible */
    }
    
    /* Input Area - Gold Accent */
    .stChatInputContainer {
        border: 1px solid #d0dae5 !important;
        border-radius: 8px !important;
    }
    .stChatInputContainer:focus-within {
        border: 2px solid #d4af37 !important; /* Gold focus */
        box-shadow: 0 0 0 1px #d4af37 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ffffff;
        color: #0c2340;
        border-radius: 6px;
        border: 1px solid #d0dae5;
        width: 100%;
        height: 100%;
        text-align: left;
        padding: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        white-space: normal;
    }
    .stButton>button:hover {
        border-color: #d4af37; /* Gold border on hover */
        box-shadow: 0 6px 12px -2px rgba(212, 175, 55, 0.3);
        transform: translateY(-2px);
        color: #0c2340;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        color: #0c2340 !important;
        background-color: #ffffff !important;
        border: 1px solid #d0dae5 !important;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# (Header moved to Hero Section below)
# -------------------------------------------------------------------

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in `.env` file. Please add it to talk to the LLM.")
    st.stop()

retriever, cwm, T3_Prompt, call_llm = load_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("<h2 style='color:#0c2340;'>Pipeline Settings</h2>", unsafe_allow_html=True)
    st.markdown("**Part D & G Pipeline Options**")
    
    alpha = st.slider("Hybrid Search Vector Weight (α)", 0.0, 1.0, 0.7, 0.1, help="1.0 = Pure Vector, 0.0 = Pure BM25 Keyword")
    retriever.alpha = alpha

    use_domain_router = st.checkbox("Enable Domain-Aware Router (Part G)", value=True, help="Boosts chunks that match query intent (Election vs Budget)")
    retriever.domain_multiplier = 1.6 if use_domain_router else 1.0

    st.markdown("---")
    st.markdown("Retrieves exact cosine similarity using **FAISS IndexFlatIP** and **BM25**.")
    st.markdown("Uses **Groq llama-3.3-70b-versatile**.")


# -------------------------------------------------------------------
# Helper to load local logo
# -------------------------------------------------------------------
def get_base64_image(image_path):
    import base64
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None

# Try direct path first, then ROOT path
logo_path = "logo.png"
if not os.path.exists(logo_path):
    logo_path = str(ROOT / "logo.png")

logo_b64 = get_base64_image(logo_path)
logo_img_tag = f"<img src='data:image/png;base64,{logo_b64}' width='140' style='margin-bottom: 0px;'>" if logo_b64 else "<div style='font-size: 60px; margin-bottom: 0px;'>🏛️ ⚖️</div>"
small_logo_tag = f"<img src='data:image/png;base64,{logo_b64}' width='35' height='35' style='margin-right: 12px; border-radius: 4px; object-fit: contain;'>" if logo_b64 else "<div style='font-size: 24px; margin-right: 12px;'>🏛️</div>"

bot_avatar = logo_path if os.path.exists(logo_path) else "🏛️"

# Display initial hero section if no messages
if len(st.session_state.messages) == 0:
    st.markdown(
        f"<div class='hero-container' style='margin-top: 0px; margin-bottom: 20px;'>"
        f"{logo_img_tag}"
        f"<h1 style='color: #0c2340; font-size: 32px; font-weight: 800; margin-top: 5px; margin-bottom: 0px;'>GovLens AI</h1>"
        f"<p style='color: #637b96; font-size: 16px; margin-bottom: 15px;'>Budget & Election Information Assistant</p>"
        f"<h2 class='hero-title' style='margin-top: 10px; font-size: 20px;'>Ask about government budgets & elections</h2>"
        f"<p class='hero-subtitle' style='margin-bottom: 10px;'>Get data-driven answers backed by official sources</p>"
        f"</div>", 
        unsafe_allow_html=True
    )

    # 4 Suggested Questions as clickable buttons
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.button("What is the total national budget projection for 2025?")
        q2 = st.button("What percentage of votes did NPP win in the 2020 election?")
    with col2:
        q3 = st.button("Show me the election results for the 2016 presidential election")
        q4 = st.button("How has primary expenditure as a percentage of GDP changed?")
    
    # If a button is clicked, set it as the prompt
    if q1: st.session_state.button_clicked = "What is the total national budget projection for 2025?"
    if q2: st.session_state.button_clicked = "What percentage of votes did NPP win in the 2020 election?"
    if q3: st.session_state.button_clicked = "Show me the election results for the 2016 presidential election"
    if q4: st.session_state.button_clicked = "How has primary expenditure as a percentage of GDP changed?"
else:
    # Small rectangular box above active chat
    st.markdown(
        f"<div style='display: flex; align-items: center; background-color: #ffffff; border: 1px solid #d0dae5; border-radius: 8px; padding: 10px 15px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>"
        f"{small_logo_tag}"
        f"<h3 style='color: #0c2340; margin: 0; font-weight: 700; font-size: 20px;'>GovLens AI</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

# Display Chat History
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(f"**GovLens AI**")
            st.markdown(msg["content"])
            if "chunks" in msg:
                with st.expander("🔍 View Retrieved Context"):
                    for c in msg["chunks"]:
                        boost_tag = "🚀 Boosted (Domain Router)" if c.get("innovation_boost") else ""
                        st.markdown(f"**Source**: `{c['source']}` | **Score**: `{c['hybrid_score']:.3f}` {boost_tag}")
                        st.text(c['text'])
            
            # Feedback Loop UI
            if "feedback" not in msg:
                st.markdown("<p style='font-size: 14px; color: #637b96; margin-bottom: 5px; margin-top: 15px;'>Was this answer helpful?</p>", unsafe_allow_html=True)
                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    if st.button("Yes", key=f"yes_{i}"):
                        msg["feedback"] = "yes"
                        st.rerun()
                with col2:
                    if st.button("No", key=f"no_{i}"):
                        msg["feedback"] = "no"
                        st.rerun()
            else:
                st.markdown("<p style='font-size: 14px; color: #d4af37; font-weight: 500; margin-top: 15px;'>Thank you for your feedback!</p>", unsafe_allow_html=True)
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You**")
            st.markdown(msg["content"])


# Input area
prompt = st.chat_input("Ask about budgets, elections, or policy data...")

# Check if a button was clicked or text was entered
if hasattr(st.session_state, 'button_clicked') and st.session_state.button_clicked:
    prompt = st.session_state.button_clicked
    del st.session_state.button_clicked

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Process new user input if it exists
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant", avatar=bot_avatar):
        with st.spinner("Searching corpus..."):
            chunks = retriever.search(user_prompt, k=5)
            context, meta = cwm.build_context(chunks)
            
        if not context.strip():
            # Short-circuit logic: chunks were below the 0.20 score threshold
            final_text = "No matching evidence found in the dataset. I only respond based on verified retrieved information."
        else:
            msgs = T3_Prompt.to_messages(context, user_prompt)
            with st.spinner("AI is determining answer..."):
                llm_response = call_llm(msgs, max_tokens=512)
                final_text = llm_response.get("text", "Error talking to Groq.")
                
            # Fallback if the LLM hallucination guard stops the generation
            if "INSUFFICIENT" in final_text.upper():
                final_text = "No matching evidence found in the dataset. I only respond based on verified retrieved information."

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
    
