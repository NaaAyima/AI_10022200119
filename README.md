# GovLens: A Retrieval-Augmented Generation Chatbot

**Student Name:** Jacqueline Naa Ayima Mensah  
**Index Number:** 10022200119  
**Course:** CS4241 Introduction to Artificial Intelligence 2026

---

## 📝 Description
GovLens is a specialized Retrieval-Augmented Generation (RAG) chatbot designed to provide grounded, evidence-based answers regarding Ghana's 2025 National Budget and historical Election Results. By combining semantic vector search with lexical keyword matching, GovLens ensures that AI responses are always tied to official government documentation, effectively eliminating common large language model (LLM) hallucinations.

## ✨ Features
- **Query Input:** Simple natural language interface for complex data retrieval.
- **Hybrid Retrieval:** Combined FAISS (Vector) and BM25 (Lexical) search for maximum accuracy.
- **Similarity Scoring:** Real-time visibility into retrieval confidence and document scores.
- **Context-Based Generation:** LLM responses restricted strictly to verified dataset evidence.
- **Hallucination Control:** Multi-layer safeguards including similarity thresholds and specialized prompt templates.
- **Innovation - Domain Router:** Automatic intent classification to prevent cross-domain data contamination.

## 🛠️ Tech Stack
- **Core:** Python 3.11
- **UI:** Streamlit
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector DB:** FAISS (IndexFlatIP)
- **Search Engine:** Rank-BM25
- **LLM:** Groq API (`llama-3.3-70b-versatile`)
- **Deployment:** Render.com

## 🚀 How to Run
### Local Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NaaAyima/ai_10022200119.git
   cd ai_10022200119
   ```
2. **Environment Configuration:**
   - Create a `.env` file in the root directory.
   - Add your Groq API Key: `GROQ_API_KEY=your_key_here`
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application:**
   ```bash
   PYTHONUTF8=1 .venv/Scripts/python.exe -m streamlit run app.py
   ```

## 📋 Usage
1. Enter a question about the budget or elections in the chat box.
2. View the generated response grounded in official data.
3. Use the **"🔍 View Retrieved Context"** dropdown to audit the search results.
4. Provide feedback using the **Yes/No** helpfulness buttons.

## 📂 Project Structure
- `part_a/`: Data cleaning and semantic chunking strategies.
- `part_b/`: Hybrid retrieval system (FAISS + BM25).
- `part_c/`: Prompt engineering and Context Window Manager.
- `part_d/`: End-to-end RAG pipeline integration.
- `part_e/`: Critical evaluation and adversarial testing logs.
- `part_g/`: Innovation component (Domain-Aware Router).
- `app.py`: Main Streamlit application.

---

## 🏛️ Assignment Context
### MAIN QUESTION: DESIGN A RAG System chatbot (60 MARKS)
You are tasked to build a RAG AI chat assistant for Academic City.
- **Dataset:** Ghana Election Result & 2025 Budget Statement.
- **PART A:** Data cleaning, chunking strategy design, and comparative analysis.
- **PART B:** Custom retrieval system with FAISS, BM25 Hybrid search, and failure case fixing.
- **PART C:** Prompt engineering with hallucination control and context window management.
- **PART D:** Complete RAG pipeline construction and logging.
- **PART E:** Adversarial testing and RAG vs Pure LLM comparison.
- **PART F:** Architecture design and system justification.
- **PART G:** Innovation component (Domain-specific scoring or similar).

**Submission Requirements:**
- GitHub Repository: `ai_10022200119`
- Cloud Deployment (Render)
- Video Walkthrough (Max 2 mins)
- Detailed Documentation & Experiment Logs

---

## 🗒️ Notes
- **Scope:** The chatbot's knowledge is strictly bounded by the provided 2025 Budget and Election datasets.
- **Fallback:** If a query falls outside the dataset scope or below the similarity threshold, the bot returns a standardized refusal to prevent hallucination.
- **Architecture:** Full technical documentation and Mermaid diagrams are available in `data/processed/reports/architecture.md`.
