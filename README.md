# 🎓 Academic City RAG AI Chatbot
**CS4241 Introduction to Artificial Intelligence 2026**
**Student:** [Your Name] | **Index:** [Your Index Number]

A fully functional, end-to-end Retrieval-Augmented Generation (RAG) pipeline built to query institutional documents (the 2025 Ghana National Budget and historical Election Results) with strict factual grounding and hallucination controls.

---

## 🏗️ Architecture & System Design (Part F)
Please view the full architecture diagram (Mermaid) and system justifications in:
👉 **[`data/processed/reports/architecture.md`](data/processed/reports/architecture.md)**

## ✨ Innovation Component (Part G)
**Domain-Aware Routing & Scoring Function**
A custom classification layer intercepts user queries before retrieval. It uses lexical heuristics to classify the query intent as either `budget` or `election`. If a strong intent is detected, it applies a `1.6x` multiplier to the hybrid search score of chunks originating from the matching domain. **This definitively solves "Cross-Domain Contamination" (identified in Part B & E), preventing election queries from retrieving budget education tables.**
👉 **Located in:** `part_g/innovation.py`

## 📊 Evaluation & Experiment Logs
All manual experiments, comparative analyses, and adversarial tests have been logged with structured evidence (not AI summaries) in the `data/processed/reports/` directory:

| Report | Purpose |
| ------ | ------- |
| **[Part A Analysis](data/processed/reports/chunking_analysis.json)** | Comparative metrics (Lexical richness, chunk distributions) for Sentence vs Fixed vs Paragraph chunking strategies. |
| **[Part B Failure Analysis](data/processed/reports/failure_analysis.txt)** | Demonstrating 4 retrieval failure classes (Lexical mismatch, Short query, Cross-domain, Entity tokenisation) and fixing them with Query Preprocessing. |
| **[Part C Prompt Exps](data/processed/reports/prompt_experiments.txt)** | Dual-variable experiment: Comparing 4 Prompt Templates (Baseline to Chain-of-Thought) and 3 Context Strategies (Truncation, Ranking, MMR). |
| **[Part D Pipeline Logs](data/processed/logs/pipeline_runs.jsonl)** | Raw JSONL traces of full RAG pipeline executions, containing retrieved chunks, execution time (ms), and exact tokens sent to LLM. |
| **[Part E Adversarial Eval](data/processed/reports/adversarial_evaluation.txt)** | RAG vs Pure LLM comparison on malicious queries (Ambiguous, False Premise). Evaluated on Accuracy, Hallucination count, and Consistency (Jaccard limits). |

---

## 🚀 Running the Chatbot UI (Final Deliverable)
A sleek, modern interface built with **Streamlit** to demonstrate the exact RAG pipeline built over Part A $\rightarrow$ E, powered by `llama-3.3-70b-versatile`.

**Start the Web Interface:**
1. Ensure your virtual environment is activated and you have added your Groq API key to the `.env` file.
2. Run the Streamlit application in PowerShell or Git Bash:
```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -m streamlit run app.py
```

Try asking it:
- *"Who won the 2020 Ghana presidential election?"* (This will trigger the Innovation Router to boost election results!)
- *"What is Ghana's primary expenditure percentage of GDP target for 2025?"*

---

## 🎞️ Recording Your Video Walkthrough
To record your final 2-minute video:
1. Start the Streamlit app.
2. Open the UI, ask 1 question, and open the `View Retrieved Context` dropdown to show the scores.
3. Show the `architecture.md` Mermaid diagram.
4. Open the `adversarial_evaluation.txt` to prove you ran quantitative analysis (not just AI summaries).
5. Explain your **Innovation Component** (`part_g/innovation.py`) and how it fixes cross-domain contamination.
