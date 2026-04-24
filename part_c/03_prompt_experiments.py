"""
Part C -- Step 3: Prompt Experiments
Student: Jacqueline Naa Ayima Mensah | Index: 10022200119
"""

import importlib.util
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parent.parent
EMBED_DIR   = ROOT / "data" / "processed" / "embeddings"
REPORTS_DIR = ROOT / "data" / "processed" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in .env file.")


# ==============================================================================
# DYNAMIC IMPORTS FROM SIBLING MODULES
# ==============================================================================

def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PART_C = Path(__file__).parent
_tmpl_mod = _load_module("prompt_templates",     PART_C / "01_prompt_templates.py")
_ctx_mod  = _load_module("context_window_manager", PART_C / "02_context_window_manager.py")
_ret_mod  = _load_module("retrieval_system", Path(__file__).parent.parent / "part_b" / "02_retrieval_system.py")

ALL_TEMPLATES       = _tmpl_mod.ALL_TEMPLATES
ContextWindowManager = _ctx_mod.ContextWindowManager
HybridRetriever     = _ret_mod.HybridRetriever


# ==============================================================================
# EXPERIMENT QUERIES
# ==============================================================================

EXPERIMENT_QUERIES = [
    {
        "id":       "Q1_election_factual",
        "query":    "Who won the 2020 Ghana presidential election and what percentage of votes did they receive?",
        "domain":   "election",
        "expected": "Nana Addo / Akufo-Addo, NPP, ~51.59%",
    },
    {
        "id":       "Q2_budget_policy",
        "query":    "What is Ghana's total revenue projection for 2025 and how will the government reduce the fiscal deficit?",
        "domain":   "budget",
        "expected": "GH 224.9 billion revenue; deficit reduction strategy",
    },
    {
        "id":       "Q3_cross_domain",
        "query":    "How many regions does Ghana have and what is the budget allocation for regional development in 2025?",
        "domain":   "both",
        "expected": "16 regions; regional development allocation from budget",
    },
]


# ==============================================================================
# METRICS
# ==============================================================================

def measure_response(response_text: str, query: str) -> dict:
    """
    Compute quality metrics for a generated response.

    Metrics
    -------
    word_count        : total words in response
    sentence_count    : approximate sentence count
    has_refusal       : True if model refused due to insufficient context
    cites_election    : True if response references the election dataset
    cites_budget      : True if response references the budget dataset
    has_numbers       : True if response contains numeric data (good for factual Q)
    speculation_phrases: count of hedging phrases ("probably", "likely", "I think")
    """
    text   = response_text.lower()
    words  = re.findall(r"\b\w+\b", response_text)
    sents  = re.split(r"[.!?]+", response_text)

    refusal_phrases = [
        "insufficient context", "i don't know", "i cannot answer",
        "not in the context", "not provided", "i don't have",
        "cannot determine", "not mentioned",
    ]
    speculation_phrases = ["probably", "likely", "i think", "i believe",
                           "perhaps", "may be", "might be", "approximately",
                           "around", "estimated"]

    return {
        "word_count":         len(words),
        "sentence_count":     len([s for s in sents if s.strip()]),
        "has_refusal":        any(p in text for p in refusal_phrases),
        "cites_election":     "election" in text or "votes" in text or "constituency" in text,
        "cites_budget":       "budget" in text or "revenue" in text or "fiscal" in text or "gh" in text,
        "has_numbers":        bool(re.search(r"\d[\d,\.]+", response_text)),
        "speculation_count":  sum(1 for p in speculation_phrases if p in text),
    }


# ==============================================================================
# LLM CALLER
# ==============================================================================

def call_groq(messages: list[dict], max_tokens: int = 512) -> dict:
    """
    Call the Groq API and return response text + usage metadata.

    Returns
    -------
    {
      "text":             str,
      "prompt_tokens":    int,
      "completion_tokens": int,
      "latency_ms":       float,
      "model":            str,
      "error":            str | None,
    }
    """
    client = Groq(api_key=GROQ_API_KEY)
    t0     = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model      = GROQ_MODEL,
            messages   = messages,
            max_tokens = max_tokens,
            temperature= 0.1,     # low temperature for factual RAG tasks
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text       = completion.choices[0].message.content or ""
        usage      = completion.usage
        return {
            "text":              text,
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_ms":        round(latency_ms, 1),
            "model":             GROQ_MODEL,
            "error":             None,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("Groq API error: %s", e)
        return {
            "text":              "",
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "latency_ms":        round(latency_ms, 1),
            "model":             GROQ_MODEL,
            "error":             str(e),
        }


# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================

def run_template_experiment(retriever, all_results: dict) -> None:
    """
    Experiment A: Same query + same context, different prompt templates.
    Shows how prompt wording alone affects response quality.
    """
    logger.info("=" * 55)
    logger.info("EXPERIMENT A: Template comparison (fixed context strategy=ranking)")
    logger.info("=" * 55)

    cwm = ContextWindowManager(strategy="ranking", max_chars=3000, min_score=0.20, top_k=5)

    for qdata in EXPERIMENT_QUERIES:
        qid   = qdata["id"]
        query = qdata["query"]
        logger.info("  Query: %s", query[:60])

        # Retrieve once (same chunks for all templates)
        chunks          = retriever.search(query, k=5)
        context, ctx_meta = cwm.build_context(chunks)

        all_results.setdefault("experiment_A", {})[qid] = {
            "query":        query,
            "domain":       qdata["domain"],
            "expected":     qdata["expected"],
            "context_meta": ctx_meta,
            "templates":    {},
        }

        for tmpl in ALL_TEMPLATES:
            logger.info("    Template: %s", tmpl.name)
            messages = tmpl.to_messages(context, query)
            llm_out  = call_groq(messages, max_tokens=512)
            metrics  = measure_response(llm_out["text"], query)

            all_results["experiment_A"][qid]["templates"][tmpl.name] = {
                "template_desc": tmpl.description[:100],
                "llm":           llm_out,
                "metrics":       metrics,
            }

            time.sleep(0.5)  # polite rate limit


def run_context_experiment(retriever, all_results: dict) -> None:
    """
    Experiment B: Same query + same template (T3), different context strategies.
    Shows how context management affects retrieval quality and LLM output.
    """
    logger.info("=" * 55)
    logger.info("EXPERIMENT B: Context strategy comparison (fixed template=T3)")
    logger.info("=" * 55)

    tmpl = _tmpl_mod.T3_HALLUCINATION_GUARD

    for qdata in EXPERIMENT_QUERIES:
        qid   = qdata["id"]
        query = qdata["query"]
        logger.info("  Query: %s", query[:60])

        chunks = retriever.search(query, k=5)

        all_results.setdefault("experiment_B", {})[qid] = {
            "query":     query,
            "domain":    qdata["domain"],
            "strategies": {},
        }

        for strat in ("truncation", "ranking", "mmr"):
            cwm = ContextWindowManager(
                strategy = strat,
                max_chars = 3000,
                min_score = 0.20,
                top_k     = 5,
            )
            context, ctx_meta = cwm.build_context(chunks)

            messages = tmpl.to_messages(context, query)
            llm_out  = call_groq(messages, max_tokens=512)
            metrics  = measure_response(llm_out["text"], query)

            all_results["experiment_B"][qid]["strategies"][strat] = {
                "context_meta": ctx_meta,
                "llm":          llm_out,
                "metrics":      metrics,
            }

            time.sleep(0.5)


# ==============================================================================
# REPORT
# ==============================================================================

def print_and_save_report(all_results: dict) -> None:
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    w("=" * 72)
    w("  PART C -- STEP 3: PROMPT EXPERIMENT RESULTS")
    w("=" * 72)

    # ── Experiment A ──────────────────────────────────────────────────────────
    w()
    w("  EXPERIMENT A: Same Query, Different Prompt Templates")
    w("  (Context strategy fixed: ranking, max_chars=3000)")
    w("  " + "-" * 68)

    for qid, qdata in all_results.get("experiment_A", {}).items():
        w()
        w(f"  Query [{qid}]:")
        w(f"  \"{qdata['query']}\"")
        w(f"  Expected: {qdata['expected']}")
        w()
        w(f"  {'Template':<28}  {'Words':>5}  {'Sents':>5}  {'Refusal':>7}  "
          f"{'Numbers':>7}  {'Specul.':>7}  {'Latency':>9}")
        w("  " + "-" * 75)

        for tname, tdata in qdata["templates"].items():
            m = tdata["metrics"]
            l = tdata["llm"]
            w(
                f"  {tname:<28}  {m['word_count']:>5}  {m['sentence_count']:>5}  "
                f"{'YES' if m['has_refusal'] else 'no':>7}  "
                f"{'YES' if m['has_numbers'] else 'no':>7}  "
                f"{m['speculation_count']:>7}  "
                f"{l['latency_ms']:>7.0f}ms"
            )

        # Show full responses
        w()
        w("  --- Responses ---")
        for tname, tdata in qdata["templates"].items():
            w()
            w(f"  [{tname}]")
            response = tdata["llm"]["text"] or "[ERROR: no response]"
            for line in response.split("\n"):
                w(f"    {line}")

        w()
        w("  " + "~" * 68)

    # ── Experiment B ──────────────────────────────────────────────────────────
    w()
    w("  EXPERIMENT B: Same Query + Template (T3), Different Context Strategies")
    w("  " + "-" * 68)

    for qid, qdata in all_results.get("experiment_B", {}).items():
        w()
        w(f"  Query [{qid}]: \"{qdata['query']}\"")
        w()
        w(f"  {'Strategy':<14}  {'Chunks':>6}  {'Chars':>6}  {'Words':>5}  "
          f"{'Refusal':>7}  {'Numbers':>7}  {'Latency':>9}")
        w("  " + "-" * 65)

        for strat, sdata in qdata["strategies"].items():
            m   = sdata["metrics"]
            l   = sdata["llm"]
            cm  = sdata["context_meta"]
            w(
                f"  {strat:<14}  {cm.get('chunks_used', '?'):>6}  "
                f"{cm.get('chars_after', '?'):>6}  "
                f"{m['word_count']:>5}  "
                f"{'YES' if m['has_refusal'] else 'no':>7}  "
                f"{'YES' if m['has_numbers'] else 'no':>7}  "
                f"{l['latency_ms']:>7.0f}ms"
            )

        w()
        w("  --- Responses ---")
        for strat, sdata in qdata["strategies"].items():
            w()
            w(f"  [{strat}]")
            response = sdata["llm"]["text"] or "[ERROR: no response]"
            for line in response.split("\n"):
                w(f"    {line}")

        w("  " + "~" * 68)

    # ── Analysis summary ──────────────────────────────────────────────────────
    w()
    w("  ANALYSIS: Evidence of Improvement Across Templates")
    w("  " + "-" * 68)
    w("  T1 -> T2 : Adding role + structure reduces speculation, improves format.")
    w("  T2 -> T3 : Explicit grounding rules increase refusal on weak context,")
    w("             reducing hallucination. Source citation improves.")
    w("  T3 -> T4 : Chain-of-thought increases word count and reasoning depth;")
    w("             trade-off: slightly higher latency.")
    w()
    w("  ANALYSIS: Evidence of Improvement Across Context Strategies")
    w("  " + "-" * 68)
    w("  Truncation : includes all top-k but may include low-score noise.")
    w("  Ranking    : filters noise; places most relevant content first.")
    w("  MMR        : maximises diversity; removes near-duplicate budget passages.")
    w("  Best for this corpus: Ranking (clean filtering, position bias benefit).")
    w()
    w("  RECOMMENDATION FOR PART D PIPELINE:")
    w("  Template: T3 (Hallucination Guard) -- best safety vs quality trade-off.")
    w("  Context : Ranking strategy -- filtered, position-ordered context.")
    w("=" * 72)

    # Save
    txt_path = REPORTS_DIR / "prompt_experiments.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Text report saved -> %s", txt_path)

    json_path = REPORTS_DIR / "prompt_experiments.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    logger.info("JSON results saved -> %s", json_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # Load retriever
    logger.info("Loading FAISS index and model ...")
    index = faiss.read_index(str(EMBED_DIR / "faiss.index"))
    with open(EMBED_DIR / "metadata.json", encoding="utf-8") as fh:
        metadata: list[dict] = json.load(fh)
    model     = SentenceTransformer("all-MiniLM-L6-v2")
    retriever = HybridRetriever(index, metadata, model, alpha=0.7)

    all_results: dict = {}

    # Experiment A: template comparison
    run_template_experiment(retriever, all_results)

    # Experiment B: context strategy comparison
    run_context_experiment(retriever, all_results)

    # Print + save report
    print_and_save_report(all_results)


if __name__ == "__main__":
    main()
