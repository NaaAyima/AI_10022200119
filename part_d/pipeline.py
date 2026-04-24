"""
Part D -- Full RAG Pipeline Implementation
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Pipeline:
  User Query
    --> Stage 1: Query Preprocessing  (stop-word removal, normalisation)
    --> Stage 2: Hybrid Retrieval     (BM25 + FAISS cosine, alpha=0.7)
    --> Stage 3: Context Selection    (Ranking strategy, score >= 0.20)
    --> Stage 4: Prompt Construction  (T3 Hallucination Guard template)
    --> Stage 5: LLM Generation       (Groq llama-3.3-70b-versatile)
    --> Stage 6: Response + Logging   (full pipeline log to JSON + txt)

Additional constraints met
---------------------------
  - Logging at each stage (timestamped, structured)
  - Retrieved documents displayed with similarity scores
  - Final prompt sent to LLM displayed verbatim
  - All pipeline runs saved to data/processed/logs/pipeline_runs.jsonl

Usage (interactive)
-------------------
  python part_d/pipeline.py
  python part_d/pipeline.py --query "Who won the 2020 Ghana election?"

Usage (as module)
-----------------
  from part_d.pipeline import RAGPipeline
  pipeline = RAGPipeline.load()
  result   = pipeline.run("Your question here")
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
LOGS_DIR  = ROOT / "data" / "processed" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MODULE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PART_B = ROOT / "part_b"
_PART_C = ROOT / "part_c"

_ret_mod  = _load("retrieval_system",       _PART_B / "02_retrieval_system.py")
_tmpl_mod = _load("prompt_templates",       _PART_C / "01_prompt_templates.py")
_ctx_mod  = _load("context_window_manager", _PART_C / "02_context_window_manager.py")

HybridRetriever      = _ret_mod.HybridRetriever
T3_HALLUCINATION_GUARD = _tmpl_mod.T3_HALLUCINATION_GUARD
ContextWindowManager = _ctx_mod.ContextWindowManager


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGE LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class PipelineLogger:
    """
    Records every stage of a single pipeline run as a structured dict.
    Writes to JSONL log file (one JSON object per line = one pipeline run).
    Also prints a rich human-readable trace to stdout.
    """

    def __init__(self):
        self.run_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        self.stages: list[dict] = []
        self.t_start  = time.perf_counter()

    def log(self, stage_num: int, stage_name: str, data: dict) -> None:
        elapsed = round((time.perf_counter() - self.t_start) * 1000, 1)
        entry   = {
            "stage":   stage_num,
            "name":    stage_name,
            "elapsed_ms": elapsed,
            **data,
        }
        self.stages.append(entry)

        # Pretty-print to stdout
        print(f"\n{'─' * 65}")
        print(f"  STAGE {stage_num}: {stage_name}  (+{elapsed:.0f}ms)")
        print(f"{'─' * 65}")
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 300:
                print(f"  {k}:")
                for line in v.split("\n"):
                    print(f"    {line}")
            elif isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    {item}")
            else:
                print(f"  {k}: {v}")

    def save(self, query: str, response: str, total_ms: float) -> dict:
        """Persist this run to the JSONL log file and return the full record."""
        record = {
            "run_id":    self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query":     query,
            "response":  response,
            "total_ms":  round(total_ms, 1),
            "stages":    self.stages,
        }
        log_path = LOGS_DIR / "pipeline_runs.jsonl"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Run logged -> %s  (run_id=%s)", log_path, self.run_id)
        return record


# ══════════════════════════════════════════════════════════════════════════════
# STOP WORDS (Stage 1 helper)
# ══════════════════════════════════════════════════════════════════════════════

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "it","its","this","that","these","those","what","which","who","how","when",
    "where","why","all","each","both","more","most","also","than","then","so",
    "if","as","up","out","about","into","through","during","before","after",
    "between","such","their","our","your","my","his","her","we","they","you",
    "he","she","i","not","no","can","just","any","some","much","many","over",
}


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Full end-to-end RAG pipeline for the Academic City chatbot.

    Pipeline stages
    ---------------
    1. Query Preprocessing  -- clean + stop-word removal
    2. Hybrid Retrieval     -- BM25 + FAISS, alpha=0.70
    3. Context Selection    -- ranking strategy, min_score=0.20, max_chars=3000
    4. Prompt Construction  -- T3 Hallucination Guard template
    5. LLM Generation       -- Groq llama-3.3-70b-versatile
    6. Response & Logging   -- structured log to JSONL

    Parameters
    ----------
    retriever   : HybridRetriever instance
    top_k       : number of chunks to retrieve (default 5)
    max_tokens  : LLM max output tokens (default 512)
    groq_model  : Groq model name
    temperature : LLM sampling temperature (default 0.1 for factual RAG)
    """

    def __init__(
        self,
        retriever:   HybridRetriever,
        top_k:       int   = 5,
        max_tokens:  int   = 512,
        groq_model:  str   = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
    ):
        self.retriever   = retriever
        self.top_k       = top_k
        self.max_tokens  = max_tokens
        self.groq_model  = groq_model
        self.temperature = temperature
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.cwm         = ContextWindowManager(
            strategy  = "ranking",
            max_chars = 3000,
            min_score = 0.20,
            top_k     = top_k,
        )
        logger.info(
            "RAGPipeline ready | model=%s | top_k=%d | temp=%.2f",
            groq_model, top_k, temperature,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        embed_dir:  Path  = EMBED_DIR,
        model_name: str   = "all-MiniLM-L6-v2",
        alpha:      float = 0.7,
        **kwargs,
    ) -> "RAGPipeline":
        """Load all components and return a ready pipeline."""
        logger.info("Loading FAISS index ...")
        index = faiss.read_index(str(embed_dir / "faiss.index"))

        logger.info("Loading metadata ...")
        with open(embed_dir / "metadata.json", encoding="utf-8") as fh:
            metadata: list[dict] = json.load(fh)

        logger.info("Loading embedding model: %s ...", model_name)
        model = SentenceTransformer(model_name)

        retriever = HybridRetriever(index, metadata, model, alpha=alpha)
        return cls(retriever=retriever, **kwargs)

    # ------------------------------------------------------------------
    # Stage 1: Query Preprocessing
    # ------------------------------------------------------------------

    def _stage1_preprocess(self, raw_query: str, plog: PipelineLogger) -> str:
        tokens   = re.findall(r"\b[a-z]{2,}\b", raw_query.lower())
        filtered = [t for t in tokens if t not in STOP_WORDS]
        clean    = " ".join(filtered) if filtered else raw_query.lower()

        plog.log(1, "Query Preprocessing", {
            "raw_query":    raw_query,
            "clean_query":  clean,
            "tokens_removed": len(tokens) - len(filtered),
            "stop_words_stripped": sorted(set(tokens) - set(filtered)),
        })
        return clean

    # ------------------------------------------------------------------
    # Stage 2: Hybrid Retrieval
    # ------------------------------------------------------------------

    def _stage2_retrieve(
        self, clean_query: str, plog: PipelineLogger
    ) -> list[dict]:
        t0      = time.perf_counter()
        chunks  = self.retriever.hybrid_search(clean_query, k=self.top_k)
        elapsed = (time.perf_counter() - t0) * 1000

        retrieved_display = [
            f"[{c['rank']}] source={c['source']} | "
            f"hybrid={c['hybrid_score']:.4f} | "
            f"vector={c['vector_score']:.4f} | "
            f"bm25={c['bm25_score']:.4f} | "
            f"text: {c['text'][:120].replace(chr(10),' ')}..."
            for c in chunks
        ]

        plog.log(2, "Hybrid Retrieval", {
            "clean_query":    clean_query,
            "chunks_retrieved": len(chunks),
            "retrieval_ms":   round(elapsed, 1),
            "retrieved_documents": retrieved_display,
        })
        return chunks

    # ------------------------------------------------------------------
    # Stage 3: Context Selection
    # ------------------------------------------------------------------

    def _stage3_context(
        self, chunks: list[dict], plog: PipelineLogger
    ) -> str:
        context, meta = self.cwm.build_context(chunks)

        plog.log(3, "Context Selection", {
            "strategy":          meta["strategy"],
            "chunks_used":       meta["chunks_used"],
            "chunks_filtered":   meta.get("chunks_filtered_out", 0),
            "min_score":         self.cwm.min_score,
            "context_chars":     meta["chars_after"],
            "scores_included":   [round(s, 4) for s in meta.get("scores", [])],
            "context_preview":   context[:400] + "..." if len(context) > 400 else context,
        })
        return context

    # ------------------------------------------------------------------
    # Stage 4: Prompt Construction
    # ------------------------------------------------------------------

    def _stage4_prompt(
        self, raw_query: str, context: str, plog: PipelineLogger
    ) -> list[dict]:
        messages = T3_HALLUCINATION_GUARD.to_messages(context, raw_query)

        # Build full prompt string for display
        full_prompt = ""
        for m in messages:
            full_prompt += f"[{m['role'].upper()}]\n{m['content']}\n\n"

        plog.log(4, "Prompt Construction", {
            "template":    T3_HALLUCINATION_GUARD.name,
            "template_desc": T3_HALLUCINATION_GUARD.description[:100] + "...",
            "num_messages": len(messages),
            "final_prompt_sent_to_LLM": full_prompt,
        })
        return messages

    # ------------------------------------------------------------------
    # Stage 5: LLM Generation
    # ------------------------------------------------------------------

    def _stage5_generate(
        self, messages: list[dict], plog: PipelineLogger
    ) -> str:
        t0 = time.perf_counter()
        try:
            completion = self.groq_client.chat.completions.create(
                model       = self.groq_model,
                messages    = messages,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            text       = completion.choices[0].message.content or ""
            usage      = completion.usage

            plog.log(5, "LLM Generation", {
                "model":             self.groq_model,
                "temperature":       self.temperature,
                "max_tokens":        self.max_tokens,
                "prompt_tokens":     usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens":      usage.total_tokens,
                "latency_ms":        latency_ms,
                "response":          text,
            })
            return text

        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            plog.log(5, "LLM Generation", {
                "model":     self.groq_model,
                "error":     str(exc),
                "latency_ms": latency_ms,
            })
            return f"[ERROR] LLM call failed: {exc}"

    # ------------------------------------------------------------------
    # Stage 6: Log & Return
    # ------------------------------------------------------------------

    def _stage6_log(
        self,
        raw_query: str,
        response:  str,
        plog:      PipelineLogger,
        t_total:   float,
    ) -> dict:
        total_ms = (time.perf_counter() - t_total) * 1000
        plog.log(6, "Response & Logging", {
            "status":      "SUCCESS" if response and not response.startswith("[ERROR]") else "ERROR",
            "total_pipeline_ms": round(total_ms, 1),
            "log_file":    str(LOGS_DIR / "pipeline_runs.jsonl"),
        })
        return plog.save(raw_query, response, total_ms)

    # ------------------------------------------------------------------
    # Public run() method
    # ------------------------------------------------------------------

    def run(self, raw_query: str) -> dict:
        """
        Execute the full RAG pipeline for a single query.

        Returns the complete pipeline record (all stages + response).
        Also prints a rich stage-by-stage trace to stdout.
        """
        plog    = PipelineLogger()
        t_total = time.perf_counter()

        print(f"\n{'=' * 65}")
        print(f"  RAG PIPELINE  |  run_id: {plog.run_id}")
        print(f"  Query: \"{raw_query}\"")
        print(f"{'=' * 65}")

        clean_query = self._stage1_preprocess(raw_query, plog)
        chunks      = self._stage2_retrieve(clean_query, plog)
        context     = self._stage3_context(chunks, plog)
        messages    = self._stage4_prompt(raw_query, context, plog)
        response    = self._stage5_generate(messages, plog)
        record      = self._stage6_log(raw_query, response, plog, t_total)

        print(f"\n{'=' * 65}")
        print(f"  FINAL RESPONSE")
        print(f"{'=' * 65}")
        print(f"\n{response}\n")
        print(f"  [Total pipeline: {record['total_ms']:.0f}ms]")
        print(f"{'=' * 65}\n")

        return record


# ══════════════════════════════════════════════════════════════════════════════
# DEMO QUERIES (run as script)
# ══════════════════════════════════════════════════════════════════════════════

DEMO_QUERIES = [
    "Who won the 2020 Ghana presidential election and what percentage of votes did they receive?",
    "What is Ghana's projected total revenue for 2025?",
    "What is the government's strategy to reduce inflation in 2025?",
    "Which party won the most parliamentary seats in the 2016 Ghana election?",
    "What is Ghana's primary expenditure target as a percentage of GDP in 2025?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Academic City RAG Pipeline -- Part D"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query to run (omit for demo mode with 5 preset queries)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive REPL mode",
    )
    args = parser.parse_args()

    # Load pipeline (printed once)
    pipeline = RAGPipeline.load()

    if args.interactive:
        print("\n  Academic City RAG Chatbot -- Interactive Mode")
        print("  Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                q = input("  Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                pipeline.run(q)

    elif args.query:
        pipeline.run(args.query)

    else:
        # Demo mode: run all preset queries
        print(f"\n  Running {len(DEMO_QUERIES)} demo queries ...\n")
        for q in DEMO_QUERIES:
            pipeline.run(q)
            time.sleep(1)   # polite rate limiting between calls

    # Print log file location
    log_path = LOGS_DIR / "pipeline_runs.jsonl"
    print(f"\n  All pipeline runs logged to: {log_path}")


if __name__ == "__main__":
    main()
