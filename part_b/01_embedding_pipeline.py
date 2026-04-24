"""
Part B -- Step 1: Embedding Pipeline
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Embedding model : sentence-transformers  all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~90 MB download on first run (cached locally after that)
  - No API key required; runs 100% locally

Vector storage  : FAISS IndexFlatIP
  - Exact nearest-neighbour search (no approximation)
  - Inner-Product on L2-normalised vectors == cosine similarity
  - Appropriate for our corpus size (<5000 chunks)

Chunk strategy used
-------------------
  Election  -> fixed_size   (219 chunks, avg 510 chars)
    Reason: uniform size prevents embedding model truncation;
            election rows are short -- fixed windows preserve full rows.
  Budget    -> sentence_based (715 chunks, avg 1257 chars)
    Reason: sentence boundaries preserve semantic coherence in policy text.
    Note  : chunks >512 tokens are auto-truncated by the model's tokeniser,
            so a hard cap of 512 chars is applied pre-encoding.

Outputs (saved to data/processed/embeddings/)
---------------------------------------------
  faiss.index       -- FAISS IndexFlatIP binary
  metadata.json     -- list of {chunk_id, source, strategy, text, char_count}
  embed_config.json -- records model name, dims, corpus size
"""

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Paths ---------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
EMBED_DIR  = ROOT / "data" / "processed" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# -- Config --------------------------------------------------------------------
MODEL_NAME       = "all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384
BATCH_SIZE       = 64          # chunks per encoding batch
MAX_CHARS        = 2000        # hard cap to avoid tokeniser overflow
ELECTION_STRAT   = "fixed_size"
BUDGET_STRAT     = "sentence_based"


# ==============================================================================
# HELPERS
# ==============================================================================

def load_chunks(chunks_path: Path,
                election_strategy: str,
                budget_strategy: str) -> list[dict]:
    """
    Load and filter chunks from Part A output.

    For each dataset the strategy that maximises retrieval quality
    (established in Part A's comparative analysis) is selected:
      election -> fixed_size   (uniform size, avoids embedding truncation)
      budget   -> sentence_based (preserves policy sentence semantics)

    Returns a flat list of chunk dicts.
    """
    with open(chunks_path, encoding="utf-8") as fh:
        all_chunks: dict = json.load(fh)

    selected: list[dict] = []

    # Election
    election_chunks = all_chunks.get("election", {}).get(election_strategy, [])
    logger.info("Election / %s: %d chunks loaded", election_strategy, len(election_chunks))
    selected.extend(election_chunks)

    # Budget
    budget_chunks = all_chunks.get("budget", {}).get(budget_strategy, [])
    logger.info("Budget   / %s: %d chunks loaded", budget_strategy, len(budget_chunks))
    selected.extend(budget_chunks)

    logger.info("Total chunks for embedding: %d", len(selected))
    return selected


def truncate_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """
    Truncate text to max_chars to avoid tokeniser overflow.
    Truncation preserves whole words: cuts at the last space within limit.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut if cut else text[:max_chars]


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row of a 2D float32 matrix.

    Design decision: normalising before adding to IndexFlatIP converts
    inner-product scores into cosine similarity scores, which are more
    interpretable (range [-1, 1], higher = more similar).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)   # avoid division by zero
    return (matrix / norms).astype(np.float32)


# ==============================================================================
# EMBEDDING
# ==============================================================================

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Encode all chunks in batches, returning an (N, D) float32 matrix.

    Batch encoding is ~8x faster than individual calls on CPU and avoids
    redundant tokenisation overhead.
    """
    texts = [truncate_text(c["text"]) for c in chunks]
    total = len(texts)
    logger.info("Encoding %d chunks in batches of %d ...", total, BATCH_SIZE)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we normalise manually below
    )
    elapsed = time.perf_counter() - t0

    logger.info(
        "Encoded %d chunks in %.1f s (%.0f chunks/s)",
        total, elapsed, total / elapsed,
    )
    return embeddings.astype(np.float32)


# ==============================================================================
# FAISS INDEX
# ==============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP index from L2-normalised embeddings.

    IndexFlatIP chosen because:
    - Exact nearest-neighbour (no approximation error).
    - With L2-normalised vectors, inner product == cosine similarity.
    - Our corpus (<5000 chunks) is small enough that exact search is fast.
    - No training required (unlike IVF or HNSW indices).

    The index is NOT wrapped in IndexIDMap because chunk positions in the
    metadata list are the implicit IDs (position == chunk index in FAISS).
    """
    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP (dim=%d, vectors=%d) ...", dim, len(embeddings))

    index = faiss.IndexFlatIP(dim)
    normalised = l2_normalise(embeddings)
    index.add(normalised)

    logger.info("FAISS index built. Total vectors: %d", index.ntotal)
    return index


# ==============================================================================
# SAVE
# ==============================================================================

def save_artifacts(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    model_name: str,
    embed_dir: Path,
) -> None:
    """Save the FAISS index, metadata, and embedding config to disk."""
    # FAISS binary index
    index_path = embed_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index -> %s", index_path)

    # Metadata (subset of chunk fields needed for retrieval display)
    metadata = [
        {
            "chunk_id":    c.get("chunk_id", ""),
            "source":      c.get("source", ""),
            "strategy":    c.get("strategy", ""),
            "text":        c.get("text", ""),
            "char_count":  c.get("char_count", 0),
        }
        for c in chunks
    ]
    meta_path = embed_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Saved metadata (%d chunks) -> %s", len(metadata), meta_path)

    # Config
    config = {
        "model":       model_name,
        "dim":         EMBEDDING_DIM,
        "corpus_size": index.ntotal,
        "datasets":    {"election": ELECTION_STRAT, "budget": BUDGET_STRAT},
        "index_type":  "IndexFlatIP (cosine similarity via L2-normalisation)",
    }
    cfg_path = embed_dir / "embed_config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Saved config -> %s", cfg_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # -- Load chunks -----------------------------------------------------------
    chunks = load_chunks(CHUNKS_DIR / "all_chunks.json", ELECTION_STRAT, BUDGET_STRAT)

    # -- Load model ------------------------------------------------------------
    logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
    logger.info("(First run downloads ~90 MB -- cached after that)")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded. Max sequence length: %d tokens", model.max_seq_length)

    # -- Embed -----------------------------------------------------------------
    embeddings = embed_chunks(chunks, model)

    # -- Build FAISS index -----------------------------------------------------
    index = build_faiss_index(embeddings)

    # -- Save ------------------------------------------------------------------
    save_artifacts(index, chunks, MODEL_NAME, EMBED_DIR)

    # -- Sanity check: query the index with a test sentence --------------------
    logger.info("Running sanity check ...")
    test_query = "Ghana presidential election results"
    q_vec = model.encode([test_query], convert_to_numpy=True).astype(np.float32)
    q_vec = l2_normalise(q_vec)
    scores, indices = index.search(q_vec, k=3)
    print("\n" + "=" * 65)
    print("  PART B -- STEP 1: EMBEDDING PIPELINE -- COMPLETE")
    print("=" * 65)
    print(f"  Model        : {MODEL_NAME} ({EMBEDDING_DIM} dims)")
    print(f"  Total chunks : {index.ntotal}")
    print(f"  Index type   : FAISS IndexFlatIP (cosine similarity)")
    print(f"  Output dir   : {EMBED_DIR}")
    print(f"\n  Sanity check -- top-3 for: \"{test_query}\"")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        chunk = chunks[idx]
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"  [{rank}] score={score:.4f} | source={chunk['source']}")
        print(f"      {preview}...")
    print("=" * 65)


if __name__ == "__main__":
    main()
