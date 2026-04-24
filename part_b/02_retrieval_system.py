"""
Part B -- Step 2: Custom Retrieval System
Student: Jacqueline Naa Ayima Mensah | Index: 10022200119
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"


# ==============================================================================
# BM25  (from scratch -- no external ranking library)
# ==============================================================================

class BM25:
    """
    Okapi BM25 keyword retrieval.

    Parameters
    ----------
    corpus   : list of raw text strings (the same chunks in the FAISS index)
    k1       : term-frequency saturation factor (default 1.5)
    b        : length normalisation factor       (default 0.75)
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.n  = len(corpus)

        # Tokenise corpus
        self.tokenized: list[list[str]] = [self._tokenize(doc) for doc in corpus]
        self.avgdl: float = sum(len(d) for d in self.tokenized) / max(self.n, 1)

        # Build document-frequency table
        self.df: dict[str, int] = {}
        for doc_tokens in self.tokenized:
            for term in set(doc_tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # Pre-compute IDF for every term in vocabulary
        # Robertson-Sparck Jones IDF variant (smoothed):
        #   IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
        self.idf: dict[str, float] = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in self.df.items()
        }

        logger.info(
            "BM25 built: %d docs | vocab %d terms | avgdl %.1f tokens",
            self.n, len(self.idf), self.avgdl,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def score(self, query: str) -> np.ndarray:
        """
        Return a (N,) float32 array of BM25 scores for *query* against
        every document in the corpus.
        """
        q_terms = self._tokenize(query)
        scores  = np.zeros(self.n, dtype=np.float32)

        if not q_terms:
            return scores

        for i, doc_tokens in enumerate(self.tokenized):
            dl     = len(doc_tokens)
            tf_map: dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            doc_score = 0.0
            for term in q_terms:
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf[term]
                # BM25 term score
                numerator   = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                doc_score  += idf * numerator / denominator

            scores[i] = doc_score

        return scores

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return [(index, score), ...] for the top-k BM25 results."""
        scores  = self.score(query)
        indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in indices]


# ==============================================================================
# VECTOR RETRIEVER  (FAISS)
# ==============================================================================

class VectorRetriever:
    """
    FAISS cosine-similarity retrieval using the pre-built IndexFlatIP.

    Cosine similarity is computed by L2-normalising the query vector before
    calling index.search() -- the same normalisation applied in Step 1.
    """

    def __init__(self, index: faiss.IndexFlatIP, model: SentenceTransformer):
        self.index = index
        self.model = model

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode and L2-normalise a single query string."""
        vec  = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        return (vec / norm).astype(np.float32)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """
        Return [(faiss_index, cosine_score), ...] for top-k results.
        Scores are in [-1, 1]; 1.0 = identical.
        """
        q_vec = self._encode_query(query)
        scores, indices = self.index.search(q_vec, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]


# ==============================================================================
# HYBRID RETRIEVER  (BM25 + Vector)
# ==============================================================================

class HybridRetriever:
    """
    Hybrid Search: combines BM25 keyword scores with FAISS vector similarity.

    Score fusion method: Normalised Linear Combination (NLC)
      1. Compute BM25 scores across ALL documents.
      2. Compute vector cosine scores across ALL documents
         (using FAISS search over full corpus).
      3. Min-max normalise both score vectors to [0, 1] independently.
      4. Fuse: hybrid = alpha * vector_norm + (1 - alpha) * bm25_norm
      5. Return top-k by hybrid score.

    Why NLC over Reciprocal Rank Fusion (RRF)?
    -------------------------------------------
    NLC preserves score magnitude information -- a very high BM25 exact-match
    score will proportionally boost the final hybrid score, whereas RRF treats
    all top-k results equally regardless of score gap.  For domain-specific
    retrieval with exact entities (constituency names, budget line items), NLC
    is more precise.
    """

    def __init__(
        self,
        index:    faiss.IndexFlatIP,
        metadata: list[dict],
        model:    SentenceTransformer,
        alpha:    float = 0.7,
    ):
        self.index    = index
        self.metadata = metadata
        self.model    = model
        self.alpha    = alpha            # weight for vector score (1-alpha for BM25)
        self.n        = len(metadata)

        # Build BM25 over corpus of chunk texts
        corpus = [m["text"] for m in metadata]
        self.bm25 = BM25(corpus)
        logger.info(
            "HybridRetriever ready | alpha=%.2f (%.0f%% vector, %.0f%% BM25) | %d docs",
            alpha, alpha * 100, (1 - alpha) * 100, self.n,
        )

    # --------------------------------------------------------------------------
    # Factory
    # --------------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        embed_dir: Path  = EMBED_DIR,
        model_name: str  = MODEL_NAME,
        alpha: float     = 0.7,
    ) -> "HybridRetriever":
        """Load FAISS index + metadata + model and return a ready retriever."""
        logger.info("Loading FAISS index ...")
        index = faiss.read_index(str(embed_dir / "faiss.index"))

        logger.info("Loading metadata ...")
        with open(embed_dir / "metadata.json", encoding="utf-8") as fh:
            metadata: list[dict] = json.load(fh)

        logger.info("Loading embedding model: %s ...", model_name)
        model = SentenceTransformer(model_name)

        return cls(index, metadata, model, alpha)

    # --------------------------------------------------------------------------
    # Score helpers
    # --------------------------------------------------------------------------

    def _vector_scores_all(self, query: str) -> np.ndarray:
        """Return cosine similarity of query against ALL documents (shape N,)."""
        vec  = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        qvec = (vec / norm).astype(np.float32)

        # FAISS FlatIP: search all N vectors
        scores, indices = self.index.search(qvec, self.n)
        # Re-order scores back to corpus order
        ordered = np.empty(self.n, dtype=np.float32)
        ordered[indices[0]] = scores[0]
        return ordered

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1]. Returns zeros if all values identical."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # --------------------------------------------------------------------------
    # Public search methods
    # --------------------------------------------------------------------------

    def vector_search(self, query: str, k: int = 5) -> list[dict]:
        """Vector-only top-k retrieval (for comparison / ablation)."""
        scores, indices = self.index.search(
            self._encode_query(query), k
        )
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            m = self.metadata[idx]
            results.append({
                "rank":          rank,
                "chunk_id":      m["chunk_id"],
                "source":        m["source"],
                "text":          m["text"],
                "vector_score":  round(float(score), 6),
                "bm25_score":    None,
                "hybrid_score":  None,
            })
        return results

    def bm25_search(self, query: str, k: int = 5) -> list[dict]:
        """BM25-only top-k retrieval (for comparison / ablation)."""
        top = self.bm25.top_k(query, k)
        results = []
        for rank, (idx, score) in enumerate(top, 1):
            m = self.metadata[idx]
            results.append({
                "rank":         rank,
                "chunk_id":     m["chunk_id"],
                "source":       m["source"],
                "text":         m["text"],
                "vector_score": None,
                "bm25_score":   round(score, 6),
                "hybrid_score": None,
            })
        return results

    def hybrid_search(self, query: str, k: int = 5) -> list[dict]:
        """
        Hybrid Search: Normalised Linear Combination of vector + BM25.

        Steps
        -----
        1. Compute vector cosine scores for ALL docs.
        2. Compute BM25 scores for ALL docs.
        3. Min-max normalise both.
        4. Fuse: hybrid = alpha * vector_norm + (1 - alpha) * bm25_norm.
        5. Return top-k by hybrid score.
        """
        vec_scores  = self._vector_scores_all(query)
        bm25_scores = self.bm25.score(query)

        vec_norm  = self._minmax_norm(vec_scores)
        bm25_norm = self._minmax_norm(bm25_scores)

        hybrid = self.alpha * vec_norm + (1.0 - self.alpha) * bm25_norm
        top_k_indices = np.argsort(hybrid)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            m = self.metadata[idx]
            results.append({
                "rank":         rank,
                "chunk_id":     m["chunk_id"],
                "source":       m["source"],
                "text":         m["text"],
                "vector_score": round(float(vec_scores[idx]),  6),
                "bm25_score":   round(float(bm25_scores[idx]), 6),
                "hybrid_score": round(float(hybrid[idx]),      6),
            })
        return results

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Default search -- calls hybrid_search."""
        return self.hybrid_search(query, k)

    def _encode_query(self, query: str) -> np.ndarray:
        vec  = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        return (vec / norm).astype(np.float32)

    # --------------------------------------------------------------------------
    # Display
    # --------------------------------------------------------------------------

    def display(self, results: list[dict], max_text: int = 200) -> None:
        """Pretty-print retrieval results to stdout."""
        print()
        for r in results:
            print(f"  Rank {r['rank']}  |  source: {r['source']}")
            if r["vector_score"] is not None:
                print(f"    vector_score : {r['vector_score']:.4f}")
            if r["bm25_score"] is not None:
                print(f"    bm25_score   : {r['bm25_score']:.4f}")
            if r["hybrid_score"] is not None:
                print(f"    hybrid_score : {r['hybrid_score']:.4f}")
            snippet = r["text"].replace("\n", " ")[:max_text]
            print(f"    text         : {snippet}...")
            print()


# ==============================================================================
# DEMO  (run as script)
# ==============================================================================

DEMO_QUERIES = [
    "Who won the Ghana presidential election?",
    "What is the government's plan for education spending in 2025?",
    "inflation and economic growth targets",
    "total votes by region constituency",
    "infrastructure investment roads and energy",
]

def main() -> None:
    retriever = HybridRetriever.load()

    print("\n" + "=" * 65)
    print("  PART B -- STEP 2: HYBRID RETRIEVAL SYSTEM DEMO")
    print("=" * 65)

    for query in DEMO_QUERIES:
        print(f"\n  Query: \"{query}\"")
        print("  " + "-" * 55)
        results = retriever.search(query, k=3)
        retriever.display(results, max_text=180)

    print("=" * 65)


if __name__ == "__main__":
    main()
