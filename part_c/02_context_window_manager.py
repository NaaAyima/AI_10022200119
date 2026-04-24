"""
Part C -- Step 2: Context Window Manager
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Manages how retrieved chunks are selected, ranked, filtered, and
formatted before being injected into a prompt template.

Three management strategies are implemented and compared:

  Strategy 1 -- Truncation
    Take top-k chunks from the retriever, concatenate, hard-cut at
    max_chars characters.  Simple but may slice mid-sentence.

  Strategy 2 -- Score-Based Ranking + Threshold Filtering
    Only include chunks whose hybrid_score >= min_score threshold.
    Sort by descending score so the most relevant content appears
    first in the context window (recency / position bias in LLMs
    means earlier content is attended to more strongly).

  Strategy 3 -- MMR-style Diversity Filter
    Maximal Marginal Relevance (simplified, no embeddings needed here):
    iteratively select the next chunk that is (a) above min_score AND
    (b) not a near-duplicate of an already-selected chunk.  Prevents
    redundant context blocks from eating the window budget.

Design rationale for context window size
-----------------------------------------
llama-3.3-70b-versatile has a 128k-token context window.
However, prompt + context + completion should fit in one request.
We cap context at 3000 chars (~750 tokens) to leave room for:
  - System message (~150 tokens)
  - Query (~50 tokens)
  - Response (~500 tokens)
Total estimated: ~1450 tokens << 128k limit.
Using a smaller context window also forces the manager to select
the most relevant chunks rather than dumping everything.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# CONFIG
# ==============================================================================

DEFAULT_MAX_CHARS  = 3000    # hard cap on total context string length
DEFAULT_MIN_SCORE  = 0.20    # minimum hybrid_score to include a chunk
DEFAULT_TOP_K      = 5       # max chunks to consider from retriever
MMR_SIM_THRESHOLD  = 0.85    # Jaccard similarity above which chunks are "duplicates"


# ==============================================================================
# HELPERS
# ==============================================================================

def _tokenize(text: str) -> set[str]:
    """Simple word tokeniser for Jaccard-based duplicate detection."""
    return set(re.findall(r"\b[a-z]{2,}\b", text.lower()))


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    inter  = ta & tb
    union  = ta | tb
    return len(inter) / len(union) if union else 0.0


def _truncate_to_chars(text: str, max_chars: int) -> str:
    """Hard-truncate at max_chars, cutting at last whitespace if possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return (cut if cut else text[:max_chars]) + " [... truncated]"


def _format_chunk(chunk: dict, index: int) -> str:
    """
    Format a single chunk for inclusion in the context window.
    Includes a numbered header with source and score for traceability.
    """
    source = chunk.get("source", "unknown")
    score  = chunk.get("hybrid_score") or chunk.get("vector_score") or 0.0
    text   = chunk.get("text", "").strip()
    return (
        f"[Document {index} | source: {source} | relevance: {score:.3f}]\n"
        f"{text}"
    )


# ==============================================================================
# STRATEGY IMPLEMENTATIONS
# ==============================================================================

@dataclass
class ContextWindowManager:
    """
    Encapsulates all three context window management strategies.

    Parameters
    ----------
    max_chars   : hard cap on total context string (characters)
    min_score   : minimum hybrid/vector score to include a chunk
    top_k       : number of chunks to request from retriever
    strategy    : "truncation" | "ranking" | "mmr"
    """
    max_chars : int   = DEFAULT_MAX_CHARS
    min_score : float = DEFAULT_MIN_SCORE
    top_k     : int   = DEFAULT_TOP_K
    strategy  : str   = "ranking"          # default strategy

    # ------------------------------------------------------------------
    # Strategy 1:  Truncation
    # ------------------------------------------------------------------

    def truncation(self, chunks: list[dict]) -> tuple[str, dict]:
        """
        Strategy 1 -- Simple Truncation.

        Takes top-k chunks in retriever-score order, concatenates them
        with a separator, then hard-truncates to max_chars.

        Pros : Predictable length; fast.
        Cons : May slice mid-sentence at the truncation boundary;
               does NOT filter low-quality chunks.
        """
        used    = chunks[:self.top_k]
        parts   = [_format_chunk(c, i + 1) for i, c in enumerate(used)]
        joined  = "\n\n".join(parts)
        context = _truncate_to_chars(joined, self.max_chars)

        return context, {
            "strategy":       "truncation",
            "chunks_used":    len(used),
            "chars_before":   len(joined),
            "chars_after":    len(context),
            "truncated":      len(joined) > self.max_chars,
            "scores":         [c.get("hybrid_score") or c.get("vector_score") or 0.0
                               for c in used],
        }

    # ------------------------------------------------------------------
    # Strategy 2:  Score Ranking + Threshold Filter
    # ------------------------------------------------------------------

    def ranking(self, chunks: list[dict]) -> tuple[str, dict]:
        """
        Strategy 2 -- Score-Based Ranking with Threshold Filtering.

        Filters out chunks below min_score, then sorts remaining chunks
        by descending relevance score so the most useful content appears
        first in the context string.

        Design decision: Position bias in transformer attention means
        content earlier in the context tends to be weighted more heavily.
        Placing the highest-scoring chunks first maximises their influence
        on the generated response.

        Pros : Removes noise; highest-quality content at top.
        Cons : If all chunks fail the threshold, context is empty.
               Partially mitigated by a fallback to top-1 chunk.
        """
        scored = [
            c for c in chunks[:self.top_k]
            if (c.get("hybrid_score") or c.get("vector_score") or 0.0) >= self.min_score
        ]
        scored.sort(
            key=lambda c: c.get("hybrid_score") or c.get("vector_score") or 0.0,
            reverse=True,
        )

        # Fallback: if nothing passes the threshold, use top-1
        if not scored and chunks:
            scored = [chunks[0]]

        parts   = [_format_chunk(c, i + 1) for i, c in enumerate(scored)]
        joined  = "\n\n".join(parts)
        context = _truncate_to_chars(joined, self.max_chars)

        return context, {
            "strategy":       "ranking",
            "chunks_used":    len(scored),
            "chunks_filtered_out": (self.top_k - len(scored)),
            "min_score":      self.min_score,
            "chars_after":    len(context),
            "scores":         [c.get("hybrid_score") or c.get("vector_score") or 0.0
                               for c in scored],
        }

    # ------------------------------------------------------------------
    # Strategy 3:  MMR Diversity Filter
    # ------------------------------------------------------------------

    def mmr(self, chunks: list[dict]) -> tuple[str, dict]:
        """
        Strategy 3 -- Maximal Marginal Relevance (simplified).

        Iteratively selects chunks that are (a) above the score threshold
        AND (b) not near-duplicates of already-selected chunks (Jaccard
        similarity < MMR_SIM_THRESHOLD).

        Rationale: Retrieved chunks from a dense passage often overlap
        heavily (e.g., multiple budget paragraphs containing "GH cedi",
        "2025", "government"). Redundant context wastes the window
        budget without adding new information for the LLM.
        MMR enforces diversity.

        Pros : Maximises information density; avoids repetitive context.
        Cons : Greedy selection is O(n^2) in Jaccard comparisons -- fast
               for our small k, but does not guarantee global optimum.
        """
        candidates = [
            c for c in chunks[:self.top_k]
            if (c.get("hybrid_score") or c.get("vector_score") or 0.0) >= self.min_score
        ]
        candidates.sort(
            key=lambda c: c.get("hybrid_score") or c.get("vector_score") or 0.0,
            reverse=True,
        )

        selected: list[dict] = []
        duplicates_skipped   = 0

        for candidate in candidates:
            is_dup = any(
                _jaccard(candidate["text"], sel["text"]) >= MMR_SIM_THRESHOLD
                for sel in selected
            )
            if is_dup:
                duplicates_skipped += 1
                continue
            selected.append(candidate)

        # Fallback
        if not selected and chunks:
            selected = [chunks[0]]

        parts   = [_format_chunk(c, i + 1) for i, c in enumerate(selected)]
        joined  = "\n\n".join(parts)
        context = _truncate_to_chars(joined, self.max_chars)

        return context, {
            "strategy":          "mmr",
            "chunks_used":       len(selected),
            "duplicates_skipped": duplicates_skipped,
            "mmr_threshold":     MMR_SIM_THRESHOLD,
            "chars_after":       len(context),
            "scores":            [c.get("hybrid_score") or c.get("vector_score") or 0.0
                                  for c in selected],
        }

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def build_context(self, chunks: list[dict]) -> tuple[str, dict]:
        """
        Build context string using the configured strategy.

        Returns
        -------
        (context_string, metadata_dict)
        """
        if self.strategy == "truncation":
            return self.truncation(chunks)
        elif self.strategy == "ranking":
            return self.ranking(chunks)
        elif self.strategy == "mmr":
            return self.mmr(chunks)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. "
                             "Choose 'truncation', 'ranking', or 'mmr'.")


# ==============================================================================
# Convenience factory
# ==============================================================================

def make_manager(strategy: str = "ranking", **kwargs) -> ContextWindowManager:
    """Create a ContextWindowManager with the given strategy and options."""
    return ContextWindowManager(strategy=strategy, **kwargs)


# ==============================================================================
# Demo (run as script)
# ==============================================================================

if __name__ == "__main__":
    # Fake chunks for demonstration (real chunks come from HybridRetriever)
    demo_chunks = [
        {"source": "budget", "text": "Total Revenue and Grants for 2025 is projected at GH 224.9 billion.", "hybrid_score": 0.91},
        {"source": "budget", "text": "Total Revenue and Grants for 2025 is GH 224.9 billion, up from prior year.", "hybrid_score": 0.88},  # near-dup
        {"source": "budget", "text": "Inflation is targeted at 11.9 percent by end of 2025 under the IMF programme.", "hybrid_score": 0.75},
        {"source": "election", "text": "Year: 2020. Candidate: Akufo-Addo. Party: NPP. Votes: 6,730,413.", "hybrid_score": 0.45},
        {"source": "election", "text": "Year: 2016. Candidate: Akufo-Addo. Party: NPP. Votes: 5,716,026.", "hybrid_score": 0.35},
    ]

    print("=" * 65)
    print("  PART C -- STEP 2: CONTEXT WINDOW MANAGER DEMO")
    print("=" * 65)

    mgr = ContextWindowManager(max_chars=500, min_score=0.40, top_k=5)

    for strat in ("truncation", "ranking", "mmr"):
        mgr.strategy = strat
        ctx, meta = mgr.build_context(demo_chunks)
        print(f"\n  Strategy: {strat.upper()}")
        print(f"  Metadata: {meta}")
        print(f"  Context preview ({len(ctx)} chars):")
        print("  " + ctx[:300].replace("\n", "\n  ") + "...")
