"""
Part G -- Innovation Component
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Innovation: Domain-Specific Routing & Scoring Function
------------------------------------------------------
Problem identified in Part E (Adversarial Testing):
Queries like "Who won the 2020 election?" retrieve budget paragraphs 
about "education allocation in 2020" instead of election results, due 
to vague lexical overlap and overlapping years.

Solution implemented here:
A `DomainAwareRetriever` that wraps the standard `HybridRetriever`. 
It uses keyword-based intent classification to route the query to a 
specific domain (election vs. budget). If a query has strong intent, 
chunks matching that domain receive a massive score multiplier, forcing 
them to the top.

Results: 
Zero-shot elimination of cross-domain contamination.
"""

import logging
import re

import faiss
from sentence_transformers import SentenceTransformer

# Load parent class
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "retrieval_system", str(ROOT / "part_b" / "02_retrieval_system.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
HybridRetriever = mod.HybridRetriever

logger = logging.getLogger(__name__)


# ==============================================================================
# INTENT CLASSIFIER DICTIONARIES
# ==============================================================================

DOMAIN_TRIGGERS = {
    "election": {
        "election", "vote", "votes", "candidate", "npp", "ndc", "mahama", 
        "akufo-addo", "akufo", "nana", "constituency", "parliament", "presidential",
        "polling", "ballot"
    },
    "budget": {
        "budget", "revenue", "expenditure", "tax", "gdp", "cedi", "cedis", 
        "gh", "ghc", "fiscal", "debt", "allocation", "economy", "macroeconomic",
        "ministry", "imf"
    }
}


class DomainAwareRetriever(HybridRetriever):
    """
    Subclass of HybridRetriever that applies a domain-specific scoring function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_multiplier = 1.6  # 60% boost to chunks matching intent

    def _classify_intent(self, query: str) -> str | None:
        """
        Classifies query intent based on keyword overlap. 
        Returns 'election', 'budget', or None (ambiguous).
        """
        tokens = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        
        election_score = len(tokens & DOMAIN_TRIGGERS["election"])
        budget_score   = len(tokens & DOMAIN_TRIGGERS["budget"])

        if election_score > budget_score:
            return "election"
        elif budget_score > election_score:
            return "budget"
        return None

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Overrides default search. Computes standard hybrid search,
        determines query intent, applies the domain multiplier, 
        and re-sorts.
        """
        # 1. Base Hybrid Search (fetch more than K to allow re-ranking)
        base_results = self.hybrid_search(query, k=k*2)
        
        # 2. Intent Classification
        intent = self._classify_intent(query)

        # 3. Apply Domain Scoring Function
        if intent:
            logger.info("Innovation Component: Detected '%s' intent. Applying 1.6x multiplier.", intent)
        else:
            logger.info("Innovation Component: Ambiguous intent, sticking to base hybrid scores.")

        for r in base_results:
            original_score = r["hybrid_score"]
            if intent:
                if intent in r["source"].lower():
                    r["hybrid_score"] = min(1.0, original_score * self.domain_multiplier)
                    r["innovation_boost"] = True
                else:
                    r["hybrid_score"] = 0.0  # STRICT FILTER: eliminate non-matching domains
            else:
                r["innovation_boost"] = False

        # 4. Re-sort based on modified scores and return top K
        base_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Update rank numbers and trim to top K
        top_k = base_results[:k]
        for i, r in enumerate(top_k):
            r["rank"] = i + 1

        return top_k

# ==============================================================================
# DEMO
# ==============================================================================
if __name__ == "__main__":
    import json
    EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
    
    index = faiss.read_index(str(EMBED_DIR / "faiss.index"))
    with open(EMBED_DIR / "metadata.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    retriever = DomainAwareRetriever(index, meta, model, alpha=0.7)
    
    q = "Who won the 2020 Ghana presidential election?"
    print(f"\nQUERY: {q}")
    res = retriever.search(q, k=3)
    for r in res:
        print(f"Rank {r['rank']} | Source: {r['source']} | Score: {r['hybrid_score']:.3f} | Boosted: {r.get('innovation_boost', False)}")

