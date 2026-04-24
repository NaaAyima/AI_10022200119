"""
Part A — Step 2: Chunking Strategy Implementation
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Three chunking strategies are implemented and compared:

┌─────────────────────┬────────────────────────────────────────────────────────┐
│ Strategy            │ Config                                                 │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Fixed-Size       │ 512 chars per chunk, 64-char overlap                   │
│ 2. Sentence-Based   │ 5 sentences per chunk, 1-sentence overlap              │
│ 3. Paragraph-Based  │ split on blank lines; merge <100 chars; cap at 1500    │
└─────────────────────┴────────────────────────────────────────────────────────┘

Justification summaries are embedded in each function's docstring.

Outputs (saved to data/processed/chunks/)
-----------------------------------------
  election_chunks.json
  budget_chunks.json
  all_chunks.json           — both datasets, all strategies, combined
"""

import json
import logging
import re
from pathlib import Path
from typing import Generator

import nltk

# Ensure required NLTK data is available on the machine
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CLEANED_DIR = ROOT / "data" / "processed" / "cleaned"
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — FIXED-SIZE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def fixed_size_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> Generator[str, None, None]:
    """
    Fixed-Size Chunking (character level).

    How it works
    ------------
    Slides a window of `chunk_size` characters over the full text, advancing
    by `chunk_size - overlap` characters each step so consecutive chunks share
    an `overlap`-character context zone.

    Chunk size (512 chars ≈ 80–120 words)
    --------------------------------------
    - Small enough to keep retrieved chunks focused and semantically tight.
    - Large enough that a standalone chunk is useful to an LLM without
      additional context.
    - Fits comfortably within the token budget when multiple chunks are
      concatenated into a prompt (Part C).

    Overlap (64 chars ≈ 10–15 words, ~12.5 %)
    -------------------------------------------
    - Prevents "boundary blindness": facts or sentences that straddle two
      chunks appear fully in at least one of them.
    - 12.5 % overlap is a standard empirical sweet-spot — large enough to
      bridge gaps, small enough not to double the chunk count.

    Trade-offs
    ----------
    ✔ Predictable, uniform chunk sizes — easy to reason about retrieval bias.
    ✔ O(n) time complexity.
    ✘ May split mid-sentence, producing grammatically incomplete chunks.
    ✘ Language-agnostic — ignores document structure entirely.
    """
    if not text:
        return
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            yield chunk
        if end >= len(text):
            break
        start += step


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — SENTENCE-BASED CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def sentence_based_chunks(
    text: str,
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1,
) -> Generator[str, None, None]:
    """
    Sentence-Based Chunking (NLTK Punkt tokeniser).

    How it works
    ------------
    Uses NLTK's trained Punkt sentence tokeniser to split the text into
    individual sentences, then groups `sentences_per_chunk` consecutive
    sentences into a chunk.  Consecutive chunks share `sentence_overlap`
    sentences of context.

    Sentences per chunk (5 sentences ≈ 100–300 words)
    ---------------------------------------------------
    - A single sentence is often too narrow for retrieval — it may lack
      the subject or predicate context that disambiguates its meaning.
    - Five sentences produce passages that are self-contained and coherent
      while remaining specific enough to match targeted queries.
    - Aligns with research showing 3–7 sentence windows maximise retrieval
      recall without exceeding LLM context windows.

    Sentence overlap (1 sentence)
    ------------------------------
    - Cross-boundary facts that begin in one chunk and conclude in the
      next are still fully captured in at least one chunk.
    - One sentence is sufficient for continuity without ballooning chunk
      counts.

    Trade-offs
    ----------
    ✔ Preserves grammatical and semantic integrity — every chunk is readable.
    ✔ Strong fit for the Budget PDF, which contains long policy sentences.
    ✘ Slower than fixed-size (requires tokenisation).
    ✘ NLTK Punkt may mis-tokenise on abbreviations ("No.", "Sec.", "Fig.")
      commonly found in government documents.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return

    step = max(1, sentences_per_chunk - sentence_overlap)
    start = 0
    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            yield chunk
        if end >= len(sentences):
            break
        start += step


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — PARAGRAPH-BASED CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def paragraph_based_chunks(
    text: str,
    min_chars: int = 100,
    max_chars: int = 1500,
) -> Generator[str, None, None]:
    """
    Paragraph-Based Chunking (document-structure aware).

    How it works
    ------------
    Splits the text on blank lines (double newlines) to produce raw
    paragraphs.  Short paragraphs (< min_chars) are merged into their
    successor.  Paragraphs that exceed max_chars are further subdivided at
    sentence boundaries to avoid oversized chunks.

    min_chars (100 chars ≈ 15–20 words)
    -------------------------------------
    - Avoids isolated headings, figure captions, or table labels being
      treated as standalone chunks — these fragments carry no retrievable
      context on their own.
    - Merging them into the following substantive paragraph preserves the
      heading–body relationship.

    max_chars (1500 chars ≈ 220–300 words)
    ----------------------------------------
    - Long sections of the Budget PDF (e.g., multi-paragraph fiscal
      analyses) would otherwise produce mono-chunks that are too diffuse
      for precise retrieval.
    - 1500 chars keeps chunks within a 350-token budget, comfortably below
      the typical 512-token embedding model limit.

    Trade-offs
    ----------
    ✔ Respects the author's own logical organisation — one paragraph ≈
      one idea, maximising topical coherence per chunk.
    ✔ Ideal for the Budget PDF: policy sections, tables of contents, and
      sector analyses are naturally paragraph-delimited.
    ✘ Variable chunk sizes complicate batching in Part B retrieval.
    ✘ PDF paragraph breaks are heuristic (relies on blank-line detection);
      some PDFs use single newlines for paragraphs, which this misses.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Accumulate short paragraphs
        if len(buffer) + len(para) < min_chars:
            buffer = (buffer + " " + para).strip()
            continue

        combined = (buffer + " " + para).strip() if buffer else para
        buffer = ""

        if len(combined) <= max_chars:
            yield combined
        else:
            # Subdivide at sentence boundaries
            sentences = nltk.sent_tokenize(combined)
            temp = ""
            for sent in sentences:
                if temp and len(temp) + len(sent) + 1 > max_chars:
                    yield temp.strip()
                    temp = sent
                else:
                    temp = (temp + " " + sent).strip()
            if temp:
                yield temp

    # Flush remaining buffer
    if buffer.strip():
        yield buffer.strip()


# ══════════════════════════════════════════════════════════════════════════════
# APPLY STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

# Strategy registry: (name, function, kwargs)
STRATEGIES = [
    ("fixed_size",       fixed_size_chunks,      {"chunk_size": 512, "overlap": 64}),
    ("sentence_based",   sentence_based_chunks,  {"sentences_per_chunk": 5, "sentence_overlap": 1}),
    ("paragraph_based",  paragraph_based_chunks, {"min_chars": 100, "max_chars": 1500}),
]


def apply_strategies(records: list[dict], dataset_label: str) -> dict[str, list[dict]]:
    """
    Apply all chunking strategies to a list of text records.

    The records are concatenated into a single corpus string so that
    chunking crosses record boundaries — this is important for the election
    CSV where each row is very short (20–80 chars) and individual rows would
    not form meaningful fixed-size or sentence-based chunks on their own.

    Returns
    -------
    dict keyed by strategy name, each value is a list of chunk dicts:
      { chunk_id, source, strategy, chunk_index, text, char_count }
    """
    corpus = "\n\n".join(r["text"] for r in records if r.get("text"))
    results: dict[str, list[dict]] = {}

    for strategy_name, fn, kwargs in STRATEGIES:
        chunks = []
        for i, chunk_text in enumerate(fn(corpus, **kwargs)):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "chunk_id": f"{dataset_label}__{strategy_name}__{i:05d}",
                "source": dataset_label,
                "strategy": strategy_name,
                "chunk_index": i,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            })
        results[strategy_name] = chunks
        logger.info(
            "  %-20s → %4d chunks  (avg %d chars)",
            strategy_name,
            len(chunks),
            int(sum(c["char_count"] for c in chunks) / max(len(chunks), 1)),
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load cleaned data ─────────────────────────────────────────────────────
    with open(CLEANED_DIR / "election_text_records.json", encoding="utf-8") as fh:
        election_records: list[dict] = json.load(fh)

    with open(CLEANED_DIR / "budget_pages.json", encoding="utf-8") as fh:
        budget_pages: list[dict] = json.load(fh)

    logger.info("Loaded %d election records and %d budget pages",
                len(election_records), len(budget_pages))

    # ── Apply chunking ────────────────────────────────────────────────────────
    logger.info("Chunking ELECTION dataset …")
    election_chunks = apply_strategies(election_records, "election")

    logger.info("Chunking BUDGET dataset …")
    budget_chunks = apply_strategies(budget_pages, "budget")

    # ── Save individual dataset files ─────────────────────────────────────────
    for fname, data in [
        ("election_chunks.json", election_chunks),
        ("budget_chunks.json",   budget_chunks),
    ]:
        path = CHUNKS_DIR / fname
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Saved → %s", path)

    # ── Save combined file ────────────────────────────────────────────────────
    combined = {"election": election_chunks, "budget": budget_chunks}
    combined_path = CHUNKS_DIR / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)
    logger.info("Saved combined → %s", combined_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PART A — STEP 2: CHUNKING — COMPLETE")
    print("═" * 65)
    for dataset_label, strategy_dict in [("ELECTION", election_chunks),
                                          ("BUDGET",   budget_chunks)]:
        print(f"\n  {dataset_label}")
        print(f"  {'Strategy':<22}  {'Chunks':>6}  {'Avg Chars':>10}  {'Min':>6}  {'Max':>6}")
        print("  " + "─" * 55)
        for name, chunks in strategy_dict.items():
            if chunks:
                chars = [c["char_count"] for c in chunks]
                print(
                    f"  {name:<22}  {len(chunks):>6}  "
                    f"{sum(chars)/len(chars):>10.0f}  "
                    f"{min(chars):>6}  {max(chars):>6}"
                )
    print(f"\n  Output directory: {CHUNKS_DIR}")
    print("═" * 65)


if __name__ == "__main__":
    main()
