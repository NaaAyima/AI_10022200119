"""
Part A — Step 1: Data Engineering & Cleaning
Student: Jacqueline Naa Ayima Mensah | Index: 10022200119
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import requests
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "processed" / "cleaned"

for d in (RAW_DIR, CLEANED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset URLs ───────────────────────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/"
    "main/Ghana_Election_Result.csv"
)
PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def download_file(url: str, dest_path: Path) -> Path:
    """Download *url* to *dest_path*, skipping if the file already exists."""
    if dest_path.exists():
        logger.info("Already downloaded: %s", dest_path.name)
        return dest_path

    logger.info("Downloading  %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (RAG-Research; Academic City)"}

    with requests.get(url, headers=headers, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest_path, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))

    logger.info("Saved → %s", dest_path)
    return dest_path


# ══════════════════════════════════════════════════════════════════════════════
# CSV CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_string(value: str) -> str:
    """NFKC-normalise and strip non-printable control characters."""
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", value)
    return value.strip()


def clean_csv(csv_path: Path) -> pd.DataFrame:
    """
    Clean the Ghana Election Results CSV.

    Returns a cleaned DataFrame.  All operations are logged so the manual
    experiment log can be produced accurately.
    """
    logger.info("═" * 60)
    logger.info("CLEANING CSV: %s", csv_path.name)
    logger.info("═" * 60)

    # 1. Load with encoding fallback
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info("Loaded with UTF-8 encoding")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")
        logger.info("Loaded with latin-1 fallback encoding")

    logger.info("Raw shape: %d rows × %d columns", *df.shape)

    # 2. Normalise column names: lowercase, strip, replace spaces/hyphens
    df.columns = [
        re.sub(r"[\s\-]+", "_", c.strip().lower()) for c in df.columns
    ]
    logger.info("Columns after normalisation: %s", list(df.columns))

    # 3. Drop fully-empty rows
    before = len(df)
    df.dropna(how="all", inplace=True)
    logger.info("Dropped %d fully-empty rows", before - len(df))

    # 4. Drop exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Dropped %d duplicate rows", before - len(df))

    # 5. Strip whitespace & normalise string columns
    string_cols = df.select_dtypes(include="object").columns
    for col in string_cols:
        df[col] = df[col].astype(str).apply(_normalise_string)
        df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    # 6. Report remaining missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values after cleaning:\n%s", missing.to_string())
    else:
        logger.info("No missing values remaining.")

    logger.info("Final shape: %d rows × %d columns", *df.shape)
    return df


def csv_rows_to_text(df: pd.DataFrame) -> list[dict]:
    """
    Convert each CSV row into a natural-language text record.

    Design Decision
    ---------------
    Structured (tabular) data must be linearised before chunking so that a
    text embedding model can process it.  Each row is serialised as a
    comma-separated sequence of "Column: Value" pairs, which preserves all
    field semantics while producing sentences that an LLM can reason about.

    Example output:
        "Region: Greater Accra. Constituency: Ablekuma Central.
         NDC Votes: 12345. NPP Votes: 11800. Winner: NDC."
    """
    records = []
    for _, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if pd.notna(val):
                label = col.replace("_", " ").title()
                parts.append(f"{label}: {val}")
        if parts:
            text = ". ".join(parts) + "."
            records.append({
                "source": "Ghana_Election_Result.csv",
                "text": text,
                "metadata": {k: (None if pd.isna(v) else v) for k, v in row.items()},
            })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# PDF CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract and clean the 2025 Ghana Budget Statement PDF page-by-page.

    Cleaning pipeline per page:
      1. Unicode NFKC normalisation — fixes ligatures and fancy quotes.
      2. Hyphenated line-break repair — "govern-\\nment" → "government".
      3. Control character removal.
      4. Page number / header / footer heuristic removal:
           - Lines whose entire content is a number ≤ 4 digits (page nums).
           - Lines shorter than 5 chars in the first or last 3 lines of a page
             (running headers/footers).
      5. Collapse 3+ consecutive newlines to two (preserve paragraph breaks).
      6. Collapse multiple inline spaces to one.

    Returns a list of {source, page_num, text} dicts.
    """
    logger.info("═" * 60)
    logger.info("CLEANING PDF: %s", pdf_path.name)
    logger.info("═" * 60)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info("Total pages in PDF: %d", total_pages)

    pages = []
    skipped = 0

    for page_idx in range(total_pages):
        raw_text = doc[page_idx].get_text("text")

        if not raw_text.strip():
            skipped += 1
            continue

        # Step 1 — Unicode normalisation
        text = unicodedata.normalize("NFKC", raw_text)

        # Step 2 — Repair hyphenated line breaks (e.g. "infra-\nstructure")
        text = re.sub(r"-\n([a-z])", r"\1", text)

        # Step 3 — Remove control characters (keep \n and \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Step 4 — Header/footer heuristic removal
        lines = text.split("\n")
        n = len(lines)
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip pure page-number lines (1–4 digit numbers alone on a line)
            if re.fullmatch(r"\d{1,4}", stripped):
                continue
            # Skip very short lines at the very start or end of a page
            if len(stripped) < 5 and (i < 3 or i > n - 4):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Step 5 — Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Step 6 — Collapse inline spaces
        text = re.sub(r"[ \t]{2,}", " ", text)

        text = text.strip()

        if len(text) > 80:  # Skip near-empty pages (covers, blank separators)
            pages.append({
                "source": "2025-Budget-Statement.pdf",
                "page_num": page_idx + 1,
                "text": text,
            })
        else:
            skipped += 1

    doc.close()
    logger.info("Extracted %d pages  |  Skipped %d blank/near-empty pages",
                len(pages), skipped)
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Download ──────────────────────────────────────────────────────────────
    csv_raw = download_file(CSV_URL, RAW_DIR / "Ghana_Election_Result.csv")
    pdf_raw = download_file(PDF_URL, RAW_DIR / "2025-Budget-Statement.pdf")

    # ── Clean & Save CSV ──────────────────────────────────────────────────────
    df_clean = clean_csv(csv_raw)
    cleaned_csv_path = CLEANED_DIR / "Ghana_Election_cleaned.csv"
    df_clean.to_csv(cleaned_csv_path, index=False, encoding="utf-8")
    logger.info("Saved cleaned CSV → %s", cleaned_csv_path)

    election_records = csv_rows_to_text(df_clean)
    election_json_path = CLEANED_DIR / "election_text_records.json"
    with open(election_json_path, "w", encoding="utf-8") as fh:
        json.dump(election_records, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved %d election text records → %s",
                len(election_records), election_json_path)

    # ── Clean & Save PDF ──────────────────────────────────────────────────────
    budget_pages = clean_pdf(pdf_raw)
    budget_json_path = CLEANED_DIR / "budget_pages.json"
    with open(budget_json_path, "w", encoding="utf-8") as fh:
        json.dump(budget_pages, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d budget pages → %s", len(budget_pages), budget_json_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PART A — STEP 1: DATA CLEANING — COMPLETE")
    print("═" * 65)
    print(f"  Election CSV     : {len(df_clean)} rows × {df_clean.shape[1]} columns")
    print(f"  Election records : {len(election_records)} text records")
    print(f"  Budget PDF pages : {len(budget_pages)} non-empty pages")
    print(f"  Output directory : {CLEANED_DIR}")
    print("═" * 65)


if __name__ == "__main__":
    main()
