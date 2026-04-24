"""
run_part_a.py -- Execute all Part A steps in sequence.
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Usage:
    python run_part_a.py
"""

import subprocess
import sys
from pathlib import Path

PART_A = Path(__file__).parent / "part_a"

STEPS = [
    ("Step 1 — Data Cleaning",             PART_A / "01_data_cleaning.py"),
    ("Step 2 — Chunking Strategies",       PART_A / "02_chunking.py"),
    ("Step 3 — Comparative Analysis",      PART_A / "03_comparative_analysis.py"),
]

def run(label: str, script: Path) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed. Fix errors above before continuing.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    for label, script in STEPS:
        run(label, script)
    print("\n✅  Part A complete — check data/processed/ for all outputs.\n")
