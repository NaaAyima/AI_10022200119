"""
run_part_b.py -- Execute all Part B steps in sequence.
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Usage:
    python run_part_b.py
"""

import subprocess
import sys
from pathlib import Path

PART_B = Path(__file__).parent / "part_b"

STEPS = [
    ("Step 1: Embedding Pipeline",    PART_B / "01_embedding_pipeline.py"),
    ("Step 2: Retrieval System Demo", PART_B / "02_retrieval_system.py"),
    ("Step 3: Failure Analysis",      PART_B / "03_failure_analysis.py"),
]

def run(label: str, script: Path) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    for label, script in STEPS:
        run(label, script)
    print("\n[OK] Part B complete -- check data/processed/embeddings/ and reports/.\n")
