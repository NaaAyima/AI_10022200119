"""
run_part_d.py -- Execute the full RAG pipeline (Part D demo).
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Usage:
    python run_part_d.py                  # runs 5 demo queries
    python run_part_d.py --interactive    # interactive chat mode
    python run_part_d.py --query "..."    # single query
"""

import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).parent / "part_d" / "pipeline.py"

if __name__ == "__main__":
    # Pass all CLI args through to the pipeline script
    extra_args = sys.argv[1:]
    result = subprocess.run(
        [sys.executable, str(PIPELINE)] + extra_args,
        check=False,
    )
    sys.exit(result.returncode)
