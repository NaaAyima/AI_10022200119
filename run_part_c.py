"""
run_part_c.py -- Execute all Part C steps in sequence.
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Usage:
    python run_part_c.py
"""

import subprocess
import sys
from pathlib import Path

PART_C = Path(__file__).parent / "part_c"

STEPS = [
    ("Step 1: Prompt Templates Preview",    PART_C / "01_prompt_templates.py"),
    ("Step 2: Context Window Manager Demo", PART_C / "02_context_window_manager.py"),
    ("Step 3: Prompt Experiments (LLM)",    PART_C / "03_prompt_experiments.py"),
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
    print("\n[OK] Part C complete -- check data/processed/reports/ for results.\n")
