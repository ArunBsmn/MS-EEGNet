"""Entry point for MS-EEGNet Phase-1 training (main_driver.py).

Edit RUN_CFG to select which tasks to train.  All tasks can be run
sequentially by listing them; each calls main_karaone.run(task).

Run from inside the Code/ directory:
    python main_driver.py
"""
from __future__ import annotations

import main_karaone

# ── Run configuration ─────────────────────────────────────────────────────────

RUN_CFG = {
    # Tasks to train.  Include "MC" plus any binary tasks needed for
    # Phase-2 saliency analysis.  Chapter 5 uses "B4" and "MC".
    "tasks": ["B1", "B2", "B3", "B4", "B5", "MC"],
}

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task in RUN_CFG["tasks"]:
        print(f"\n{'=' * 60}")
        print(f"  Phase 1 — Task: {task}")
        print(f"{'=' * 60}")
        main_karaone.run(task=task)