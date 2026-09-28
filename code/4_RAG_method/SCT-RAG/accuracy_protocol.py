"""Final answer-accuracy protocol used by the released experiments.

Industrial answer accuracy is the binary pass rate of the frozen answer judge.
The judge emits a score in [0, 1], and a response passes at the fixed threshold
of 0.7.
"""
from __future__ import annotations


def llm_judge_threshold() -> float:
    """Return the fixed judge threshold (0.7 in the paper)."""
    return 0.7


def llm_judge_pass(score: float | None, threshold: float | None = None) -> bool:
    """Convert a continuous judge score to the paper's binary accuracy."""
    if score is None:
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    if not 0.0 <= value <= 1.0:
        return False
    cutoff = float(threshold) if threshold is not None else llm_judge_threshold()
    return value >= cutoff
