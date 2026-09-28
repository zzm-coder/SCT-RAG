"""Dataset and output paths for the public reproduction package."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from project_config import (  # noqa: E402
    HOTPOT_QA_50, INDSTD_QA_DEV, INDSTD_QA_RERANKER_TRAIN,
    INDSTD_QA_TEST, INDSTD_QA_TRAIN, QA_DATA_ROOT, RAG_RESULTS_DIR,
    SCT_RERANKER_MODEL,
)

DATASET_QA_MAP: dict[str, Path] = {
    "indstd": INDSTD_QA_TEST,
    "indstd_design_qa_clause": INDSTD_QA_TEST,
    "indstd-dev": INDSTD_QA_DEV,
    "hotpot": HOTPOT_QA_50,
}


def resolve_qa_path(dataset_key: str) -> Path:
    try:
        path = DATASET_QA_MAP[dataset_key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {dataset_key}") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def default_indstd_dataset() -> str:
    return "indstd"


def results_dir() -> Path:
    RAG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RAG_RESULTS_DIR


def method_log_dir(output_subdir: str) -> Path:
    path = results_dir() / output_subdir / "method_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
