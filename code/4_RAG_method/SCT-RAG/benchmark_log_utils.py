# -*- coding: utf-8 -*-
"""Benchmark logging under each experiment's ``method_logs`` directory."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from benchmark_paths import method_log_dir, results_dir


def safe_method_name(method: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", method)


def method_jsonl_path(output_subdir: str, method: str) -> Path:
    return method_log_dir(output_subdir) / f"{safe_method_name(method)}.jsonl"


def method_text_log_path(output_subdir: str, method: str) -> Path:
    return method_log_dir(output_subdir) / f"{safe_method_name(method)}.log"


def resolve_run_jsonl(
    system: Any,
    method: str,
    output_subdir: str,
) -> Path:
    """解析当前方法应写入的 jsonl 路径（优先 system._run_log_jsonl）。"""
    custom = getattr(system, "_run_log_jsonl", None)
    if custom:
        return Path(custom)
    if output_subdir:
        return method_jsonl_path(output_subdir, method)
    return results_dir() / f"{safe_method_name(method)}.jsonl"


def append_run_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def bind_system_run_log(
    system: Any, jsonl_path: Path, output_subdir: str = "", method: str = ""
) -> None:
    """绑定单次 benchmark 的 jsonl 与 output_path。"""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(system, "_run_log_jsonl"):
        system._run_log_jsonl = jsonl_path
    setattr(system, "_output_subdir", output_subdir)
    setattr(system, "_run_method_name", method)
    if hasattr(system, "config") and output_subdir:
        system.config.output_path = str(results_dir() / output_subdir / "method_logs")
