"""Lightweight input validation and reproducibility manifests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_formal_dataset(qa_file: Path, allow_unvalidated: bool = False) -> dict:
    """Validate the released QA schema without requiring restricted corpus text."""
    qa_file = qa_file.resolve()
    rows = json.loads(qa_file.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(rows, list) or not rows:
        errors.append("dataset must be a non-empty JSON list")
    else:
        ids = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"row {index} is not an object")
                continue
            if not str(row.get("question") or "").strip():
                errors.append(f"row {index} has no question")
            if not str(row.get("answer") or row.get("ground_truth") or "").strip():
                errors.append(f"row {index} has no reference answer")
            ids.append(str(row.get("id") or row.get("question_id") or index))
        # The locked test split requires one stable ID per evaluated item.
        # Training/dev may contain derived variants that intentionally share
        # a source ID, so uniqueness is not imposed on those files.
        if qa_file.stem == "test" and len(ids) != len(set(ids)):
            errors.append("test question identifiers are not unique")
    status = "pass" if not errors else "fail"
    if errors and not allow_unvalidated:
        raise RuntimeError("Dataset validation failed: " + "; ".join(errors[:10]))
    report = {
        "status": status if not allow_unvalidated else ("pass" if not errors else "bypassed"),
        "rows": len(rows) if isinstance(rows, list) else 0,
        "errors": errors,
        "qa_sha256": sha256_file(qa_file),
    }
    clause_path = os.environ.get("SCT_CLAUSE_RECORDS", "")
    if clause_path and Path(clause_path).exists():
        report["clause_records_sha256"] = sha256_file(Path(clause_path))
    return report


def write_run_manifest(out_dir: Path, qa_file: Path, gate: dict, settings: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"qa": {"path": str(qa_file.resolve()), "sha256": sha256_file(qa_file)}}
    clause_path = os.environ.get("SCT_CLAUSE_RECORDS", "")
    if clause_path and Path(clause_path).exists():
        path = Path(clause_path)
        inputs["clauses"] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "sct-run-manifest-v1",
        "formal_run": gate.get("status") == "pass",
        "input_validation": gate,
        "inputs": inputs,
        "settings": settings,
    }
    target = out_dir / "run_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
