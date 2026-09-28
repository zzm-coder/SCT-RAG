#!/usr/bin/env python3
"""Build the canonical clause-level FAISS index used by SCT-RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import sys

RAG_ROOT = Path(__file__).resolve().parents[1] / "4_RAG_method" / "SCT-RAG"
sys.path.insert(0, str(RAG_ROOT))
from api_embedding import APIEmbeddingModel


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAUSES = (
    CODE_ROOT / "2_kg_construction" / "clause_records" / "clauses.jsonl"
)


def split_clause(text: str, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + max_chars // 2, end),
                text.rfind("。", start + max_chars // 2, end),
                text.rfind("；", start + max_chars // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_chunks(path: Path, max_chars: int, overlap: int) -> list[dict]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clause = json.loads(line)
        parts = split_clause(clause.get("raw_text", ""), max_chars=max_chars, overlap=overlap)
        for part_index, part in enumerate(parts):
            metadata = {
                "schema_version": clause.get("schema_version"),
                "standard_id": clause.get("standard_id"),
                "standard_version": clause.get("standard_version"),
                "clause_id": clause.get("clause_id"),
                "clause_title": clause.get("clause_title"),
                "clause_path": clause.get("clause_path"),
                "evidence_id": clause.get("evidence_id"),
                "file_name": clause.get("source_file"),
                "source_path": clause.get("source_path"),
                "content_hash": clause.get("content_hash"),
                "part_index": part_index,
                "part_count": len(parts),
            }
            chunks.append(
                {
                    "original_text": part,
                    "context_text": part,
                    "chunk_type": "clause_text",
                    "metadata": metadata,
                }
            )
    for index, chunk in enumerate(chunks):
        chunk["metadata"]["global_index"] = index
    return chunks


def build(args: argparse.Namespace) -> dict:
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty index: {output}")
    output.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.clauses, args.max_chars, args.overlap)
    texts = [chunk["context_text"] for chunk in chunks]
    model = APIEmbeddingModel(str(args.model))
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    np.save(output / "embeddings.npy", embeddings)
    faiss.write_index(index, str(output / "faiss.index"))
    (output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    input_hash = hashlib.sha256(args.clauses.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "sct-clause-vector-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clauses_path": str(args.clauses),
        "clauses_sha256": input_hash,
        "model_name": str(args.model),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized_embeddings": True,
        "faiss_index": "IndexFlatIP",
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "total_chunks": len(chunks),
        "unique_evidence_ids": len({c["metadata"]["evidence_id"] for c in chunks}),
    }
    (output / "config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clauses", type=Path, default=DEFAULT_CLAUSES)
    parser.add_argument("--model", required=True, help="Embedding API model identifier")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
