#!/usr/bin/env python3
"""Build the entity index used by clause-linked KG traversal.

Entities are derived only from accepted source--clause--triple links. No
question-specific lexicon or manually encoded domain rule is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAG_ROOT = HERE.parent / "4_RAG_method" / "SCT-RAG"
sys.path.insert(0, str(RAG_ROOT))

from api_embedding import APIEmbeddingModel


def entity_id(name: str, entity_type: str) -> str:
    value = f"{entity_type.strip()}\0{name.strip()}".encode("utf-8")
    return "ent_" + hashlib.sha256(value).hexdigest()[:16]


def build(input_path: Path, output_dir: Path, model: str, batch_size: int) -> dict:
    triples_by_source = json.loads(input_path.read_text(encoding="utf-8"))
    entities: dict[str, dict] = {}
    for triples in triples_by_source.values():
        for triple in triples:
            if not triple.get("evidence_id"):
                continue
            for role in ("head", "tail"):
                value = triple.get(role) or {}
                if isinstance(value, str):
                    name, entity_type = value.strip(), ""
                else:
                    name = str(value.get("name") or "").strip()
                    entity_type = str(value.get("type") or "").strip()
                if not name:
                    continue
                key = entity_id(name, entity_type)
                row = entities.setdefault(
                    key,
                    {
                        "name": name,
                        "type": entity_type,
                        "evidence_ids": [],
                        "standard_ids": [],
                    },
                )
                if triple["evidence_id"] not in row["evidence_ids"]:
                    row["evidence_ids"].append(triple["evidence_id"])
                standard_id = str(triple.get("standard_id") or "")
                if standard_id and standard_id not in row["standard_ids"]:
                    row["standard_ids"].append(standard_id)

    ordered_ids = sorted(entities)
    names = [entities[key]["name"] for key in ordered_ids]
    embedding_model = APIEmbeddingModel(model)
    embeddings = embedding_model.encode(
        names,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "entity_cache.json").write_text(
        json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "entity_id_to_index.json").write_text(
        json.dumps({key: index for index, key in enumerate(ordered_ids)}, indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "entity_embeddings.npy", embeddings)
    manifest = {
        "schema_version": "sct-kg-entity-v1",
        "source": str(input_path),
        "embedding_model": model,
        "entity_count": len(ordered_ids),
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.size else 0,
    }
    (output_dir / "entity_index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "kg_clause_cache" / "triplet_clause_cache.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "kg_clause_cache"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.input, args.output_dir, args.model, args.batch_size),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
