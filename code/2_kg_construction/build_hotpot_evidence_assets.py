#!/usr/bin/env python3
"""Build self-contained Hotpot passage assets with stable evidence identity.

The unit is a Hotpot paragraph, not an industrial-standard clause.  ``clause_id``
is also stored in the shared evaluator's clause-identity field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
RAG_ROOT = CODE_ROOT / "4_RAG_method" / "SCT-RAG"
sys.path.insert(0, str(RAG_ROOT))
from api_embedding import APIEmbeddingModel


def evidence_id(source: str) -> str:
    return f"hotpot::{source}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entities(title: str, text: str) -> list[str]:
    # Dataset-independent, deterministic surface entities; gold supporting facts
    # and answers are deliberately not consulted.
    values = [title.strip()] if title.strip() else []
    values += re.findall(r"\b(?:[A-Z][\w'.-]*(?:\s+(?:of|the|and|de|[A-Z][\w'.-]*)){0,5})\b", text)
    return list(dict.fromkeys(x.strip(" ,.;:()") for x in values if len(x.strip()) >= 3))[:32]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--embedding-model", default="")
    args = p.parse_args()
    rows = json.loads(args.qa.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    chunks, text_units = [], {}
    entity_units: dict[str, set[str]] = defaultdict(set)
    normalized = []
    for item in rows:
        item = dict(item); qid = str(item["id"]); local = []
        for pos, raw in enumerate(item.get("chunks") or []):
            c = dict(raw); source = str(c.get("source") or f"{qid}_{c.get('id', pos)}")
            eid = evidence_id(source); text = c.get("chunk") or c.get("text") or ""
            c.update({"source": source, "passage_id": source, "clause_id": source, "evidence_id": eid})
            local.append(c)
            meta = {"dataset": "hotpotqa", "question_id": qid, "title": c.get("title", ""),
                    "file_name": source, "passage_id": source, "clause_id": source,
                    "evidence_id": eid, "global_index": len(chunks)}
            chunks.append({"chunk_id": source, "source": source, "context_text": text,
                           "original_text": text, "metadata": meta})
            names = entities(str(c.get("title") or ""), text)
            text_units[source] = {"id": source, "passage_id": source, "clause_id": source,
                                  "evidence_id": eid, "title": c.get("title", ""),
                                  "content": text, "entities": names}
            for name in names: entity_units[name].add(source)
        item["chunks"] = local
        facts = []
        for raw in item.get("supporting_facts") or []:
            f = dict(raw); source = str(f.get("source") or "")
            f.update({"passage_id": source, "clause_id": source, "evidence_id": evidence_id(source)})
            facts.append(f)
        item["supporting_facts"] = facts
        normalized.append(item)

    (args.output / "hotpotqa_50_evidence.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in chunks: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Emit streaming and array serializations from the same canonical records.
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    entity_map = {name: {"name": name, "text_unit_ids": sorted(units)}
                  for name, units in sorted(entity_units.items())}
    relations = {}; rid = 0
    # Co-occurrence edges are constructed from passage text only; no gold labels.
    for uid, unit in text_units.items():
        names = unit["entities"][:12]
        if not names: continue
        anchor = names[0]
        for other in names[1:]:
            relations[str(rid)] = {"source": anchor, "target": other,
                                   "relation_type": "co_occurs_in_passage",
                                   "text_unit_ids": [uid], "evidences": [unit["content"][:800]]}
            rid += 1
    # KG loader consumes row lists and builds keyed in-memory dictionaries.
    graph = {"schema_version": "hotpot-evidence-v1", "identity_unit": "evidence_id",
             "entities": list(entity_map.values()), "relations": list(relations.values()),
             "text_units": list(text_units.values()), "communities": []}
    (args.output / "hotpot_knowledge_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False), encoding="utf-8")

    vector_status = "not_requested"
    if args.embedding_model:
        import faiss
        model = APIEmbeddingModel(args.embedding_model)
        emb = np.asarray(model.encode([x["context_text"] for x in chunks], batch_size=64,
                                      show_progress_bar=True, normalize_embeddings=True), dtype="float32")
        index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
        faiss.write_index(index, str(args.output / "faiss.index"))
        np.save(args.output / "embeddings.npy", emb)
        vector_status = "complete"
    manifest = {"schema_version": "hotpot-evidence-v1", "source": str(args.qa.resolve()),
                "source_sha256": sha256(args.qa), "questions": len(rows), "passages": len(chunks),
                "unique_evidence_ids": len({x["metadata"]["evidence_id"] for x in chunks}),
                "entities": len(entity_map), "relations": len(relations),
                "gold_used_to_build_retrieval_assets": False, "vector_index": vector_status,
                "embedding_model": args.embedding_model or None}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
