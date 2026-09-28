# -*- coding: utf-8 -*-
"""
HotPot distractor setting：仅在当前题目的 chunks（通常 10 段）内检索，避免跨题 FAISS 污染。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from data_types import QuestionType, RetrievedChunk, SystemConfig


def hotpot_evidence_id(source: str) -> str:
    """HotPot的证据单元是段落；使用稳定ID对齐工业集的clause identity。"""
    source = str(source or "").strip()
    if source.startswith("hotpot::"):
        return source
    return f"hotpot::{source}" if source else ""


def ensure_chunk_sources(qa_item: dict) -> List[dict]:
    """补齐source/passage_id/clause_id/evidence_id。"""
    chunks = list(qa_item.get("chunks") or [])
    qid = qa_item.get("id", "")
    for c in chunks:
        if not c.get("source"):
            c["source"] = f"{qid}_{c.get('id', 0)}"
        c.setdefault("passage_id", c["source"])
        # The shared evaluator stores the HotPot passage identity in clause_id.
        c.setdefault("clause_id", c["source"])
        c.setdefault("evidence_id", hotpot_evidence_id(c["source"]))
    return chunks


def get_local_chunks(qa_item: Optional[dict]) -> Optional[List[dict]]:
    if not qa_item:
        return None
    chunks = ensure_chunk_sources(qa_item)
    return chunks if chunks else None


def _chunk_text(c: dict) -> str:
    return (c.get("chunk") or c.get("text") or c.get("original_text") or "").strip()


def _english_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def bm25_rank_local(question: str, chunks: List[dict], top_k: int = 10) -> List[dict]:
    """在题目局部 chunks 上做 BM25（英文分词）。"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    corpus = [_chunk_text(c) for c in chunks]
    tokenized = [_english_tokens(t) for t in corpus]
    if not any(tokenized):
        return []
    q_tokens = _english_tokens(question)
    if not q_tokens:
        return []
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for idx, score in ranked:
        c = chunks[idx]
        out.append({
            "chunk_id": str(idx),
            "source": c.get("source", ""),
            "chunk_text": corpus[idx],
            "similarity_score": float(score),
            "retrieval_source": "local_bm25",
            "metadata": {
                "file_name": c.get("source", ""), "title": c.get("title", ""),
                "passage_id": c.get("passage_id", c.get("source", "")),
                "clause_id": c.get("clause_id", c.get("source", "")),
                "evidence_id": c.get("evidence_id", hotpot_evidence_id(c.get("source", ""))),
            },
        })
    return out


def dense_rank_local(
    question: str,
    chunks: List[dict],
    embedding_model,
    top_k: int = 10,
) -> List[RetrievedChunk]:
    """在题目局部 chunks 上做稠密向量相似度排序。"""
    texts = [_chunk_text(c) for c in chunks]
    if not texts:
        return []
    q_emb = embedding_model.encode([question])[0]
    d_emb = embedding_model.encode(texts)
    q = np.asarray(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q) or 1.0
    results: List[RetrievedChunk] = []
    scored = []
    for i, (c, text) in enumerate(zip(chunks, texts)):
        if not text:
            continue
        v = np.asarray(d_emb[i], dtype=np.float32)
        sim = float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) or 1.0)))
        scored.append((sim, i, c, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    for sim, i, c, text in scored[:top_k]:
        results.append(
            RetrievedChunk(
                chunk_id=str(i),
                source=c.get("source", f"local_{i}"),
                chunk_text=text,
                metadata={
                    "file_name": c.get("source", ""), "title": c.get("title", ""),
                    "passage_id": c.get("passage_id", c.get("source", "")),
                    "clause_id": c.get("clause_id", c.get("source", "")),
                    "evidence_id": c.get("evidence_id", hotpot_evidence_id(c.get("source", ""))),
                },
                similarity_score=sim,
                retrieval_source="local_dense",
            )
        )
    return results


def merge_local_hits(
    dense_hits: List[RetrievedChunk],
    bm25_hits: List[dict],
    top_k: int,
) -> List[RetrievedChunk]:
    """合并 dense + BM25 局部结果并去重。"""
    merged: List[RetrievedChunk] = list(dense_hits)
    seen = {(h.source, (h.chunk_text or "")[:200]) for h in merged}
    for h in bm25_hits:
        key = (h.get("source", ""), (h.get("chunk_text") or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            RetrievedChunk(
                chunk_id=h.get("chunk_id", ""),
                source=h.get("source", ""),
                chunk_text=h.get("chunk_text", ""),
                metadata=h.get("metadata") or {
                    "file_name": h.get("source", ""),
                    "passage_id": h.get("source", ""),
                    "clause_id": h.get("source", ""),
                    "evidence_id": hotpot_evidence_id(h.get("source", "")),
                },
                similarity_score=float(h.get("similarity_score", 0)),
                retrieval_source=h.get("retrieval_source", "local_bm25"),
            )
        )
    merged.sort(key=lambda x: x.similarity_score, reverse=True)
    return merged[:top_k]
