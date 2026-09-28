# -*- coding: utf-8 -*-
"""HotPot 引用解析：chunk source id 与评测 GT 对齐。"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Any

HOTPOT_SRC_PAT = re.compile(r"\b([a-f0-9]{24}_\d+)\b", re.IGNORECASE)


def extract_hotpot_source_ids(text: str) -> List[str]:
    """从文本中提取 HotPot chunk source id 列表（去重保序）。"""
    if not text:
        return []
    seen = set()
    out = []
    for m in HOTPOT_SRC_PAT.finditer(text):
        sid = m.group(1)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def build_hotpot_context_index_map(rag_context) -> Dict[int, str]:
    """构建与 get_enhanced_context / 简化上下文一致的 [N] -> source id 映射。"""
    index_map: Dict[int, str] = {}
    idx = 1
    docs = rag_context.reranked_chunks if rag_context.reranked_chunks else rag_context.vector_chunks
    for chunk in (docs or [])[:12]:
        src = getattr(chunk, "source", None) or ""
        src = str(src).strip()
        if src:
            index_map[idx] = src
            idx += 1
    return index_map


def resolve_hotpot_citations(
    answer: str,
    raw_response: str,
    evidence_citations: Optional[List[str]],
    rag_context,
    evidence_block: str = "",
) -> List[str]:
    """
    合并 LLM 解析引用、[N] 序号映射、证据区 source id。
    用于评测 citation_llm_use。
    """
    citations: List[str] = []
    seen = set()

    def _add(items):
        for c in items or []:
            c = str(c).strip()
            if not c or c in seen:
                continue
            seen.add(c)
            citations.append(c)

    _add(evidence_citations)
    _add(extract_hotpot_source_ids(evidence_block))
    _add(extract_hotpot_source_ids(raw_response or ""))

    index_map = build_hotpot_context_index_map(rag_context)
    for num in re.findall(r"\[(\d+)\]", answer or ""):
        src = index_map.get(int(num))
        if src:
            _add([src])

    return citations


def build_hotpot_llm_context(rag_context, top_k: int = 10) -> str:
    """HotPot 生成专用上下文：局部段落（优先多实体覆盖）+ KG 三元组。"""
    chunks = list(rag_context.reranked_chunks or rag_context.vector_chunks or [])
    # 按 similarity / entity_cover 再排一次，保证桥接段靠前
    def _key(c):
        if isinstance(c, dict):
            return (
                float(c.get("similarity_score") or c.get("rerank_score") or 0),
                int((c.get("metadata") or {}).get("hotpot_entity_cover") or 0),
            )
        meta = getattr(c, "metadata", None) or {}
        return (
            float(getattr(c, "similarity_score", 0) or getattr(c, "rerank_score", 0) or 0),
            int(meta.get("hotpot_entity_cover") or 0),
        )

    chunks = sorted(chunks, key=_key, reverse=True)
    parts = [
        build_hotpot_ctx_from_passages(
            [_chunk_to_passage(c) for c in chunks],
            top_k=top_k,
        )
    ]
    # 追加 KG 证据（若有），帮助桥接实体
    kg = getattr(rag_context, "kg_results", None)
    triples = []
    if kg is not None:
        triples = getattr(kg, "triples", None) or []
        if not triples and isinstance(kg, dict):
            triples = kg.get("triples") or kg.get("ke_results") or []
    kg_lines = []
    for i, t in enumerate(triples[:12], 1):
        if isinstance(t, dict):
            h = t.get("head") or t.get("subject") or ""
            r = t.get("relation") or t.get("predicate") or ""
            tail = t.get("tail") or t.get("object") or ""
            src = t.get("source") or ""
            para = t.get("paragraph") or ""
            line = f"{h} --{r}--> {tail}".strip(" -")
            if src:
                line += f" (source: {src})"
            if para:
                line += f"\n   evidence: {str(para)[:240]}"
            if line.strip("- >"):
                kg_lines.append(f"K{i}. {line}")
        else:
            h = getattr(t, "head", "")
            r = getattr(t, "relation", "")
            tail = getattr(t, "tail", "")
            src = getattr(t, "source", "")
            para = getattr(t, "paragraph", "") or ""
            line = f"{h} --{r}--> {tail}".strip(" -")
            if src:
                line += f" (source: {src})"
            if para:
                line += f"\n   evidence: {str(para)[:240]}"
            s = line.strip()
            if s:
                kg_lines.append(f"K{i}. {s}")
    if kg_lines:
        parts.append("## Knowledge Graph Facts\n" + "\n".join(kg_lines))
    return "\n\n".join(p for p in parts if p) or "No relevant context."


def _chunk_to_passage(c) -> dict:
    """RetrievedChunk 或 dict 统一为 passage dict。"""
    if isinstance(c, dict):
        return c
    return {
        "source": getattr(c, "source", "") or "",
        "chunk_text": getattr(c, "chunk_text", None) or "",
    }


def build_hotpot_ctx_from_passages(passages: List[dict], top_k: int = 10) -> str:
    """基线方法 HotPot 上下文：带 [N] source id，便于 LLM 写引用。"""
    parts = []
    idx = 1
    for p in (passages or [])[:top_k]:
        src = (p.get("source") if isinstance(p, dict) else "") or ""
        text = (
            (p.get("chunk_text") or p.get("text") or p.get("text_preview") or "")
            if isinstance(p, dict)
            else ""
        ).strip()
        if not text:
            continue
        parts.append(f"[{idx}] source: {src}\n{text}\n---")
        idx += 1
    return "\n".join(parts) if parts else "No relevant context."


def extract_evidence_block(raw_response: str) -> str:
    """从 LLM 原始输出提取【证据/Evidence】区。"""
    if not raw_response:
        return ""
    m = re.search(r"【(?:证据|Evidence)】\s*(.*)", raw_response, re.DOTALL)
    return m.group(1).strip() if m else ""


def resolve_hotpot_citations_from_passages(
    answer: str,
    raw_response: str,
    evidence_citations: Optional[List[str]],
    passages: List[dict],
    evidence_block: str = "",
) -> List[str]:
    """基线方法：从 passage 列表解析 HotPot 引用。"""

    class _StubCtx:
        def __init__(self, ps):
            self.reranked_chunks = None
            self.vector_chunks = [_StubChunk(p) for p in ps]

    class _StubChunk:
        def __init__(self, p):
            self.source = p.get("source", "")

    block = evidence_block or extract_evidence_block(raw_response or "")
    return resolve_hotpot_citations(
        answer=answer or "",
        raw_response=raw_response or "",
        evidence_citations=evidence_citations,
        rag_context=_StubCtx(passages or []),
        evidence_block=block,
    )
