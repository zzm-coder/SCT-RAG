"""对比方法共享工具：中文分词、检索结果归一化、上下文构建"""
from __future__ import annotations

import re
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List


def tokenize_zh(text: str) -> List[str]:
    """中文 BM25 分词：汉字单字 + 英文数字词（无需 jieba）"""
    if not text:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)
    return tokens if tokens else list(text.replace(" ", ""))


def load_clause_corpus(vector_db_path: str) -> List[Dict]:
    """Load the frozen clause corpus used by every industrial baseline."""
    path = Path(str(vector_db_path or "")) / "chunks.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else list(raw.values())
    out = []
    for i, row in enumerate(rows):
        meta = dict(row.get("metadata") or {})
        text = row.get("original_text") or row.get("context_text") or row.get("text") or ""
        if not str(text).strip():
            continue
        evidence_id = str(meta.get("evidence_id") or "")
        out.append({
            "id": evidence_id or str(row.get("id") or f"p{i}"),
            "filename": meta.get("file_name") or row.get("source") or f"chunk_{i}",
            "text": str(text),
            "metadata": meta,
        })
    return out


_CLAUSE_CORPUS_INDEX: Dict[str, tuple] = {}


def clause_corpus_indexes(vector_db_path: str) -> tuple:
    """Index canonical clauses by evidence ID and standard/clause identity."""
    key = str(vector_db_path or "")
    cached = _CLAUSE_CORPUS_INDEX.get(key)
    if cached is not None:
        return cached
    by_eid: Dict[str, Dict] = {}
    by_sc: Dict[tuple, Dict] = {}
    by_sc_norm: Dict[tuple, Dict] = {}
    for row in load_clause_corpus(key):
        meta = dict(row.get("metadata") or {})
        text = str(row.get("text") or "")
        eid = str(meta.get("evidence_id") or "").strip()
        sid = str(meta.get("standard_id") or "").strip()
        cid = str(meta.get("clause_id") or "").strip()
        if eid and len(text) >= len(str((by_eid.get(eid) or {}).get("text") or "")):
            by_eid[eid] = row
        if sid and cid:
            cur = by_sc.get((sid, cid))
            if cur is None or len(text) >= len(str(cur.get("text") or "")):
                by_sc[(sid, cid)] = row
            norm = (sid, re.sub(r"[.\s]", "", cid))
            cur_n = by_sc_norm.get(norm)
            if cur_n is None or len(text) >= len(str(cur_n.get("text") or "")):
                by_sc_norm[norm] = row
    _CLAUSE_CORPUS_INDEX[key] = (by_eid, by_sc, by_sc_norm)
    return _CLAUSE_CORPUS_INDEX[key]


def align_retrieved_to_corpus_clauses(chunks: List[Any], vector_db_path: str) -> List[Any]:
    """把检索块（含 KG 短句）还原为语料库同一 evidence_id 的条款全文与标准身份。"""
    if not chunks or not vector_db_path:
        return list(chunks or [])
    by_eid, by_sc, by_sc_norm = clause_corpus_indexes(vector_db_path)
    if not by_eid:
        return list(chunks)

    def _lookup(meta: Dict) -> Dict | None:
        eid = str(meta.get("evidence_id") or "").strip()
        if eid and eid in by_eid:
            return by_eid[eid]
        sid = str(meta.get("standard_id") or "").strip()
        cid = str(meta.get("clause_id") or "").strip()
        if sid and cid:
            row = by_sc.get((sid, cid))
            if row:
                return row
            return by_sc_norm.get((sid, re.sub(r"[.\s]", "", cid)))
        return None

    out: List[Any] = []
    for chunk in chunks:
        is_obj = hasattr(chunk, "chunk_text") and hasattr(chunk, "metadata")
        if is_obj:
            meta = dict(getattr(chunk, "metadata", None) or {})
        elif isinstance(chunk, dict):
            meta = dict(chunk.get("metadata") or {})
            if chunk.get("evidence_id") and not meta.get("evidence_id"):
                meta["evidence_id"] = chunk.get("evidence_id")
        else:
            out.append(chunk)
            continue
        row = _lookup(meta)
        if not row:
            out.append(chunk)
            continue
        rmeta = dict(row.get("metadata") or {})
        text = str(row.get("text") or "").strip()
        fname = str(row.get("filename") or rmeta.get("file_name") or "")
        merged_meta = dict(meta)
        merged_meta.update(rmeta)
        if is_obj:
            from data_types import RetrievedChunk

            out.append(
                RetrievedChunk(
                    chunk_id=str(rmeta.get("evidence_id") or getattr(chunk, "chunk_id", "")),
                    source=fname or str(getattr(chunk, "source", "") or ""),
                    chunk_text=text or str(getattr(chunk, "chunk_text", "") or ""),
                    metadata=merged_meta,
                    similarity_score=float(getattr(chunk, "similarity_score", 0.0) or 0.0),
                    rerank_score=float(getattr(chunk, "rerank_score", 0.0) or 0.0),
                    retrieval_source=str(
                        getattr(chunk, "retrieval_source", "") or "corpus_aligned_kg"
                    ),
                )
            )
        else:
            aligned = dict(chunk)
            aligned["chunk_text"] = text or aligned.get("chunk_text") or ""
            aligned["source"] = fname or aligned.get("source") or ""
            aligned["metadata"] = merged_meta
            aligned["evidence_id"] = rmeta.get("evidence_id") or aligned.get("evidence_id")
            aligned["standard_id"] = rmeta.get("standard_id") or aligned.get("standard_id")
            aligned["clause_id"] = rmeta.get("clause_id") or aligned.get("clause_id")
            aligned["chunk_id"] = aligned.get("evidence_id") or aligned.get("chunk_id")
            out.append(aligned)
    return out


def clause_ref_from_chunk(chunk: Any) -> Dict[str, str]:
    """从检索块取出可评测的条款身份（evidence_id / 标准号 / 条款号）。"""
    if chunk is None:
        return {}
    data = chunk.to_dict() if hasattr(chunk, "to_dict") else chunk
    if not isinstance(data, dict):
        meta = dict(getattr(chunk, "metadata", None) or {})
        data = {
            "metadata": meta,
            "evidence_id": getattr(chunk, "evidence_id", "") or meta.get("evidence_id", ""),
            "standard_id": getattr(chunk, "standard_id", "") or meta.get("standard_id", ""),
            "clause_id": getattr(chunk, "clause_id", "") or meta.get("clause_id", ""),
            "source": getattr(chunk, "source", ""),
        }
    meta = dict(data.get("metadata") or {})
    eid = str(
        meta.get("evidence_id") or data.get("evidence_id") or ""
    ).strip()
    if not eid:
        return {}
    return {
        "evidence_id": eid,
        "standard_id": str(meta.get("standard_id") or data.get("standard_id") or "").strip(),
        "clause_id": str(meta.get("clause_id") or data.get("clause_id") or "").strip(),
    }


def source_md_header(chunk: Any) -> str:
    """取出用于展示的 .md 文件名（无文件名时回退到来源字符串）。"""
    if chunk is None:
        return "unknown"
    data = chunk.to_dict() if hasattr(chunk, "to_dict") else chunk
    src = ""
    if isinstance(data, dict):
        src = str(data.get("source") or data.get("filename") or "")
    else:
        src = str(getattr(chunk, "source", "") or getattr(chunk, "filename", "") or "")
    m = re.search(r"([^\\/\s]+\.md)$", src, re.IGNORECASE)
    return m.group(1) if m else (src or "unknown")


def format_clause_identity_line(chunk: Any) -> str:
    """生成可直接抄写的条款身份行。"""
    ref = clause_ref_from_chunk(chunk)
    if not ref:
        return ""
    return (
        f"{ref['standard_id']}; Clause {ref['clause_id']}; Evidence {ref['evidence_id']}"
    )


def citation_scan_text(raw_response: str) -> str:
    """抽取【答案】正文；无【答案】标记时去掉【证据】区。"""
    raw = raw_response or ""
    ans_m = re.search(
        r"【(?:答案|Answer)】(.*?)(?=【(?:证据|Evidence)】|\Z)", raw, flags=re.S
    )
    if ans_m:
        return ans_m.group(1) or ""
    return re.split(r"【(?:证据|Evidence)】", raw, maxsplit=1)[0] or raw


def evidence_index_refs(
    raw_response: str, allowed: Dict[str, Dict[str, str]]
) -> Dict[int, Dict[str, str]]:
    """解析【证据】区「1. ... Evidence ev_xxx」到编号→条款。"""
    mapping: Dict[int, Dict[str, str]] = {}
    ev_m = re.search(r"【(?:证据|Evidence)】(.*)", raw_response or "", flags=re.S)
    if not ev_m:
        return mapping
    text = ev_m.group(1)
    for m in re.finditer(
        r"(?:^|[;；\n])\s*(\d+)\s*[\.、]\s*(.*?)(?=(?:[;；\n]\s*)\d+\s*[\.、]|\Z)",
        text,
        flags=re.S,
    ):
        idx = int(m.group(1))
        eids = re.findall(r"ev_[0-9a-f]{20}", m.group(2) or "", flags=re.I)
        for eid in eids:
            ref = allowed.get(eid.lower())
            if ref:
                mapping[idx] = ref
                break
    return mapping


def explicit_clause_citation_refs(raw_response: str, chunks: List[Dict]) -> List[Dict]:
    """Resolve emitted citations against canonical clauses in the context.

    Numbered citations map to the corresponding structured evidence entry.
    Explicit evidence identifiers are accepted only when they occur in the
    selected generation context; unsupported identifiers are discarded.
    """
    slots: List[Dict[str, str] | None] = []
    allowed: Dict[str, Dict[str, str]] = {}
    for chunk in chunks or []:
        ref = clause_ref_from_chunk(chunk)
        slots.append(ref or None)
        if ref:
            allowed[ref["evidence_id"].lower()] = {
                "evidence_id": ref["evidence_id"],
                "standard_id": ref["standard_id"],
                "clause_id": ref["clause_id"],
            }
    seen, refs = set(), []

    def _add(ref: Dict[str, str] | None) -> None:
        if not ref:
            return
        key = str(ref.get("evidence_id") or "").lower()
        if key in allowed and key not in seen:
            seen.add(key)
            refs.append(allowed[key])

    body = citation_scan_text(raw_response)
    cited_n = [int(x) for x in re.findall(r"\[(\d+)(?:\.\d+)?\]", body)]
    idx_map = evidence_index_refs(raw_response, allowed)
    if cited_n:
        for n in cited_n:
            if n in idx_map:
                _add(idx_map[n])
            elif 0 <= n - 1 < len(slots):
                _add(slots[n - 1])
        for value in re.findall(r"\bev_[0-9a-f]{20}\b", body, flags=re.I):
            _add(allowed.get(value.lower()))
        return refs

    found = re.findall(r"\bev_[0-9a-f]{20}\b", raw_response or "", flags=re.I)
    for value in found:
        _add(allowed.get(value.lower()))
    for n_s in re.findall(r"\[(\d+)(?:\.\d+)?\]", raw_response or ""):
        idx = int(n_s) - 1
        if 0 <= idx < len(slots):
            _add(slots[idx])
    return refs


def collect_retrieved_chunks(retrieval_data: Dict) -> List[Dict]:
    """从各对比方法的 retrieval 字段统一抽取实际供给生成器的证据。

    优先使用重排/向量/稀疏结果；只在这些字段均为空时，才将
    GraphRAG 的 ``ke_results -> paragrepa -> triples`` 展平成条款证据。
    这样既不会把 SCT-RAG 未供给生成器的 KG 候选混入 top-k，
    也不会将 GraphRAG 的有效检索误判为空。
    """
    if not retrieval_data:
        return []
    reranked = retrieval_data.get("reranked_results") or []
    if reranked:
        return [_as_dict(r) for r in reranked]

    for key in ("vector_results", "bm25_results", "combined_results"):
        items = retrieval_data.get(key) or []
        if items:
            return [_as_dict(r) for r in items]

    sparse = retrieval_data.get("sparse_results") or []
    dense = retrieval_data.get("dense_results") or []
    merged = [_as_dict(r) for r in (sparse + dense)]
    if merged:
        return merged
    return _collect_kg_clause_chunks(retrieval_data.get("kg_results") or {})


def _collect_kg_clause_chunks(kg_results: Dict) -> List[Dict]:
    """Flatten KG retrieval output while preserving rank and clause identity."""
    if not isinstance(kg_results, dict):
        return []
    groups = kg_results.get("ke_results") or kg_results.get("kg_results") or []
    chunks: List[Dict] = []
    seen = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        source = group.get("source") or ""
        paragraphs = group.get("paragrepa") or group.get("paragraphs") or []
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue
            text = paragraph.get("text") or paragraph.get("chunk_text") or ""
            triples = paragraph.get("triples") or []
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                evidence_id = str(triple.get("evidence_id") or "").strip()
                standard_id = str(triple.get("standard_id") or "").strip()
                clause_id = str(triple.get("clause_id") or "").strip()
                identity = evidence_id or (
                    f"{standard_id}::{clause_id}" if standard_id and clause_id else ""
                )
                # Anonymous triples cannot participate in clause-level Hit@K.
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                metadata = {
                    "evidence_id": evidence_id,
                    "standard_id": standard_id,
                    "clause_id": clause_id,
                }
                chunks.append({
                    "id": evidence_id or identity,
                    "evidence_id": evidence_id,
                    "standard_id": standard_id,
                    "clause_id": clause_id,
                    "source": source,
                    "chunk_text": text,
                    "metadata": metadata,
                    "triple": triple,
                })
    return chunks


def _as_dict(r: Any) -> Dict:
    if isinstance(r, dict):
        return r
    if hasattr(r, "to_dict"):
        return r.to_dict()
    meta = dict(getattr(r, "metadata", None) or {})
    return {
        "source": getattr(r, "source", ""),
        "chunk_text": getattr(r, "chunk_text", ""),
        "similarity_score": getattr(r, "similarity_score", 0.0),
        "metadata": meta,
        "evidence_id": meta.get("evidence_id", ""),
        "standard_id": meta.get("standard_id", ""),
        "clause_id": meta.get("clause_id", ""),
    }


def build_ctx_from_chunks(chunks: List[Dict], limit: int) -> str:
    """按检索顺序编号拼接上下文，每条写出 Evidence ev_，供答案 [n] 回映射。

    仍按 source(.md) 记录同文件多条款，但输出顺序与编号保持检索秩，避免 [n] 错位。
    """
    grouped = OrderedDict()
    ordered_blocks: List[str] = []
    n = 0
    for c in (chunks or [])[:limit]:
        header = source_md_header(c)
        text = (
            (c.get("chunk_text") if isinstance(c, dict) else None)
            or (c.get("text") if isinstance(c, dict) else None)
            or (c.get("text_preview") if isinstance(c, dict) else None)
            or getattr(c, "chunk_text", "")
            or ""
        )
        if not str(text).strip():
            continue
        n += 1
        ident = format_clause_identity_line(c)
        ident_line = f"条款身份: {ident}" if ident else ""
        block = f"[{n}] 来源: {header}"
        if ident:
            block += f" | {ident}"
        if ident_line:
            block += f"\n{ident_line}"
        block += f"\n{str(text).strip()}"
        grouped.setdefault(header, []).append(block)
        ordered_blocks.append(block)
    if not ordered_blocks:
        return ""
    return "\n\n".join(ordered_blocks).strip()


def baseline_router_stub(method_name: str) -> Dict:
    """基线方法路由占位（评估器可选读取）"""
    return {
        "type_id": 0,
        "question_type": "baseline",
        "entities": [],
        "intent": method_name,
        "metadata": {"method": method_name},
    }


def normalize_doc_key(name: str) -> str:
    """标准化文档名用于匹配：HB+8364-2013(2017).md ≈ HB 8364-2013.md"""
    if not name:
        return ""
    s = str(name).strip().lower()
    s = s.replace("+", "").replace(" ", "").replace("－", "-").replace("—", "-")
    s = re.sub(r"\(\d{4}\)", "", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff.\-_]", "", s)
    return s


def canonical_standard_id(name: str) -> str:
    """Return a strict comparable standard id.

    Examples:
    - GB/T 28878.1-2012 -> gbt:28878:1:2012
    - 28878_1-2012-gbt-e-300.md -> gbt:28878:1:2012
    - HB_Z+415-2014(2017).md -> hbz:415::2014
    """
    s = normalize_doc_key(name).replace(".md", "")
    if not s:
        return ""
    s = s.replace("_", "-")
    s = s.replace("hb-z", "hbz").replace("hb/z", "hbz")
    s = s.replace("gb-t", "gbt").replace("gb/t", "gbt")

    kind = ""
    if "hbz" in s:
        kind = "hbz"
    elif "gbt" in s or "-gbt" in s:
        kind = "gbt"
    elif "gjb" in s:
        kind = "gjb"
    elif "qj" in s:
        kind = "qj"
    elif re.search(r"\bhb\b", s) or s.startswith("hb"):
        kind = "hb"

    # Prefix-style names: hb8764-2025, gbt28878.1-2012, hbz415-2014.
    m = re.search(r"(hbz|gbt|gjb|qj|hb)(\d{2,6})(?:[.\-_](\d+))?-(\d{4})", s)
    if m:
        k, num, part, year = m.groups()
        return f"{k}:{num}:{part or ''}:{year}"

    # Suffix-style mineru names: 28878-1-2012-gbt-e-300.
    m = re.search(r"(\d{2,6})(?:[.\-_](\d+))?-(\d{4}).*?(gbt|gjb|qj|hbz|hb)", s)
    if m:
        num, part, year, k = m.groups()
        return f"{k}:{num}:{part or ''}:{year}"

    # HB file names sometimes keep kind as prefix after punctuation removal.
    m = re.search(r"(?:^|[^a-z])(hbz|gbt|gjb|qj|hb)[^\d]*(\d{2,6})(?:[.\-_](\d+))?[^\d]*(\d{4})", s)
    if m:
        k, num, part, year = m.groups()
        return f"{k}:{num}:{part or ''}:{year}"

    # Bare GB/T mineru names without visible suffix: keep them strict by year.
    m = re.search(r"(\d{2,6})(?:[.\-_](\d+))?-(\d{4})", s)
    if m and kind:
        num, part, year = m.groups()
        return f"{kind}:{num}:{part or ''}:{year}"
    return ""


def docs_match(a: str, b: str) -> bool:
    """判断两个文档标识是否指向同一标准文件"""
    ka, kb = normalize_doc_key(a), normalize_doc_key(b)
    if not ka or not kb:
        return False
    ca, cb = canonical_standard_id(a), canonical_standard_id(b)
    if ca and cb:
        return ca == cb
    if ka == kb:
        return True
    ka_base = ka.replace(".md", "")
    kb_base = kb.replace(".md", "")
    if ka_base == kb_base:
        return True
    # Avoid broad substring matches such as 28878 matching 28878.10.
    if min(len(ka_base), len(kb_base)) < 8:
        return False
    return ka_base in kb_base or kb_base in ka_base
