# -*- coding: utf-8 -*-
"""FAISS dense retrieval and cross-encoder reranking."""
import sys
from pathlib import Path
import os

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import json
import time
import logging
import re
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import CrossEncoder
from api_embedding import APIEmbeddingModel

from data_types import QuestionType, SystemConfig, RetrievedChunk
from compare_rag.utils import docs_match, normalize_doc_key, canonical_standard_id

logger = logging.getLogger(__name__)
class VectorRetriever:
    """Dense retriever shared by WithinStd and comparison baselines."""

    def __init__(self, config: SystemConfig, dataset_class: str = None):
        self.config = config
        self.dataset_class = dataset_class
        try:
            if dataset_class == "ht":
                chosen = getattr(config, "vector_db_path1", None)
            else:
                chosen = getattr(config, "vector_db_path2", None)
            self.vector_db_path = Path(chosen) if chosen else Path(".")
        except Exception:
            self.vector_db_path = Path(".")

        try:
            if dataset_class == "ht":
                logger.info(f"加载语义模型: {config.semantic_model_path}")
                self.embedding_model = APIEmbeddingModel(config.semantic_model_path)
                self.cross_encoder = self._load_cross_encoder(config.cross_encoder_model)
            else:
                logger.info(f"加载语义模型: {config.semantic_model_path2}")
                self.embedding_model = APIEmbeddingModel(config.semantic_model_path2)
                self.cross_encoder = self._load_cross_encoder(config.cross_encoder_model2)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

        self.index = self._load_faiss_index()
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()
        self._compute_embeddings_norms()
        self.chunk_mapping = self._build_chunk_mapping()
        self._bm25 = None
        self._bm25_row_ids = []
        logger.info(
            f"向量检索器初始化完成，块数量: {len(self.chunks)}，FAISS索引大小: {self.index.ntotal}"
        )


    def _get_best_chunk_text(self, chunk_data: Dict) -> str:
        """Return the canonical text carried by a retrieved chunk."""
        if not chunk_data:
            return ""
        return (
            chunk_data.get("original_text")
            or chunk_data.get("chunk")  # HotPot QA 字段名
            or chunk_data.get("text")
            or chunk_data.get("chunk_text")
            or chunk_data.get("context_text")
            or chunk_data.get("context")
            or ""
        )

    def _load_cross_encoder(self, model_path: str):
        """加载 CrossEncoder；失败则回退为 None。"""
        if not model_path:
            logger.warning("未配置 cross_encoder_model，跳过重排序模型加载")
            return None
        try:
            logger.info(f"加载交叉编码器: {model_path}")
            return CrossEncoder(model_path, local_files_only=True)
        except Exception as e:
            logger.warning(f"CrossEncoder 加载失败，将使用向量相似度重排: {e}")
            return None

    def _load_faiss_index(self) -> faiss.Index:
        index_path = self.vector_db_path / "faiss.index"
        if not index_path.exists():
            index_files = list(self.vector_db_path.glob("*.index"))
            if index_files:
                index_path = index_files[0]
                logger.info(f"使用找到的索引文件: {index_path}")
            else:
                raise FileNotFoundError(f"FAISS索引不存在: {index_path}")
        try:
            index = faiss.read_index(str(index_path))
            logger.info(f"FAISS索引加载成功，维度: {index.d}, 文档数: {index.ntotal}")
            return index
        except Exception as e:
            logger.error(f"加载FAISS索引失败: {e}")
            raise

    def _load_chunks(self) -> Dict[str, Dict]:
        chunks_path = self.vector_db_path / "chunks.json"
        if not chunks_path.exists():
            chunk_files = list(self.vector_db_path.glob("*chunk*.json"))
            if chunk_files:
                chunks_path = chunk_files[0]
                logger.info(f"使用找到的块文件: {chunks_path}")
            else:
                logger.error(f"块文件不存在: {chunks_path}")
                return {}
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            if isinstance(chunks_data, list):
                logger.info(f"chunks.json 是列表格式，包含 {len(chunks_data)} 个元素")
                self.chunks_list = chunks_data
                return {str(i): chunk for i, chunk in enumerate(chunks_data)}
            if isinstance(chunks_data, dict):
                logger.info(f"加载文档块成功，数量: {len(chunks_data)}")
                return chunks_data
            logger.error(f"chunks.json 格式未知: {type(chunks_data)}")
            return {}
        except Exception as e:
            logger.error(f"加载文档块失败: {e}")
            return {}

    def _build_chunk_mapping(self) -> Dict[int, Dict]:
        mapping = {}
        if not isinstance(self.chunks, dict):
            logger.error(f"chunks 不是字典格式: {type(self.chunks)}")
            return mapping
        if hasattr(self, "chunks_list") and isinstance(self.chunks_list, list):
            for i, chunk in enumerate(self.chunks_list):
                mapping[i] = chunk
            logger.info(f"使用 chunks_list 建立按序映射，映射数量: {len(mapping)}")
            return mapping
        for chunk_id_str, chunk_data in self.chunks.items():
            try:
                mapping[int(chunk_id_str)] = chunk_data
            except ValueError:
                import re

                match = re.search(r"\d+", chunk_id_str)
                if match:
                    mapping[int(match.group())] = chunk_data
        logger.info(f"块映射建立完成，映射数量: {len(mapping)}")
        return mapping

    def _retrieve_local(
        self, question: str, local_chunks: List[dict], question_type: QuestionType = None
    ) -> List[RetrievedChunk]:
        """HotPot distractor：本题段落内 dense(+BM25) 检索，再可选 CrossEncoder 重排。"""
        if not local_chunks:
            return []
        # HotPot QA 用字段名 chunk；与 hotpot_local_retrieval._chunk_text 对齐
        texts = []
        for c in local_chunks:
            texts.append(self._get_best_chunk_text(c).strip())
        nonempty = [(i, t) for i, t in enumerate(texts) if t]
        if not nonempty:
            logger.warning(
                "局部检索：local_chunks=%d 但文本全空（请检查 chunk/chunk_text 字段）",
                len(local_chunks),
            )
            return []
        max_chunks = {
            QuestionType.SINGLE_STANDARD: 15,
            QuestionType.CROSS_STANDARD: 20,
            QuestionType.CORRELATION: 10,
        }.get(question_type, 20)
        try:
            # 复用已验证的 HotPot 局部 dense+BM25 融合，避免字段/排序不一致
            from hotpot_local_retrieval import bm25_rank_local, dense_rank_local, merge_local_hits

            dense_hits = dense_rank_local(
                question, local_chunks, self.embedding_model, top_k=max_chunks
            )
            bm25_hits = bm25_rank_local(question, local_chunks, top_k=max_chunks)
            out = merge_local_hits(dense_hits, bm25_hits, top_k=max_chunks)
            for h in out:
                if not getattr(h, "retrieval_source", None):
                    h.retrieval_source = "vector_local"
                elif h.retrieval_source == "local_dense":
                    h.retrieval_source = "vector_local"
            # 题干实体/标题命中加权，缓解 distractor 干扰
            out = self._boost_hotpot_local_by_question(question, out, local_chunks)
            # 仅当 CrossEncoder 可用且不像「ST 误转 CE」时才重排（避免随机分类头打乱排序）
            if self._hotpot_ce_usable() and out:
                out = self.rerank_chunks(question, out, top_k=max_chunks)
            return out
        except Exception as e:
            logger.warning(f"局部 dense+BM25 融合失败，回退纯稠密: {e}")
        try:
            idx_list = [i for i, _ in nonempty]
            enc_texts = [t for _, t in nonempty]
            q_emb = self.embedding_model.encode([question])[0].astype("float32")
            doc_embs = self.embedding_model.encode(enc_texts).astype("float32")
        except Exception as e:
            logger.warning(f"局部向量检索失败: {e}")
            return []
        q_norm = float(np.linalg.norm(q_emb) + 1e-8)
        scores = []
        for emb in doc_embs:
            d_norm = float(np.linalg.norm(emb) + 1e-8)
            scores.append(float(np.dot(q_emb, emb) / (q_norm * d_norm)))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        out = []
        for rank_i, sc in ranked[:max_chunks]:
            idx = idx_list[rank_i]
            c = local_chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=str(c.get("id", idx)),
                    source=str(c.get("source") or f"local_{idx}"),
                    chunk_text=texts[idx],
                    metadata=dict(c.get("metadata") or {}),
                    similarity_score=sc,
                    retrieval_source="vector_local",
                )
            )
        out = self._boost_hotpot_local_by_question(question, out, local_chunks)
        if self._hotpot_ce_usable() and out:
            out = self.rerank_chunks(question, out, top_k=max_chunks)
        return out

    def _boost_hotpot_local_by_question(
        self, question: str, hits: List[RetrievedChunk], local_chunks: List[dict]
    ) -> List[RetrievedChunk]:
        """按题干实体覆盖度提升分数：多跳题优先同时覆盖多个题干实体的段落。"""
        if not hits:
            return hits
        names = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b", question or "")
        # 保留带数字的专名，如 Scream 2
        names += re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+\d+)\b", question or "")
        names += [
            w
            for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", question or "")
            if w.lower()
            not in {
                "which", "what", "where", "when", "whose", "whom", "both", "were",
                "with", "from", "that", "this", "among", "after", "before", "about",
            }
        ]
        # 去重保序；优先保留更长专名
        names = sorted(set(names), key=lambda s: (-len(s), s.lower()))
        seen = set()
        ents = []
        for n in names:
            k = n.lower()
            # 若已被更长专名覆盖则跳过短词（避免 Scream 被 Scream 2 重复计）
            if any(k != sk and k in sk for sk in seen):
                continue
            if k in seen or len(k) < 3:
                continue
            seen.add(k)
            ents.append(n)
        title_by_src = {
            str(c.get("source") or ""): str(c.get("title") or "") for c in (local_chunks or [])
        }
        q_l = (question or "").lower()
        for h in hits:
            text = h.chunk_text or ""
            title = title_by_src.get(str(h.source or ""), "") or (h.metadata or {}).get("title", "")
            blob = f"{title}\n{text}"
            blob_l = blob.lower()
            bonus = 0.0
            cover = 0
            phrase_hits = 0
            for n in ents:
                nl = n.lower()
                if nl in blob_l:
                    cover += 1
                    w = 0.35 if (" " in nl or any(ch.isdigit() for ch in nl)) else 0.15
                    bonus += w
                    if " " in nl or any(ch.isdigit() for ch in nl):
                        phrase_hits += 1
                    if title and nl in title.lower():
                        bonus += 0.25
            if cover >= 2:
                bonus += 0.70 + 0.35 * phrase_hits
            elif cover == 1 and len(ents) >= 2:
                bonus += 0.05
            for tok in re.findall(r"[a-z0-9]{5,}", q_l):
                if tok in blob_l:
                    bonus += 0.01
            h.similarity_score = float(h.similarity_score or 0.0) + min(bonus, 2.5)
            try:
                h.metadata = dict(h.metadata or {})
                h.metadata["hotpot_entity_cover"] = cover
                h.metadata["hotpot_phrase_hits"] = phrase_hits
            except Exception:
                pass
        hits.sort(
            key=lambda x: (
                x.similarity_score,
                (x.metadata or {}).get("hotpot_phrase_hits", 0),
                (x.metadata or {}).get("hotpot_entity_cover", 0),
            ),
            reverse=True,
        )
        return hits

    def _hotpot_ce_usable(self) -> bool:
        """HotPot 局部重排：避免把 SentenceTransformer 误当 CrossEncoder。"""
        if self.cross_encoder is None:
            return False
        name = str(
            getattr(self.config, "cross_encoder_model2", "")
            or getattr(self.config, "cross_encoder_model", "")
            or ""
        ).lower()
        # msmarco-*-cos-v5 等是双塔模型，强行转 CE 会 MISSING classifier
        if "cos-v5" in name or "sentence-transformers/msmarco-minilm-l6-cos" in name:
            return False
        return True

    @staticmethod
    def _bm25_tokens(text: str) -> List[str]:
        """中文单字 + 英文/数字词；不依赖分词词典，保证实验可复现。"""
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+(?:\.\d+)*", str(text or "").lower())

    @staticmethod
    def _evidence_key(chunk: RetrievedChunk) -> str:
        meta = getattr(chunk, "metadata", None) or {}
        evidence_id = meta.get("evidence_id")
        if evidence_id:
            return f"evidence::{evidence_id}"
        standard_id = meta.get("standard_id")
        clause_id = meta.get("clause_id")
        if standard_id and clause_id:
            return f"clause::{standard_id}::{clause_id}"
        return f"chunk::{getattr(chunk, 'chunk_id', '')}"

    def _ensure_bm25(self) -> bool:
        if self._bm25 is False:
            return False
        if self._bm25 is not None:
            return True
        try:
            from rank_bm25 import BM25Okapi

            rows = []
            row_ids = []
            for idx in sorted(self.chunk_mapping):
                text = self._get_best_chunk_text(self.chunk_mapping[idx])
                tokens = self._bm25_tokens(text)
                if not tokens:
                    continue
                rows.append(tokens)
                row_ids.append(idx)
            self._bm25 = BM25Okapi(rows)
            self._bm25_row_ids = row_ids
            logger.info("BM25 索引建立完成，块数: %d", len(row_ids))
            return True
        except Exception as e:
            logger.error("BM25 索引建立失败: %s", e)
            self._bm25 = False
            return False

    def retrieve_bm25(self, question: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """条款块稀疏检索；返回的身份信息与稠密通道完全一致。"""
        if not self._ensure_bm25():
            return []
        tokens = self._bm25_tokens(question)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        limit = max(1, int(top_k or getattr(self.config, "top_k_vector", 20) or 20))
        candidate_n = min(len(scores), limit * 3)
        if candidate_n == 0:
            return []
        order = np.argsort(-scores, kind="stable")[:candidate_n]
        out = []
        for pos in order:
            score = float(scores[int(pos)])
            if score <= 0:
                continue
            idx = self._bm25_row_ids[int(pos)]
            chunk_data = self.chunk_mapping[idx]
            meta = dict(chunk_data.get("metadata", {}) or {})
            meta["retrieval_channels"] = ["bm25"]
            out.append(
                RetrievedChunk(
                    chunk_id=str(idx),
                    source=meta.get("file_name", f"chunk_{idx}"),
                    chunk_text=self._get_best_chunk_text(chunk_data),
                    metadata=meta,
                    similarity_score=score,
                    retrieval_source="bm25",
                )
            )
        return out[:limit]

    def retrieve(
        self,
        question: str,
        entities: List[str],
        kg_entities: List[str] = None,
        question_type: QuestionType = None,
        local_chunks: List[dict] = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> List[RetrievedChunk]:
        """按显式开关执行 Dense/BM25，以 RRF 融合并按条款证据去重。"""
        # HotPot 保留既有的本题局部融合逻辑。
        if local_chunks:
            return self._retrieve_dense(question, entities, kg_entities, question_type, local_chunks)
        dense = self._retrieve_dense(question, entities, kg_entities, question_type) if use_dense else []
        sparse = self.retrieve_bm25(question) if use_bm25 else []
        if not dense:
            return sparse
        if not sparse:
            for chunk in dense:
                meta = dict(chunk.metadata or {})
                meta["retrieval_channels"] = ["dense"]
                chunk.metadata = meta
            return dense

        fused = {}
        rrf_k = 60.0
        for channel, hits in (("dense", dense), ("bm25", sparse)):
            for rank, chunk in enumerate(hits, start=1):
                key = self._evidence_key(chunk)
                if key not in fused:
                    fused[key] = {"chunk": chunk, "score": 0.0, "channels": []}
                fused[key]["score"] += 1.0 / (rrf_k + rank)
                if channel not in fused[key]["channels"]:
                    fused[key]["channels"].append(channel)
        ranked = sorted(fused.values(), key=lambda row: row["score"], reverse=True)
        limit = int(getattr(self.config, "top_k_vector", 20) or 20)
        out = []
        for row in ranked[:limit]:
            chunk = row["chunk"]
            chunk.similarity_score = float(row["score"])
            chunk.retrieval_source = "+".join(row["channels"])
            meta = dict(chunk.metadata or {})
            meta["retrieval_channels"] = list(row["channels"])
            chunk.metadata = meta
            out.append(chunk)
        return out

    def _retrieve_dense(
        self,
        question: str,
        entities: List[str],
        kg_entities: List[str] = None,
        question_type: QuestionType = None,
        local_chunks: List[dict] = None,
    ) -> List[RetrievedChunk]:
        """单独的稠密检索通道。"""
        if local_chunks:
            return self._retrieve_local(question, local_chunks, question_type)

        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS索引为空，无法进行检索")
                return []

            # The query contains the original question and routed/KG entities.
            router_entities = [str(e).strip() for e in (entities or []) if str(e).strip()]
            kg_ents = [str(e).strip() for e in (kg_entities or []) if str(e).strip()]
            enhanced_query = self._build_enhanced_query(
                question, [], router_entities, kg_ents
            )
            logger.info(f"原始查询: '{question[:50]}...'")
            logger.info(f"增强查询: '{enhanced_query[:100]}...'")

            query_embedding = self.embedding_model.encode([enhanced_query])[0]
            query_embedding = np.array([query_embedding]).astype("float32")

            top_k = min(self.config.top_k_vector * 3, self.index.ntotal)
            if top_k == 0:
                return []

            _, indices = self.index.search(query_embedding, top_k)
            retrieved_chunks = []
            q_vec = np.asarray(query_embedding).reshape(-1)
            q_norm = np.linalg.norm(q_vec)

            for idx in indices[0]:
                if idx < 0 or idx not in self.chunk_mapping:
                    continue
                chunk_data = self.chunk_mapping[idx]
                try:
                    if self.embeddings is None or idx >= self.embeddings.shape[0]:
                        similarity = 0.0
                    else:
                        emb_vec = self.embeddings[idx]
                        emb_norm = (
                            self.embeddings_norms[idx]
                            if getattr(self, "embeddings_norms", None) is not None
                            else np.linalg.norm(emb_vec)
                        )
                        if q_norm == 0 or emb_norm == 0:
                            similarity = 0.0
                        else:
                            similarity = float(np.dot(q_vec, emb_vec) / (q_norm * emb_norm))
                except Exception:
                    similarity = 0.0

                entity_boost = self._calculate_entity_boost(
                    chunk_data, [], router_entities, kg_ents
                )
                enhanced_similarity = similarity * (1.0 + entity_boost)
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=str(idx),
                        source=chunk_data.get("metadata", {}).get(
                            "file_name", f"chunk_{idx}"
                        ),
                        chunk_text=self._get_best_chunk_text(chunk_data),
                        metadata=chunk_data.get("metadata", {}),
                        similarity_score=enhanced_similarity,
                        retrieval_source="vector",
                    )
                )

            retrieved_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
            filtered_chunks = [
                chunk
                for chunk in retrieved_chunks[: self.config.top_k_vector]
                if chunk.similarity_score >= self.config.similarity_threshold
            ]
            max_chunks = {
                QuestionType.SINGLE_STANDARD: 15,
                QuestionType.CROSS_STANDARD: 20,
                QuestionType.CORRELATION: 20,
            }.get(question_type, 20)
            filtered_chunks = filtered_chunks[:max_chunks]
            logger.info(
                f"向量检索完成: 找到 {len(filtered_chunks)} 个相关块，"
                f"最佳相似度: {filtered_chunks[0].similarity_score if filtered_chunks else 0:.3f}"
            )
            return filtered_chunks
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _build_enhanced_query(
        self,
        question: str,
        keywords: List[str],
        entities: List[str],
        kg_entities: List[str] = None,
    ) -> str:
        """Build a query from the question and routed or KG entities."""
        parts = [question]
        _ = keywords
        if entities:
            parts.extend([str(e) for e in entities[:5]])
        if kg_entities:
            parts.extend([str(e) for e in kg_entities[:5]])
        return " ".join(parts)

    def _calculate_entity_boost(
        self,
        chunk_data: Dict,
        keywords: List[str],
        entities: List[str],
        kg_entities: List[str] = None,
    ) -> float:
        """Apply a small score adjustment for routed or KG entity matches."""
        boost = 0.0
        if not chunk_data:
            return boost
        chunk_text = self._get_best_chunk_text(chunk_data).lower()
        for entity in (entities or [])[:3]:
            ent = str(entity or "").lower()
            if ent and len(ent) >= 2 and ent in chunk_text:
                boost += 0.15
        if kg_entities:
            for kg_entity in kg_entities[:3]:
                ke = str(kg_entity or "").lower()
                if ke and len(ke) >= 2 and ke in chunk_text:
                    boost += 0.2
        return min(boost, 0.5)

    def rerank_chunks(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        top_k: int = None,
    ) -> List[RetrievedChunk]:
        """使用交叉编码器重排序；无模型时回退到 similarity_score。"""
        if not chunks:
            return []
        limit = int(top_k) if top_k is not None else int(self.config.top_k_vector or 20)
        if self.cross_encoder is None:
            reranked = sorted(chunks, key=lambda x: x.similarity_score, reverse=True)
            return reranked[:limit]
        try:
            pairs = []
            for chunk in chunks:
                meta = dict(getattr(chunk, "metadata", None) or {})
                serialized = (
                    f"Source: {getattr(chunk, 'source', '') or meta.get('standard_id', '')}\n"
                    f"Clause: {meta.get('clause_id', '')}\n"
                    f"Text: {(chunk.chunk_text or '')[:800]}"
                )
                pairs.append([question, serialized])
            scores = self.cross_encoder.predict(pairs)
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(scores[i])
            reranked = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
            logger.info(
                f"重排序完成: 最佳重排序分数: {reranked[0].rerank_score if reranked else 0:.3f}"
            )
            return reranked[:limit]
        except Exception as e:
            logger.error(f"重排序失败，回退到向量分数: {e}")
            reranked = sorted(chunks, key=lambda x: x.similarity_score, reverse=True)
            return reranked[:limit]

    def find_chunks_by_standard(self, std_keys: List[str], limit: int = 12) -> List[RetrievedChunk]:
        """按标准号文件名定向取块；排序交给后续 CrossEncoder，此处不做题干词加减分。"""
        if not std_keys or not self.chunk_mapping:
            return []
        hits: List[RetrievedChunk] = []
        for idx, chunk_data in self.chunk_mapping.items():
            meta = chunk_data.get("metadata", {}) or {}
            fname = meta.get("file_name") or meta.get("source") or chunk_data.get("source") or ""
            fname_str = str(fname)
            if not self._source_matches_any(fname_str, std_keys):
                continue
            text = self._get_best_chunk_text(chunk_data)
            compact = "".join((text or "").split())
            if len(compact) < 20:
                continue
            # 仅过滤封面噪声，不按约束词/焦点词打分
            if any(m in compact for m in ("目次", "目录", "书号", "定价", "北京1665信箱")):
                continue
            hits.append(
                RetrievedChunk(
                    chunk_id=str(idx),
                    source=fname_str,
                    chunk_text=text,
                    metadata=meta,
                    similarity_score=1.0,  # 占位；语义排序由 CrossEncoder 完成
                    retrieval_source="standard_inject",
                )
            )
        # 注入阶段只按长度粗排，精排交给 CE + RerankBoost
        hits.sort(key=lambda x: len((x.chunk_text or "").strip()), reverse=True)
        return hits[:limit]

    def find_chunks_by_standard_clause(
        self, references: List[tuple[str, str]], limit: int = 12
    ) -> List[RetrievedChunk]:
        """精确 StdDirect：仅返回题干显式绑定的（标准号，条款号）。"""
        if not references or not self.chunk_mapping:
            return []

        def norm_clause(value: str) -> str:
            return re.sub(r"\s+", "", str(value or "")).lower().strip("第条.;；")

        wanted = [(std, norm_clause(clause)) for std, clause in references if norm_clause(clause)]
        hits: List[RetrievedChunk] = []
        for idx, chunk_data in self.chunk_mapping.items():
            meta = chunk_data.get("metadata", {}) or {}
            fname = str(meta.get("file_name") or meta.get("source") or chunk_data.get("source") or "")
            clause_id = norm_clause(meta.get("clause_id"))
            if not clause_id:
                continue
            if not any(
                self._source_matches_any(fname, [std]) and clause_id == target_clause
                for std, target_clause in wanted
            ):
                continue
            hits.append(
                RetrievedChunk(
                    chunk_id=str(idx), source=fname,
                    chunk_text=self._get_best_chunk_text(chunk_data), metadata=meta,
                    similarity_score=1.0, retrieval_source="std_direct_clause",
                )
            )
        hits.sort(key=lambda c: (str((c.metadata or {}).get("standard_id", "")), str((c.metadata or {}).get("clause_id", ""))))
        return hits[:limit]

    def dense_retrieve_within_standards(
        self,
        question: str,
        std_keys: List[str],
        limit: int = 36,
        sentence_level: bool = True,
        sentence_neighbor_window: int = 0,
    ) -> List[RetrievedChunk]:
        """同标准内二次稠密检索：仅在命中标准号的文件块上按问题向量排序（非题干词扫库）。

        sentence_level=True 时：块内切句后再按句向量与问题相似度排序，提高短条款召回。
        """
        if not question or not std_keys or not self.chunk_mapping:
            return []
        if self.embeddings is None or self.embedding_model is None:
            # 无预计算向量时退化为文件内截断取块
            return self.find_chunks_by_standard(std_keys, limit=limit)

        candidates = []  # (idx, fname, text, meta)
        for idx, chunk_data in self.chunk_mapping.items():
            try:
                iidx = int(idx)
            except Exception:
                continue
            if iidx < 0 or iidx >= len(self.embeddings):
                continue
            meta = chunk_data.get("metadata", {}) or {}
            fname = meta.get("file_name") or meta.get("source") or chunk_data.get("source") or ""
            fname_str = str(fname)
            if not self._source_matches_any(fname_str, std_keys):
                continue
            text = self._get_best_chunk_text(chunk_data)
            compact = "".join((text or "").split())
            if len(compact) < 24:
                continue
            if any(m in compact[:80] for m in ("目次", "目录", "书号", "定价", "规范性引用文件")):
                continue
            candidates.append((iidx, fname_str, text, meta))
        if not candidates:
            return []

        try:
            # 先按块向量取同标准候选池，再可选切句精排
            q_emb = self.embedding_model.encode([question], normalize_embeddings=True)
            q = np.asarray(q_emb[0], dtype=np.float32)
            q_norm = float(np.linalg.norm(q) + 1e-8)
            scored_chunks = []
            for iidx, fname_str, text, meta in candidates:
                v = np.asarray(self.embeddings[iidx], dtype=np.float32)
                denom = float(np.linalg.norm(v) * q_norm + 1e-8)
                sim = float(np.dot(v, q) / denom)
                scored_chunks.append((sim, iidx, fname_str, text, meta))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            # 同标准块较少时尽量全量进句级池，避免金标块被块级预筛丢掉
            pool_n = len(scored_chunks) if len(scored_chunks) <= 160 else max(int(limit) * 3, 96)
            top_chunks = scored_chunks[: min(pool_n, len(scored_chunks))]

            if not sentence_level:
                out = [
                    RetrievedChunk(
                        chunk_id=str(iidx),
                        source=fname_str,
                        chunk_text=text,
                        metadata=meta,
                        similarity_score=sim,
                        retrieval_source="within_std_dense",
                    )
                    for sim, iidx, fname_str, text, meta in top_chunks[: max(1, int(limit))]
                ]
            else:
                # 句级：对池内块切句，批量编码后按与问题相似度排序
                try:
                    from standard_boost import split_evidence_sentences
                except Exception:
                    split_evidence_sentences = None
                sent_items = []  # (parent_idx, fname, sent, meta, parent_sim, sent_idx, parent_text)
                for sim, iidx, fname_str, text, meta in top_chunks:
                    sents = []
                    if split_evidence_sentences is not None:
                        sents = split_evidence_sentences(text) or []
                    if not sents:
                        compact = "".join((text or "").split())
                        if 12 <= len(compact) <= 900:
                            sents = [text.strip()]
                    for si, sent in enumerate(sents[:12]):
                        sent_items.append((iidx, fname_str, sent, meta, float(sim), si, text))
                if not sent_items:
                    out = []
                else:
                    sent_texts = [t for _, _, t, _, _, _, _ in sent_items]
                    s_emb = self.embedding_model.encode(
                        sent_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
                    )
                    s_mat = np.asarray(s_emb, dtype=np.float32)
                    # q 已归一化；encode normalize 后点积即余弦
                    sims = (s_mat @ q.reshape(-1, 1)).reshape(-1)
                    fused = []
                    for sim_s, item in zip(sims.tolist(), sent_items):
                        fused.append((float(sim_s), item))
                    ranked = sorted(fused, key=lambda x: float(x[0]), reverse=True)
                    out = []
                    seen_txt = set()
                    for sim_s, (iidx, fname_str, sent, meta, _ps, si, parent_text) in ranked:
                        key = "".join(sent.split())[:240]
                        if key in seen_txt:
                            continue
                        seen_txt.add(key)
                        meta2 = dict(meta or {})
                        meta2["within_dense"] = round(float(sim_s), 4)
                        display_text = sent
                        if sentence_neighbor_window > 0 and split_evidence_sentences is not None:
                            siblings = split_evidence_sentences(parent_text) or []
                            lo = max(0, si - int(sentence_neighbor_window))
                            hi = min(len(siblings), si + int(sentence_neighbor_window) + 1)
                            window = siblings[lo:hi]
                            if window:
                                display_text = " ".join(window)
                                meta2["sentence_window"] = [lo, hi]
                                meta2["matched_sentence_index"] = si
                        out.append(
                            RetrievedChunk(
                                chunk_id=f"{iidx}#w{si}",
                                source=fname_str,
                                chunk_text=display_text,
                                metadata=meta2,
                                similarity_score=float(sim_s),
                                retrieval_source="within_std_sent",
                            )
                        )
                        if len(out) >= max(1, int(limit)):
                            break

            if out:
                logger.info(
                    "同标准内二次检索: stds=%d cand=%d keep=%d top=%.3f sent=%s",
                    len(std_keys),
                    len(candidates),
                    len(out),
                    float(out[0].similarity_score or 0),
                    sentence_level,
                )
            return out
        except Exception as e:
            logger.warning(f"同标准内二次检索失败，回退文件取块: {e}")
            return self.find_chunks_by_standard(std_keys, limit=limit)

    def find_chunks_by_sources(self, sources: List[str], per_source: int = 3) -> List[RetrievedChunk]:
        """按已有来源文件名取块（供 correlation：KG source → 向量块）。"""
        if not sources or not self.chunk_mapping:
            return []
        keys = []
        seen = set()
        for s in sources:
            s = str(s or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            keys.append(s)
        if not keys:
            return []
        out: List[RetrievedChunk] = []
        for key in keys:
            found = self.find_chunks_by_standard([key], limit=per_source)
            for c in found:
                c.retrieval_source = "kg_source_inject"
                out.append(c)
        return out

    @staticmethod
    def _source_matches_any(source: str, std_keys: List[str]) -> bool:
        if not source or not std_keys:
            return False
        fname_canon = canonical_standard_id(source)
        for k in std_keys:
            k_str = str(k or "")
            k_canon = k_str if ":" in k_str else (canonical_standard_id(k_str) or k_str)
            if fname_canon and k_canon and fname_canon == k_canon:
                return True
            if fname_canon and k_canon and "::" in fname_canon and "::" in k_canon:
                if fname_canon.split("::", 1)[0] == k_canon.split("::", 1)[0]:
                    return True
            if docs_match(source, k) or docs_match(source, f"{k}.md"):
                return True
            src_core = normalize_doc_key(source).replace(".md", "")
            k_core = normalize_doc_key(k).replace(".md", "")
            if src_core and k_core and (src_core == k_core or k_core in src_core or src_core in k_core):
                return True
        return False

    def _load_embeddings(self):
        emb_path = self.vector_db_path / "embeddings.npy"
        if not emb_path.exists():
            logger.warning(f"embeddings.npy 不存在: {emb_path}")
            return None
        try:
            emb = np.load(str(emb_path))
            if emb.dtype != np.float32:
                emb = emb.astype("float32")
            logger.info(
                f"加载 embeddings.npy 成功，数量: {emb.shape[0]}, "
                f"维度: {emb.shape[1] if emb.ndim > 1 else 'N/A'}"
            )
            return emb
        except Exception as e:
            logger.error(f"加载 embeddings.npy 失败: {e}")
            return None

    def _compute_embeddings_norms(self):
        if getattr(self, "embeddings", None) is None:
            self.embeddings_norms = None
            return
        try:
            self.embeddings_norms = np.linalg.norm(self.embeddings, axis=1)
        except Exception:
            self.embeddings_norms = None
