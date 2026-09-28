from data_types import SystemConfig
from compare_rag.Hybrid_rag.sparse_retriever import SparseRetriever
from compare_rag.Hybrid_rag.dense_retriever import DenseRetriever
from compare_rag.utils import (build_ctx_from_chunks, baseline_router_stub,
                               explicit_clause_citation_refs)
from llm_generator import LLMGenerator
import logging
import time
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class HybridRAGsystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.sparse_retriever = SparseRetriever(config)
        self.dense_retriever = DenseRetriever(config)
        self.llm_generator = LLMGenerator(config)
        self._hotpot_vr = None

    def _normalize_hits(self, hits):
        """将检索器返回结果归一化为 dict 包含 source, chunk_text, score 等字段"""
        out = []
        for h in hits or []:
            if isinstance(h, dict):
                chunk_text = h.get('chunk_text') or h.get('text') or h.get('text_preview') or h.get('content') or ''
                source = h.get('source') or h.get('filename') or ''
                score = float(h.get('similarity_score', h.get('score', 0.0))) if h else 0.0
                out.append({'chunk_id': h.get('chunk_id', None), 'source': source, 'chunk_text': chunk_text,
                            'metadata': dict(h.get('metadata') or {}), 'score': score})
            else:
                # 支持对象类型，尝试属性读取
                try:
                    chunk_text = getattr(h, 'chunk_text', '') or getattr(h, 'text', '')
                    source = getattr(h, 'source', '') or getattr(h, 'filename', '')
                    score = float(getattr(h, 'similarity_score', 0.0))
                    meta = dict(getattr(h, "metadata", None) or {})
                    out.append({
                        'chunk_id': getattr(h, 'chunk_id', None),
                        'source': source,
                        'chunk_text': chunk_text,
                        'metadata': meta,
                        'score': score,
                    })
                except Exception:
                    continue
        return out

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> dict:
        start_time = time.time()
        try:
            top_k = getattr(self.config, 'top_k_vector', 20)

            # HotPot / 非 ht：使用 HotPot 向量库，避免检索工业标准语料
            if dataset_class == 'hotpot':
                from vector_retriever import VectorRetriever
                from hotpot_local_retrieval import get_local_chunks, bm25_rank_local, dense_rank_local, merge_local_hits
                from hotpot_citation_utils import build_hotpot_ctx_from_passages, resolve_hotpot_citations_from_passages
                if not hasattr(self, '_hotpot_vr') or self._hotpot_vr is None:
                    self._hotpot_vr = VectorRetriever(self.config, dataset_class='hotpot')
                sparse_start_time = time.time()
                local = get_local_chunks(qa_item)
                if local:
                    dense_hits = dense_rank_local(question, local, self._hotpot_vr.embedding_model, top_k=top_k)
                    bm25_hits = bm25_rank_local(question, local, top_k=top_k)
                    merged = merge_local_hits(dense_hits, bm25_hits, top_k)
                    sparse_passages = self._normalize_hits([h.to_dict() if hasattr(h, 'to_dict') else h for h in merged])
                else:
                    hits = self._hotpot_vr.retrieve(question=question, entities=[], kg_entities=[])
                    sparse_passages = self._normalize_hits([h.to_dict() if hasattr(h, 'to_dict') else h for h in hits])
                sparse_retrieval_time = time.time() - sparse_start_time
                dense_start_time = time.time()
                dense_passages = list(sparse_passages)
                dense_retrieval_time = time.time() - dense_start_time
                combined = sparse_passages[:top_k]
                context = build_hotpot_ctx_from_passages(combined, top_k=min(10, len(combined)))
            else:
                # 稀疏检索
                sparse_start_time = time.time()
                sparse_raw = self.sparse_retriever.retrieve(question, top_k=top_k, dataset_class=dataset_class)
                sparse_retrieval_time = time.time() - sparse_start_time
                sparse_passages = self._normalize_hits(sparse_raw)

                # 密集检索
                dense_start_time = time.time()
                dense_raw = self.dense_retriever.retrieve(question, top_k=top_k, dataset_class=dataset_class)
                dense_retrieval_time = time.time() - dense_start_time
                dense_passages = self._normalize_hits(dense_raw)

            if dataset_class != 'hotpot':
                top_k = getattr(self.config, 'top_k_vector', 20)
                fused = {}
                rrf_k = 60.0
                for channel, passages in (("sparse", sparse_passages), ("dense", dense_passages)):
                    for rank, p in enumerate(passages, start=1):
                        meta = dict(p.get("metadata") or {})
                        key = (
                            meta.get("evidence_id") or p.get("evidence_id") or
                            (p.get('source', ''), p.get('chunk_text', '')[:200])
                        )
                        entry = fused.setdefault(key, {"passage": dict(p), "score": 0.0, "channels": []})
                        entry["score"] += 1.0 / (rrf_k + rank)
                        entry["channels"].append(channel)
                ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
                combined = []
                for item in ranked[:top_k]:
                    passage = item["passage"]
                    passage["fusion_score"] = item["score"]
                    passage["retrieval_channels"] = item["channels"]
                    combined.append(passage)

                context = build_ctx_from_chunks(combined, limit=len(combined))

            # 生成答案
            generation_start_time = time.time()
            llm_response = self.llm_generator.generate_answer(question, context, dataset_class)
            generation_time = time.time() - generation_start_time

            if dataset_class == 'hotpot':
                from hotpot_citation_utils import resolve_hotpot_citations_from_passages
                citations = resolve_hotpot_citations_from_passages(
                    answer=llm_response.answer,
                    raw_response=llm_response.raw_response,
                    evidence_citations=llm_response.evidence_citations,
                    passages=combined,
                )
            else:
                citations = llm_response.evidence_citations

            # 打包结果
            result = {
                'question': question,
                'router_analysis': baseline_router_stub('Hybrid-RAG'),
                'retrieval': {
                    'sparse_results': sparse_passages,
                    'sparse_retrieval_time': sparse_retrieval_time,
                    'dense_results': dense_passages,
                    'dense_retrieval_time': dense_retrieval_time,
                    'combined_results': combined,
                    'vector_results': combined,
                    'retrieval_time': sparse_retrieval_time + dense_retrieval_time
                },
                'generation': {
                    'answer': llm_response.answer,
                    'citation_llm_use': citations,
                    'citation_refs': explicit_clause_citation_refs(llm_response.raw_response, combined),
                    'generation_time': generation_time,
                    'raw_response': llm_response.raw_response
                },
                'performance': {
                    'total_time': sparse_retrieval_time + dense_retrieval_time + generation_time,
                    'retrieval_time': sparse_retrieval_time + dense_retrieval_time,
                    'generation_time': generation_time
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_class': dataset_class
            }

            # 追加到 method_logs/{method}.jsonl
            from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
            log_file = resolve_run_jsonl(
                self,
                getattr(self, "_run_method_name", "Hybrid-RAG"),
                getattr(self, "_output_subdir", ""),
            )
            append_run_jsonl(log_file, result)
            logger.info("结果已保存: %s", log_file)

            return result
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return {
                'error': str(e),
                'question': question,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_class': dataset_class
            }

    def close(self):
        """Release resources after a benchmark run."""
        pass
