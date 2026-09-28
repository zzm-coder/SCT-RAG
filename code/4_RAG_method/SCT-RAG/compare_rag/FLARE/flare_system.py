"""FLARE (Active Retrieval Augmented Generation) skeleton.
This module implements a simple active retrieval loop: generate an initial answer, identify missing facts,
query for those facts, and then regenerate.
"""
from typing import Dict, Any, List
from data_types import SystemConfig
from vector_retriever import VectorRetriever
from llm_generator import LLMGenerator
import time
from difflib import SequenceMatcher
from pathlib import Path
import json
import logging
import re
import numpy as np
from compare_rag.utils import build_ctx_from_chunks, baseline_router_stub
from api_embedding import APIEmbeddingModel

logger = logging.getLogger(__name__)

class FLARESystem:
    """Active Retrieval (FLARE) 简化实现：
    - 初次生成短片段答案，估计置信度（基于与已检索上下文的语义/表面相似度）
    - 若置信度低于阈值则触发一次向量检索并将结果拼接后继续生成
    """

    def __init__(self, config: SystemConfig, use_vector: bool = True):
        self.config = config
        self.use_vector = use_vector
        self.vector_retriever = None
        self.llm = LLMGenerator(config)
        # Unified evaluator interface used by all benchmark methods.
        self.llm_generator = self.llm
        self.semantic_model = None
        try:
            self.semantic_model = APIEmbeddingModel(config.semantic_model_path)
            logger.info(f'Configured embedding API model {config.semantic_model_path}')
        except Exception as e:
            self.semantic_model = None
            logger.warning(f'无法配置 embedding API，回退到表面相似度: {e}')

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> Dict[str, Any]:
        if self.use_vector and self.vector_retriever is None:
            self.vector_retriever = VectorRetriever(self.config, dataset_class=dataset_class)
        from hotpot_local_retrieval import get_local_chunks
        self._local_chunks = get_local_chunks(qa_item) if dataset_class == "hotpot" else None
        start = time.time()

        # 计时
        retrieval_time = 0.0
        generation_time_initial = 0.0
        generation_time_final = 0.0

        # 初始尝试：先从向量库拿少量上下文（cheap）并计时
        cheap_ctx = []
        try:
            rstart = time.time()
            cheap_ctx_raw = self.vector_retriever.retrieve(
                question=question, entities=[], kg_entities=[], local_chunks=self._local_chunks
            )
            retrieval_time += time.time() - rstart
            # 取前3作为 cheap
            cheap_ctx_raw = cheap_ctx_raw[:3]
        except Exception as e:
            logger.warning(f'FLARE cheap retrieval failed: {e}')
            cheap_ctx_raw = []

        # 统一转换为 dict
        def _to_dict_safe(c):
            if isinstance(c, dict):
                return c
            if hasattr(c, 'to_dict'):
                return c.to_dict()
            return {'chunk_text': getattr(c, 'chunk_text', ''), 'source': getattr(c, 'source', '')}
        cheap_ctx = [_to_dict_safe(c) for c in cheap_ctx_raw]
        if dataset_class == "hotpot":
            from hotpot_citation_utils import build_hotpot_ctx_from_passages
            cheap_text = build_hotpot_ctx_from_passages(cheap_ctx, top_k=3)
        else:
            cheap_text = build_ctx_from_chunks(cheap_ctx, limit=3)

        # 初次生成并计时
        try:
            gstart = time.time()
            gen = self.llm.generate_answer(question, cheap_text, dataset_class)
            generation_time_initial = time.time() - gstart
            answer = gen.answer # getattr(gen, 'answer', '')
            citiations = gen.evidence_citations # getattr(gen, 'evidence_citations', [])
        except Exception:
            try:
                gstart = time.time()
                raw = self.llm._call_llm(f"问题：{question}\n\n简短回答：")
                generation_time_initial = time.time() - gstart
                answer, citiations = self.llm._parse_response(raw)
            except Exception:
                answer = ''
                citiations = []

        # 估计置信度：优先使用语义相似度（若模型可用），否则回退到 SequenceMatcher
        conf = 0.0
        try:
            if self.semantic_model is not None and cheap_text.strip() and answer.strip():
                a_emb = self.semantic_model.encode(answer, convert_to_numpy=True)
                t_emb = self.semantic_model.encode(cheap_text, convert_to_numpy=True)
                a_norm = np.linalg.norm(a_emb) or 1e-8
                t_norm = np.linalg.norm(t_emb) or 1e-8
                sim = float((a_emb @ t_emb) / (a_norm * t_norm))
                conf = (sim + 1.0) / 2.0
            else:
                conf = SequenceMatcher(None, answer, cheap_text).ratio()
        except Exception:
            conf = SequenceMatcher(None, answer, cheap_text).ratio() if cheap_text and answer else 0.0

        result = {
            "question": question,
            "router_analysis": baseline_router_stub('FLARE'),
            "retrieval": {"vector_results": cheap_ctx},
            "generation": {"answer": answer, "citation_llm_use": citiations,
                           "raw_response": gen.raw_response or "" if 'gen' in locals() else ""},
            "performance": {},
        }
        result['analysis'] = {'flare_initial_confidence': float(conf)}

        # 若置信度低于阈值则主动检索更多上下文并重生成（计时）
        # 论文 FLARE：用即将生成/已生成的不确定句作为检索查询，而不是原问题再搜一遍
        # 答案-上下文余弦会偏高（答案由 cheap 上下文生成），不能当论文 token 置信度。
        # 无 logprob 时阈值取 0.99，使前向句检索成为默认主动检索步。
        threshold = getattr(self.config, 'flare_conf_threshold', 0.99)
        if conf < threshold:
            fwd_query = self._flare_forward_query(question, answer)
            try:
                rstart2 = time.time()
                vecs = self.vector_retriever.retrieve(
                    question=fwd_query or question,
                    entities=[],
                    kg_entities=[],
                    local_chunks=self._local_chunks
                )
                # 保留原问题检索作为补充通道，避免前向句过短漏召
                if fwd_query and fwd_query != question:
                    extra = self.vector_retriever.retrieve(
                        question=question, entities=[], kg_entities=[],
                        local_chunks=self._local_chunks,
                    )
                    vecs = list(vecs or []) + list(extra or [])
                retrieval_time += time.time() - rstart2
            except Exception as e:
                logger.warning(f'FLARE full retrieval failed: {e}')
                vecs = []

            vecs_dict = [v.to_dict() if hasattr(v, 'to_dict') else v for v in vecs]
            # 前向句检索与原问题检索去重
            dedup = []
            seen = set()
            for p in vecs_dict:
                meta = dict(p.get("metadata") or {}) if isinstance(p, dict) else {}
                key = meta.get("evidence_id") or (
                    p.get("source", ""), (p.get("chunk_text") or p.get("text_preview") or "")[:180]
                )
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(p)
            vecs_dict = dedup
            result['retrieval']['vector_results'] = vecs_dict
            result['analysis']['flare_forward_query'] = fwd_query
            if dataset_class == "hotpot":
                from hotpot_citation_utils import build_hotpot_ctx_from_passages, resolve_hotpot_citations_from_passages
                ctx = build_hotpot_ctx_from_passages(vecs_dict, top_k=min(10, len(vecs_dict)))
            else:
                ctx = build_ctx_from_chunks(vecs_dict, limit=self.config.top_k_vector)
            gen2_raw = ""
            try:
                gstart2 = time.time()
                gen2 = self.llm.generate_answer(question, ctx, dataset_class)
                generation_time_final = time.time() - gstart2
                answer2 = gen2.answer
                citations2 = gen2.evidence_citations
                gen2_raw = gen2.raw_response or ""
            except Exception:
                try:
                    gstart2 = time.time()
                    raw2 = self.llm._call_llm(f"基于下列检索内容回答问题：\n{ctx}\n\n问题：{question}\n\n详细回答：")
                    generation_time_final = time.time() - gstart2
                    answer2, citations2 = self.llm._parse_response(raw2)
                    gen2_raw = raw2
                except Exception:
                    answer2 = ''
                    citations2 = []
                    gen2_raw = ""

            result['generation']['answer'] = answer2
            result['generation']['raw_response'] = gen2_raw
            if dataset_class == "hotpot":
                result['generation']['citation_llm_use'] = resolve_hotpot_citations_from_passages(
                    answer=answer2,
                    raw_response=gen2_raw,
                    evidence_citations=citations2,
                    passages=vecs_dict,
                )
            else:
                result['generation']['citation_llm_use'] = citations2
            result['analysis']['flare_triggered'] = True
        else:
            result['analysis']['flare_triggered'] = False

        result['performance']['total_time'] = time.time() - start
        result['performance']['retrieval_time'] = float(retrieval_time)
        result['performance']['generation_time'] = float(generation_time_initial + generation_time_final)
        result['performance']['generation_time_initial'] = float(generation_time_initial)
        result['performance']['generation_time_final'] = float(generation_time_final)

        from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
        log_file = resolve_run_jsonl(
            self,
            getattr(self, "_run_method_name", "FLARE"),
            getattr(self, "_output_subdir", ""),
        )
        try:
            append_run_jsonl(log_file, result)
        except Exception:
            logger.debug("无法写入 FLARE 日志: %s", log_file)
        
        logger.info(f"FLARE 结果已保存: {log_file}")

        return result

    def _flare_forward_query(self, question: str, draft: str) -> str:
        """用草稿中需查证的下一句构造检索查询（FLARE forward-looking query）。"""
        draft = (draft or "").strip()
        if not draft:
            return question
        prompt = (
            "根据问题与当前草稿，写出下一条需要查证的检索查询。"
            "保留标准号与未证实的对象/公式/限值。只输出查询。\n"
            f"问题：{question}\n草稿：{draft[:400]}\n/no_think"
        )
        try:
            import re as _re
            raw = self.llm._call_llm(prompt) or ""
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.I | _re.S).strip()
            q = raw.splitlines()[0].strip().strip('"')
            if 8 <= len(q) <= 200:
                return q
        except Exception:
            pass
        # 回退：用草稿末句当查询
        last = [s.strip() for s in re.split(r"[。！？\n]", draft) if s.strip()]
        return last[-1][:120] if last else question

    def close(self):
        self.vector_retriever = None
