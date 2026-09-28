"""CRAG (Corrective Retrieval Augmented Generation) skeleton.
This module implements a corrective loop: generate, verify against evidence, and apply corrective retrieval if contradictions detected.
"""
from typing import Dict, Any
from data_types import SystemConfig
from vector_retriever import VectorRetriever
from llm_generator import LLMGenerator
import time
import re
from pathlib import Path
import json
import logging
import numpy as np

from compare_rag.utils import baseline_router_stub
from api_embedding import APIEmbeddingModel

logger = logging.getLogger(__name__)


class CRAGSystem:
    """Corrective RAG 实现：
    - 标准向量检索 Top-K
    - 轻量检索评估器：分解、重排、过滤段落（基于与问题的相关性）
    - 使用修正后的上下文进行生成
    """

    def __init__(self, config: SystemConfig, use_vector: bool = True):
        self.config = config
        self.use_vector = use_vector
        self.vector_retriever = None

        self.llm = LLMGenerator(config)
        self.llm_generator = self.llm
        self.semantic_model = None
        try:
            self.semantic_model = APIEmbeddingModel(config.semantic_model_path)
            logger.info(f'Configured embedding API model {config.semantic_model_path}')
        except Exception as e:
            self.semantic_model = None
            logger.warning(f'无法配置 embedding API，回退到启发式评分: {e}')

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> Dict[str, Any]:

        if self.use_vector and self.vector_retriever is None:
            self.vector_retriever = VectorRetriever(self.config, dataset_class=dataset_class)
        from hotpot_local_retrieval import get_local_chunks
        self._local_chunks = get_local_chunks(qa_item) if dataset_class == "hotpot" else None
        start = time.time()

        # 1) 标准向量检索 Top-K（作为初始候选）并计时
        retrieval_time = 0.0
        try:
            retr_start = time.time()
            vecs = self.vector_retriever.retrieve(
                question=question, entities=[], kg_entities=[], local_chunks=self._local_chunks
            )
            retrieval_time += time.time() - retr_start
        except Exception as e:
            logger.warning(f'初始向量检索失败: {e}')
            vecs = []
        vec_list = [v.to_dict() if hasattr(v, 'to_dict') else v for v in vecs]

        # 2) 论文 CRAG：检索评估器判 Correct / Incorrect / Ambiguous
        scored = []
        try:
            if self.semantic_model is not None and len(vec_list) > 0:
                texts = [v.get('chunk_text') or v.get('text_preview') or '' for v in vec_list]
                q_emb = self.semantic_model.encode(question, convert_to_numpy=True)
                p_embs = self.semantic_model.encode(texts, convert_to_numpy=True)
                q_norm = np.linalg.norm(q_emb) or 1e-8
                p_norms = np.linalg.norm(p_embs, axis=1)
                p_norms[p_norms == 0] = 1e-8
                sims = (p_embs @ q_emb) / (p_norms * q_norm)
                for v, s in zip(vec_list, sims.tolist()):
                    scored.append({'passage': v, 'score': float(s)})
            else:
                for v in vec_list:
                    text = v.get('chunk_text') or v.get('text_preview') or ''
                    score = self._score_passage_relevance(question, text)
                    scored.append({'passage': v, 'score': float(score)})
        except Exception as e:
            logger.warning(f'语义评分出错，回退到启发式评分: {e}')
            scored = []
            for v in vec_list:
                text = v.get('chunk_text') or v.get('text_preview') or ''
                score = self._score_passage_relevance(question, text)
                scored.append({'passage': v, 'score': float(score)})

        scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
        top_k = scored_sorted[: max(1, min(len(scored_sorted), self.config.top_k_vector))]
        avg_top_score = float(sum([s['score'] for s in top_k]) / max(1, len(top_k)))
        cosine_fallback = float(avg_top_score)
        action = self._evaluate_retrieval_quality(
            question, [s['passage'] for s in top_k[:8]]
        )
        if action not in ("correct", "incorrect", "ambiguous"):
            # 评估器失败时才用余弦阈值（论文尺度为原始 cosine）
            HIGH_T = getattr(self.config, 'crag_high_threshold', 0.65)
            LOW_T = getattr(self.config, 'crag_low_threshold', 0.35)
            if cosine_fallback >= HIGH_T:
                action = 'correct'
            elif cosine_fallback <= LOW_T:
                action = 'incorrect'
            else:
                action = 'ambiguous'
        decision_score = cosine_fallback

        # 4) 根据 action 执行相应操作（Incorrect 用语料内二次检索替代网页）
        refined_passages = []
        web_results = []
        if action == 'correct':
            refined_passages = self._decompose_and_refine_passages(
                [s['passage'] for s in top_k], question
            )
        elif action == 'incorrect':
            web_start = time.time()
            web_results = self._complementary_retrieve(question, [s['passage'] for s in top_k])
            retrieval_time += time.time() - web_start
            refined_passages = web_results
        else:  # ambiguous
            refined = self._decompose_and_refine_passages(
                [s['passage'] for s in top_k], question
            )
            web_start = time.time()
            web_results = self._complementary_retrieve(question, refined)
            retrieval_time += time.time() - web_start
            merged = []
            seen = set()
            for p in (refined + web_results):
                meta = dict(p.get("metadata") or {}) if isinstance(p, dict) else {}
                eid = str(meta.get("evidence_id") or "")
                txt = (p.get('chunk_text') or p.get('text_preview','') or '')[:200]
                key = eid or (p.get('source',''), txt)
                if key not in seen:
                    seen.add(key)
                    merged.append(p)
            refined_passages = merged[: self.config.top_k_vector]

        # 最终上下文
        if dataset_class == "hotpot":
            from hotpot_citation_utils import build_hotpot_ctx_from_passages, resolve_hotpot_citations_from_passages
            passage_dicts = [
                {
                    "source": p.get("source", "") if isinstance(p, dict) else "",
                    "chunk_text": (p.get("chunk_text") or p.get("text_preview") or "") if isinstance(p, dict) else "",
                }
                for p in refined_passages
            ]
            final_ctx = build_hotpot_ctx_from_passages(passage_dicts, top_k=min(10, len(passage_dicts)))
        else:
            from compare_rag.utils import build_ctx_from_chunks
            final_ctx = build_ctx_from_chunks(refined_passages, limit=self.config.top_k_vector)
        generation_time = 0.0
        try:
            gen_start = time.time()
            gen = self.llm.generate_answer(question, final_ctx, dataset_class)
            generation_time = time.time() - gen_start
            answer = gen.answer # getattr(gen, 'answer', '')
            raw_response = gen.raw_response or ""
            citations = gen.evidence_citations # getattr(gen, 'evidence_citations', [])
            if dataset_class == "hotpot":
                from hotpot_citation_utils import resolve_hotpot_citations_from_passages
                citations = resolve_hotpot_citations_from_passages(
                    answer=answer,
                    raw_response=gen.raw_response or "",
                    evidence_citations=citations,
                    passages=[
                        {
                            "source": p.get("source", ""),
                            "chunk_text": p.get("chunk_text") or p.get("text_preview") or "",
                        }
                        for p in refined_passages
                    ],
                )
        except Exception:
            # 回退到直接 prompt
            try:
                prompt = f"基于上下文片段，回答问题并给引用的证据。\n上下文：{final_ctx}\n\n问题：{question}\n\n【答案】\n\n【证据】"
                gen_start = time.time()
                raw = self.llm._call_llm(prompt)
                generation_time = time.time() - gen_start
                answer, citations = self.llm._parse_response(raw)
                raw_response = raw
            except Exception:
                answer = ''
                citations = []
                raw_response = ''

        # 构建返回结果并记录
        result = {
            "question": question,
            "router_analysis": baseline_router_stub('CRAG'),
            'retrieval': {
                'vector_results': refined_passages,
                'initial_vector_results': vec_list,
            },
            'generation': {'answer': answer, 'citation_llm_use': citations,
                           'raw_response': raw_response},
            'analysis': {
                'crag_action': action,
                'avg_top_score': avg_top_score,
                'crag_decision_score': decision_score,
                'crag_top_scores': [s['score'] for s in top_k],
                'refined_count': len(refined_passages),
                'web_fallback_count': len(web_results),
                'complementary_count': len(web_results),
            },
            'performance': {'total_time': time.time() - start, 'generation_time': generation_time, 'retrieval_time': retrieval_time}
        }

        from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
        try:
            log_file = resolve_run_jsonl(
                self,
                getattr(self, "_run_method_name", "CRAG"),
                getattr(self, "_output_subdir", ""),
            )
            append_run_jsonl(log_file, result)
            logger.info("CRAG 结果已保存: %s", log_file)
        except Exception:
            logger.exception(f"无法写入 CRAG 日志到 {output_path}")

        logger.info(f"CRAG 完成: action={action}, avg_top_score={avg_top_score}, generation_time={result['performance']['generation_time']}, retrieval_time={result['performance'].get('retrieval_time',0.0)}")
        return result

    def close(self):
        self.vector_retriever = None

    def _evaluate_retrieval_quality(self, question: str, passages: list) -> str:
        """论文 CRAG 检索评估器：Correct / Incorrect / Ambiguous（LLM 三分类）。"""
        if not passages:
            return "incorrect"
        snippets = []
        for i, p in enumerate(passages[:6], start=1):
            if not isinstance(p, dict):
                continue
            meta = dict(p.get("metadata") or {})
            ident = meta.get("evidence_id") or meta.get("clause_id") or ""
            text = (p.get("chunk_text") or p.get("text_preview") or "")[:280]
            snippets.append(f"[{i}] {ident} {text}")
        prompt = f"""你是检索评估器。判断下列检索证据能否回答用户问题。
只输出一个标签：correct / incorrect / ambiguous
- correct：证据已包含直接回答所需的关键事实（定义/数值/公式/禁止项）
- incorrect：证据与问点无关或明显答非所问
- ambiguous：部分相关但缺关键句或公式不完整

问题：{question}

证据：
{chr(10).join(snippets) or '（空）'}
/no_think
"""
        try:
            raw = self.llm._call_llm(prompt) or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S)
            low = raw.strip().lower()
            if "incorrect" in low or "error" in low:
                return "incorrect"
            if "ambiguous" in low or "fuzzy" in low:
                return "ambiguous"
            if "correct" in low:
                return "correct"
        except Exception as e:
            logger.warning("CRAG 评估器失败: %s", e)
        return ""

    def _complementary_retrieve(self, question: str, current_passages: list) -> list:
        """封闭语料上的纠正检索：改写查询后再向量检索（对应论文 web search）。"""
        rewrite = question
        try:
            prompt = (
                "根据问题写出一条更具体的中文检索查询，必须保留题干标准号和问点对象"
                "（定义/公式/限值）。只输出查询本身。\n问题：" + question + "\n/no_think"
            )
            raw = self.llm._call_llm(prompt) or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S).strip()
            raw = raw.splitlines()[0].strip().strip('"')
            if 8 <= len(raw) <= 200:
                rewrite = raw
        except Exception:
            rewrite = question
        try:
            vecs = self.vector_retriever.retrieve(
                question=rewrite,
                entities=[],
                kg_entities=[],
                local_chunks=getattr(self, "_local_chunks", None),
            )
            out = [v.to_dict() if hasattr(v, "to_dict") else v for v in vecs]
        except Exception:
            out = self._web_search(question)
        return out[: self.config.top_k_vector]

    def _decompose_and_refine_passages(self, passages: list, question: str) -> list:
        """知识精炼：以完整条款为 strip，过滤无关条；公式块不按句号切开。"""
        from standard_boost import split_evidence_sentences, chunk_has_real_formula

        scored_chunks = []
        for p in passages or []:
            if isinstance(p, dict):
                text = p.get("chunk_text") or p.get("text_preview") or ""
                src = p.get("source", "")
                metadata = dict(p.get("metadata") or {})
            elif isinstance(p, str):
                text, src, metadata = p, "", {}
            else:
                continue
            text = str(text or "").strip()
            if not text:
                continue
            # 工业条款：默认整段作为一条 knowledge strip
            candidates = [text]
            if len(re.sub(r"\s+", "", text)) > 800 and not chunk_has_real_formula(text):
                sents = split_evidence_sentences(text)
                if sents:
                    candidates = sents
            best_txt, best_sc = text, self._score_passage_relevance(question, text)
            for cand in candidates:
                sc = self._score_passage_relevance(question, cand)
                if sc > best_sc:
                    best_sc, best_txt = sc, cand
            if best_sc <= 0:
                continue
            scored_chunks.append(
                {
                    "passage": {
                        "chunk_text": text if chunk_has_real_formula(text) else best_txt,
                        "source": src,
                        "metadata": metadata,
                    },
                    "score": float(best_sc),
                }
            )

        if not scored_chunks:
            fallback = []
            for p in (passages or [])[: self.config.top_k_vector]:
                if isinstance(p, dict) and (p.get("chunk_text") or p.get("text_preview")):
                    fallback.append(
                        {
                            "source": p.get("source", ""),
                            "chunk_text": p.get("chunk_text") or p.get("text_preview") or "",
                            "metadata": dict(p.get("metadata") or {}),
                        }
                    )
            return fallback

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        selected = [s["passage"] for s in scored_chunks[: self.config.top_k_vector]]
        return selected

    def _web_search(self, query: str) -> list:
        """尝试调用可配置的 web-search 接口；若不可用则降级到扩大检索范围的向量检索。
        返回与段落相同结构的 dict 列表（包含 chunk_text/source）。"""
        # 如果 VectorRetriever 提供 web_search 接口，优先使用
        if hasattr(self.vector_retriever, 'web_search'):
            try:
                web = self.vector_retriever.web_search(query)
                return [w.to_dict() if hasattr(w, 'to_dict') else w for w in web]
            except Exception:
                pass

        # 降级：使用更宽松的向量检索作为近似 web-search
        try:
            broaden_q = query + ' background OR overview'
            vecs = self.vector_retriever.retrieve(
                question=broaden_q, entities=[], kg_entities=[], local_chunks=getattr(self, "_local_chunks", None)
            )
            return [v.to_dict() if hasattr(v, 'to_dict') else v for v in vecs]
        except Exception:
            return []

    def _score_passage_relevance(self, question: str, passage: str) -> float:
        # 简单启发式相关性评分：基于关键片段重叠与长度比
        if not passage:
            return 0.0
        # 交叉句子关键词覆盖
        q_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', question))
        p_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', passage))
        if not q_tokens or not p_tokens:
            return 0.0
        overlap = q_tokens.intersection(p_tokens)
        score = len(overlap) / max(1, min(len(q_tokens), len(p_tokens)))
        # 长度惩罚/奖励
        if len(passage) < 40:
            score *= 0.6
        return float(min(1.0, score))
