"""RAG评估器"""
import json
import os
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import re

from config import SystemConfig
from sct_rag_system import SCTRAGSystem
from data_types import EvaluationResult
from collections import defaultdict
from difflib import SequenceMatcher
from compare_rag.utils import (collect_retrieved_chunks, docs_match,
                               normalize_doc_key, explicit_clause_citation_refs)
from clause_metrics import citation_metrics, penalized_accuracy, retrieval_at_k, same_std_ranking_metrics
from accuracy_protocol import llm_judge_pass

logger = logging.getLogger(__name__)

# 评测时剔除标准号 token，避免仅因题干/段落共现标准号而虚高条款命中
_STD_TOKEN_RE = re.compile(
    r"(?:GB/T|GBT|GB|HB(?:/Z)?|HB_Z|GJB|QJ|SJ)[\s\+]*[\d\-—－/\.A-Za-z（）()]+",
    re.I,
)
_GENERIC_EVAL_TOKENS = {
    "根据", "标准", "要求", "规定", "进行", "采用", "设计", "检验", "试验",
    "应当", "应", "可", "或", "及", "与", "的", "和", "等", "下列", "以下",
}

PAPER_TYPE_MAP = {
    "single": "single",
    "cross": "cross",
    "correlation": "correlation",
}


def normalize_paper_type(value):
    if value is None:
        return None
    raw = str(value).strip()
    return PAPER_TYPE_MAP.get(raw, PAPER_TYPE_MAP.get(raw.lower(), raw))

class RAGEvaluator:
    """RAG评估器"""
    
    def __init__(self, qa_dataset_path: str):
        self.qa_dataset = self._load_qa_dataset(qa_dataset_path)
        self.evaluation_results = []
    
    def _load_qa_dataset(self, path: str) -> Dict[str, Any]:
        """加载QA数据集"""
        dataset_path = Path(path)
        
        if dataset_path.is_dir():
            dataset = {}
            for file in dataset_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            dataset[item["id"]] = item
                    else:
                        dataset[file.stem] = data
            return dataset
        elif dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {item["id"]: item for item in data}
                else:
                    return data
        else:
            raise FileNotFoundError(f"QA数据集不存在: {path}")
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """HotPot / 通用答案规范化"""
        t = (text or "").strip().lower()
        # 去掉引用标记与拒答套话，避免噪声拉低 token F1
        t = re.sub(r"\[\d+\]", " ", t)
        t = re.sub(r"unable to answer based on the provided information", " ", t)
        t = re.sub(r"[^\w\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _hotpot_answer_score(self, predicted: str, ground_truth: str) -> float:
        """HotPot 答案评分：yes/no 精确匹配 + 实体子串匹配 + token F1"""
        raw = (predicted or "").strip().lower()
        # 拒答不得分（规范化后空串会命中 ``"" in gt``，必须先拦截）
        if not raw or "unable to answer" in raw:
            return 0.0
        pred = self._normalize_answer(predicted)
        gt = self._normalize_answer(ground_truth)
        if not gt or not pred:
            return 0.0
        if gt in ("yes", "no"):
            if re.search(rf"\b{gt}\b", pred):
                return 1.0
            return 0.0
        if gt in pred or pred in gt:
            return 1.0
        gt_tokens = set(gt.split())
        pred_tokens = set(pred.split())
        if not gt_tokens:
            return 0.0
        overlap = len(gt_tokens & pred_tokens)
        precision = overlap / max(len(pred_tokens), 1)
        recall = overlap / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _prepare_qa_item(self, qa_item: Dict, dataset_class: str) -> Dict:
        """补齐 chunk.source 与 supporting_facts（HotPot 常为空）。"""
        item = dict(qa_item)
        chunks = item.get("chunks") or []
        qid = item.get("id", "")
        for c in chunks:
            if not c.get("source"):
                c["source"] = f"{qid}_{c.get('id', 0)}"
            if dataset_class == "hotpot":
                from hotpot_local_retrieval import hotpot_evidence_id
                c.setdefault("passage_id", c["source"])
                c.setdefault("clause_id", c["source"])
                c.setdefault("evidence_id", hotpot_evidence_id(c["source"]))
        sf = item.get("supporting_facts") or []
        if dataset_class == "hotpot" and sf:
            from hotpot_local_retrieval import hotpot_evidence_id
            normalized_sf = []
            for fact in sf:
                fact = dict(fact)
                source = fact.get("source") or fact.get("doc") or fact.get("document") or ""
                fact.setdefault("passage_id", source)
                fact.setdefault("clause_id", source)
                fact.setdefault("evidence_id", hotpot_evidence_id(source))
                normalized_sf.append(fact)
            sf = normalized_sf
        if (not sf) and chunks:
            if dataset_class == "hotpot":
                answer = (item.get("answer") or "").strip()
                ans_l = answer.lower()
                inferred = []
                if ans_l in ("yes", "no"):
                    names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", item.get("question", ""))
                    for c in chunks:
                        text = c.get("chunk") or c.get("text") or ""
                        if any(n in text for n in names if len(n) > 3):
                            inferred.append({"source": c["source"], "chunk": text})
                    sf = inferred[:4] if inferred else [chunks[0]]
                else:
                    for c in chunks:
                        text = c.get("chunk") or c.get("text") or ""
                        if ans_l and ans_l in text.lower():
                            inferred.append({"source": c["source"], "chunk": text})
                    sf = inferred if inferred else [chunks[0]]
                sf = [{"source": x.get("source", ""), "chunk": x.get("chunk", x.get("text", ""))} for x in sf]
            else:
                sf = item.get("supporting_facts") or item.get("chunks") or []
        item["_eval_supporting"] = sf
        return item
    
    def evaluate_system(self, rag_system: SCTRAGSystem, 
                       sample_size: int = 50, dataset_class: str = "ht",
                       method_name: str = "SCT-RAG",
                       output_subdir: str = "benchmark_output",
                       sample_offset: int = 0,
                       sample_mode: str = "auto",
                       sample_seed: int = 42) -> EvaluationResult:
        """评估RAG系统

        sample_mode:
          - auto: full evaluation or deterministic sampling with seed 42
          - slice: 按 id 排序后取 [offset, offset+size)（分片并行）
          - head: 取排序后前 sample_size 条（固定小测集）
        """
        logger.info(f"开始评估RAG系统，样本大小: {sample_size}, offset: {sample_offset}, mode: {sample_mode}")
        # 保留对 rag_system 的引用以便调用 LLM 评估接口
        self.rag_system = rag_system
        
        all_items = list(self.qa_dataset.values())
        all_items.sort(key=lambda x: x.get("id", ""))
        if sample_mode == "slice" or sample_offset > 0:
            end = min(sample_offset + sample_size, len(all_items))
            sample_items = all_items[sample_offset:end]
        elif sample_mode == "head":
            sample_items = all_items[:sample_size]
        elif len(all_items) > sample_size:
            import random
            random.seed(sample_seed)
            sample_items = random.sample(all_items, sample_size)
        else:
            sample_items = all_items
        
        self._last_sample_offset = sample_offset
        self._last_sample_mode = sample_mode

        all_metrics = {
            "retrieval_metrics": [],
            "generation_metrics": [],
            "performance_metrics": [],
        }
        per_item_details = []
        
        # 用于路由类型混淆矩阵统计
        type_confusion = defaultdict(lambda: defaultdict(int))

        for i, qa_item in enumerate(sample_items):
            logger.info(f"评估进度: {i+1}/{len(sample_items)}")
            
            try:
                qa_item = self._prepare_qa_item(qa_item, dataset_class)
                question = qa_item.get("question") or ""
                ground_truth = qa_item.get("answer") or ""
                supporting_facts = qa_item.get("_eval_supporting") or qa_item.get("supporting_facts", []) or qa_item.get("chunks", [])
                if not question:
                    logger.warning("跳过无 question 的样本: id=%s", qa_item.get("id"))
                    continue
                
                result = rag_system.process_query(question, dataset_class, qa_item=qa_item)

                # 工业集统一引用协议：显式 ev_ + 答案 [n] 映射到当前方法供给的条款。
                if dataset_class == "ht" and isinstance(result, dict):
                    generation = result.setdefault("generation", {})
                    chunks = collect_retrieved_chunks(result.get("retrieval", {}))
                    generation["citation_refs"] = explicit_clause_citation_refs(
                        generation.get("raw_response", "") or "",
                        chunks,
                    )

                # 记录路由类型混淆统计（若数据集提供 ground truth type）
                gt_type = normalize_paper_type(qa_item.get("type") or qa_item.get("question_type"))
                pred_type = normalize_paper_type(result.get("router_analysis", {}).get("question_type"))
                if gt_type and pred_type:
                    type_confusion[gt_type][pred_type] += 1
                
                metrics = self._calculate_metrics(
                    question=question,
                    ground_truth=ground_truth,
                    supporting_facts=supporting_facts,
                    system_result=result,
                    dataset_class=dataset_class,
                    qa_type=gt_type,
                )
                
                all_metrics["retrieval_metrics"].append(metrics["retrieval"])
                all_metrics["generation_metrics"].append(metrics["generation"])
                all_metrics["performance_metrics"].append(metrics["performance"])

                # 收集每项的引用提取与匹配详情，便于在结果中展示
                extracted = result.get("generation", {}).get("citation_llm_use", [])
                gt_docs = [ (sf.get("source") or "") for sf in supporting_facts ]
                matched = [d for d in gt_docs if any(docs_match(d, c) for c in extracted)]
                per_item_details.append({
                    "id": qa_item.get("id"),
                    "gt_type": gt_type,
                    "pred_type": pred_type,
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": result.get("generation", {}).get("answer", ""),
                    "extracted_citations": extracted,
                    "supporting_facts_docs": gt_docs,
                    "matched_docs": matched,
                    "metrics": metrics
                })
            except Exception as e:
                logger.error(f"评估项目失败: {e}")
                continue
        
        final_result = self._aggregate_metrics(all_metrics)
        # 将混淆矩阵附加到保存的结果中，但仍然返回 EvaluationResult 对象
        confusion_dict = {g: dict(d) for g, d in type_confusion.items()}
        self._save_evaluation_results(
            final_result, sample_size,
            confusion_matrix=confusion_dict,
            per_item_details=per_item_details,
            method_name=method_name,
            output_subdir=output_subdir,
        )
        return final_result
    
    def _calculate_metrics(self, question: str, ground_truth: str, 
                          supporting_facts: List[Dict], 
                          system_result: Dict,
                          dataset_class: str = "ht",
                          qa_type: str = None) -> Dict[str, Dict]:
        """计算单个查询的指标"""
        
        metrics = {
            "retrieval": {},
            "generation": {},
            "performance": {}
        }
        
        # 检索指标
        retrieval_data = system_result.get("retrieval", {})
        vector_results = collect_retrieved_chunks(retrieval_data)
        if dataset_class == "hotpot":
            # All HotPotQA methods are evaluated on canonical passage identity.
            from hotpot_local_retrieval import hotpot_evidence_id
            for result in vector_results:
                if isinstance(result, dict):
                    source = result.get("source") or (result.get("metadata") or {}).get("file_name") or ""
                    metadata = result.setdefault("metadata", {})
                    metadata.setdefault("passage_id", source)
                    metadata.setdefault("clause_id", source)
                    metadata.setdefault("evidence_id", hotpot_evidence_id(source))
                else:
                    source = getattr(result, "source", "") or ""
                    metadata = getattr(result, "metadata", None)
                    if not isinstance(metadata, dict):
                        metadata = {}
                        result.metadata = metadata
                    metadata.setdefault("passage_id", source)
                    metadata.setdefault("clause_id", source)
                    metadata.setdefault("evidence_id", hotpot_evidence_id(source))
        
        # 支持事实来源与段落（尝试多种可能的字段名）
        def _get_gt_para(sf: Dict) -> str:
            for k in ("chunk", "paragraph", "para", "text", "paragraph_text"):
                v = sf.get(k)
                if v:
                    return v
            # 有时 supporting_fact 可能包含嵌套结构
            if isinstance(sf.get("context"), str):
                return sf.get("context")
            return ""

        ground_truth_docs = [sf.get("source") or sf.get("doc") or sf.get("document") for sf in supporting_facts]
        ground_truth_paras = [_get_gt_para(sf) for sf in supporting_facts]

        retrieved_docs = [
            (r.get("source") if isinstance(r, dict) else getattr(r, "source", None))
            for r in vector_results
        ]
        # 取出检索段落文本用于段落级命中判断，尝试多种字段
        def _get_chunk_text(r: Any) -> str:
            if not r:
                return ""
            if isinstance(r, dict):
                for k in ("text_preview", "chunk_text", "chunk", "text", "content", "snippet"):
                    v = r.get(k)
                    if v:
                        return v
                return ""
            else:
                return getattr(r, "chunk_text", getattr(r, "chunk", getattr(r, "text_preview", ""))) or ""

        retrieved_paras = [_get_chunk_text(r) for r in vector_results]
        has_clause_gold = any(sf.get("evidence_id") or sf.get("clause_id") for sf in supporting_facts)
        if has_clause_gold:
            for k in (1, 5, 10):
                metrics["retrieval"].update(retrieval_at_k(vector_results, supporting_facts, k))
            metrics["retrieval"].update(
                same_std_ranking_metrics(vector_results, supporting_facts)
            )
        
        # 文档级命中（诊断用）
        doc_hit_1 = self._calculate_hit_at_k(retrieved_docs, ground_truth_docs, 1)
        doc_hit_3 = self._calculate_hit_at_k(retrieved_docs, ground_truth_docs, 3)
        doc_hit_5 = self._calculate_hit_at_k(retrieved_docs, ground_truth_docs, 5)
        metrics["retrieval"]["doc_hit_at_1"] = doc_hit_1
        metrics["retrieval"]["doc_hit_at_3"] = doc_hit_3
        metrics["retrieval"]["doc_hit_at_5"] = doc_hit_5

        # 段落级 hit@k & recall@k（论文 Hit@5 对齐 supporting clauses \bar{C}）
        para_hit_1 = self._calculate_para_hit_at_k(retrieved_paras, ground_truth_paras, 1)
        para_hit_5 = self._calculate_para_hit_at_k(retrieved_paras, ground_truth_paras, 5)
        para_hit_10 = self._calculate_para_hit_at_k(retrieved_paras, ground_truth_paras, 10)
        metrics["retrieval"]["para_hit_at_1"] = para_hit_1
        metrics["retrieval"]["para_hit_at_5"] = para_hit_5
        metrics["retrieval"]["para_hit_at_10"] = para_hit_10

        # 工业标准主指标：Hit@k = 条款/段落级，避免“同标准任意段落”虚高
        if dataset_class in ("ht", "indstd", "industrial"):
            metrics["retrieval"]["hit_at_1"] = para_hit_1
            metrics["retrieval"]["hit_at_3"] = self._calculate_para_hit_at_k(
                retrieved_paras, ground_truth_paras, 3
            )
            metrics["retrieval"]["hit_at_5"] = para_hit_5
        else:
            metrics["retrieval"]["hit_at_1"] = doc_hit_1
            metrics["retrieval"]["hit_at_3"] = doc_hit_3
            metrics["retrieval"]["hit_at_5"] = doc_hit_5
        
        metrics["retrieval"]["recall_at_1"] = self._calculate_recall_at_k(retrieved_docs, ground_truth_docs, 1)
        metrics["retrieval"]["recall_at_3"] = self._calculate_recall_at_k(retrieved_docs, ground_truth_docs, 3)
        metrics["retrieval"]["recall_at_5"] = self._calculate_recall_at_k(retrieved_docs, ground_truth_docs, 5)

        # 段落级 recall@k
        metrics["retrieval"]["para_recall_at_1"] = self._calculate_para_recall_at_k(retrieved_paras, ground_truth_paras, 1)
        metrics["retrieval"]["para_recall_at_5"] = self._calculate_para_recall_at_k(retrieved_paras, ground_truth_paras, 5)
        metrics["retrieval"]["para_recall_at_10"] = self._calculate_para_recall_at_k(retrieved_paras, ground_truth_paras, 10)
        
        metrics["retrieval"]["mrr"] = self._calculate_mrr(retrieved_docs, ground_truth_docs)
        # 上下文质量
        metrics["retrieval"]["context_precision"] = self._calculate_context_precision(retrieved_docs, ground_truth_docs)
        # 使用语义匹配计算 context_recall：比较检索到的段落与 ground-truth 段落的语义相似度
        metrics["retrieval"]["context_recall"] = self._calculate_context_recall(retrieved_paras, ground_truth_paras)
        # 论文对齐：相关性对黄金条款，不用题干关键词/标准号抬分
        metrics["retrieval"]["context_relevance"] = self._calculate_context_relevance(
            retrieved_paras, ground_truth_paras
        )
        
        # 生成指标
        generated_answer = system_result.get("generation", {}).get("answer", "")
        eval_answer = self._clean_answer_for_eval(generated_answer)
        eval_ground_truth = self._clean_answer_for_eval(ground_truth)
        
        if dataset_class == "hotpot":
            hotpot_score = self._hotpot_answer_score(eval_answer, eval_ground_truth)
            metrics["generation"]["em_score"] = hotpot_score
            metrics["generation"]["accuracy"] = hotpot_score
            metrics["generation"]["f1_score"] = hotpot_score
            metrics["generation"]["answer_similarity"] = hotpot_score
        else:
            gt_text = (eval_ground_truth or "").strip()
            gen_text = (eval_answer or "").strip()
            metrics["generation"]["em_score"] = 1.0 if gt_text == gen_text else 0.0
            # Final paper protocol: the frozen LLM judge supplies the answer score.
            accuracy_val = None
            try:
                if hasattr(self, 'rag_system') and getattr(self.rag_system, 'llm_generator', None):
                    acc = self.rag_system.llm_generator.evaluate_accuracy(question, eval_answer, eval_ground_truth)
                    if isinstance(acc, float) and acc >= 0.0:
                        accuracy_val = acc
            except Exception as e:
                logger.warning(f"LLM answer judge failed: {e}")

            # Final paper metrics: token F1 plus frozen LLM-judge accuracy.
            token_f1 = self._calculate_token_f1(eval_answer, eval_ground_truth)
            metrics["generation"]["token_f1"] = token_f1
            tok_prec, tok_rec = self._token_precision_recall(eval_answer, eval_ground_truth)
            metrics["generation"]["token_precision"] = tok_prec
            metrics["generation"]["token_recall"] = tok_rec
            llm_ok = isinstance(accuracy_val, float) and accuracy_val >= 0.0

            llm_score = float(accuracy_val) if llm_ok else None
            if has_clause_gold:
                metrics["generation"]["answer_judge_available"] = bool(llm_ok)
                metrics["generation"]["answer_judge_score"] = (
                    float(max(0.0, min(1.0, llm_score))) if llm_ok else None
                )
            # 主 Acc：LLM 判定生成答案是否回答问题且与金标要点一致，≥0.7 记 1
            metrics["generation"]["accuracy"] = (
                1.0 if llm_judge_pass(llm_score if llm_ok else None) else 0.0
            )
            metrics["generation"]["judge_acc"] = float(metrics["generation"]["accuracy"])
            # F1 = 标准 token-F1（与 Acc 解耦）
            metrics["generation"]["f1_score"] = float(token_f1)
            metrics["generation"]["answer_similarity"] = self._semantic_similarity(
                eval_answer, eval_ground_truth
            )
        citations = system_result.get("generation", {}).get("citation_llm_use", [])
        gt_docs = [ (sf.get("source") or sf.get("doc") or sf.get("document") or "") for sf in supporting_facts ]
        if has_clause_gold:
            citation_refs = system_result.get("generation", {}).get("citation_refs", [])
            if dataset_class == "hotpot":
                from hotpot_local_retrieval import hotpot_evidence_id
                # Hotpot answers cite numbered passages. Systems resolve those
                # numbers into source IDs in citation_llm_use; citation_refs is
                # reserved for explicit industrial clause IDs and is normally
                # empty here.
                citation_refs = system_result.get("generation", {}).get("citation_llm_use", [])
                citation_refs = [hotpot_evidence_id(ref) for ref in citation_refs if ref]
            clause_citation = citation_metrics(citation_refs, supporting_facts)
            metrics["generation"]["citation_precision"] = clause_citation.precision
            metrics["generation"]["citation_recall"] = clause_citation.recall
            metrics["generation"]["citation_f1"] = clause_citation.f1
            metrics["generation"]["exact_clause_match"] = clause_citation.exact_match
            metrics["generation"]["penalized_accuracy"] = penalized_accuracy(
                metrics["generation"].get("accuracy", 0.0), clause_citation.exact_match
            )
        metrics["generation"]["hallucination_rate"] = self._calculate_hallucination_rate(generated_answer, system_result)
        
        # 性能指标
        perf_data = system_result.get("performance", {})
        metrics["performance"]["retrieval_time"] = perf_data.get("retrieval_time", 0.0)
        gen_time = perf_data.get("generation_time")
        if gen_time is None:
            gen_time = float(perf_data.get("generation_time_initial", 0.0)) + float(
                perf_data.get("generation_time_final", 0.0)
            )
        metrics["performance"]["generation_time"] = float(gen_time or 0.0)
        metrics["performance"]["total_time"] = perf_data.get("total_time", 0.0)
        
        return metrics
    
    def _doc_in_list(self, doc: str, relevant: List[str]) -> bool:
        if not doc:
            return False
        for r in relevant:
            if docs_match(doc, r):
                return True
        return False

    def _calculate_hit_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if not retrieved:
            return 0.0
        
        top_k = retrieved[:k]
        for doc in top_k:
            if self._doc_in_list(doc, relevant):
                return 1.0
        return 0.0

    def _calculate_para_hit_at_k(self, retrieved_paras: List[str], relevant_paras: List[str], k: int) -> float:
        if not retrieved_paras:
            return 0.0

        top_k = retrieved_paras[:k]
        for para in top_k:
            for gt in relevant_paras:
                if self._is_paragraph_match(para, gt):
                    return 1.0
        return 0.0

    def _calculate_para_recall_at_k(self, retrieved_paras: List[str], relevant_paras: List[str], k: int) -> float:
        if not relevant_paras:
            return 0.0

        top_k = retrieved_paras[:k]
        found = 0
        for gt in relevant_paras:
            for para in top_k:
                if self._is_paragraph_match(para, gt):
                    found += 1
                    break
        return found / len(relevant_paras)

    def _para_eval_tokens(self, text: str) -> set:
        """条款匹配用 token：去掉标准号与泛化词，并用中文 2/3-gram 提高可比性。"""
        cleaned = _STD_TOKEN_RE.sub(" ", text or "")
        toks = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", cleaned))
        # 中文连续串再切 2/3-gram，避免整句成一个超长 token 无法重叠
        for run in re.findall(r"[\u4e00-\u9fff]{4,}", cleaned):
            for n in (2, 3):
                for i in range(0, len(run) - n + 1):
                    toks.add(run[i : i + n])
        return {t for t in toks if t not in _GENERIC_EVAL_TOKENS and not t.isdigit()}

    def _is_paragraph_match(self, a: str, b: str, threshold: float = 0.5) -> bool:
        """判断两个段落是否匹配：去掉标准号后做词重叠，避免仅因同标准号虚高命中。"""
        if not a or not b:
            return False
        a_tokens = self._para_eval_tokens(a)
        b_tokens = self._para_eval_tokens(b)
        if not a_tokens or not b_tokens:
            return False

        overlap = a_tokens.intersection(b_tokens)
        min_len = min(len(a_tokens), len(b_tokens))
        overlap_ratio = len(overlap) / max(min_len, 1)
        if overlap_ratio >= 0.35 and len(overlap) >= 3:
            return True

        jaccard = len(overlap) / max(len(a_tokens.union(b_tokens)), 1)
        if jaccard >= max(threshold * 0.7, 0.22) and len(overlap) >= 3:
            return True

        a2 = _STD_TOKEN_RE.sub(" ", a)
        b2 = _STD_TOKEN_RE.sub(" ", b)
        ratio = SequenceMatcher(None, a2, b2).ratio()
        return ratio >= 0.62

    def _calculate_context_relevance(self, retrieved_paras: List[str], reference) -> float:
        """
        计算检索上下文相关性（0-1）。

        论文对齐：reference 优先为黄金 supporting 段落列表；
        若传入问题字符串则剥离标准号后回退比较，避免题干标准号抬分。
        """
        if not retrieved_paras:
            return 0.0

        if isinstance(reference, str):
            ref_paras = [_STD_TOKEN_RE.sub(" ", reference).strip()]
        else:
            ref_paras = [
                _STD_TOKEN_RE.sub(" ", str(p)).strip()
                for p in (reference or [])
                if str(p or "").strip()
            ]
        ref_paras = [p for p in ref_paras if p]
        if not ref_paras:
            return 0.0

        sims = []
        for para in retrieved_paras:
            para_c = _STD_TOKEN_RE.sub(" ", str(para)).strip()
            if not para_c:
                continue
            best = 0.0
            for ref in ref_paras:
                try:
                    sim = self._semantic_similarity(ref, para_c)
                    best = max(best, float(sim))
                except Exception:
                    continue
            sims.append(max(0.0, min(1.0, best)))

        if not sims:
            return 0.0
        return float(np.mean(sims[:5]))

    def _calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_found = sum(1 for doc in relevant if self._doc_in_list(doc, top_k))
        return relevant_found / len(relevant)

    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        for i, doc in enumerate(retrieved):
            if self._doc_in_list(doc, relevant):
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_context_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        if not retrieved:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved if self._doc_in_list(doc, relevant))
        return relevant_retrieved / len(retrieved)

    def _calculate_context_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """上下文召回：段落文本用语义匹配；文档 id 用集合覆盖。"""
        if not relevant:
            return 0.0
        if not retrieved:
            return 0.0
        try:
            avg_len = sum(len(str(x)) for x in retrieved) / max(len(retrieved), 1)
        except Exception:
            avg_len = 0
        is_texts = avg_len > 30 or any(len(str(x)) > 30 for x in retrieved)
        if is_texts:
            scores = []
            for gt in relevant:
                best = 0.0
                for r in retrieved:
                    try:
                        sim = self._semantic_similarity(str(gt), str(r))
                        if sim > best:
                            best = sim
                    except Exception:
                        continue
                scores.append(best)
            return float(np.mean(scores)) if scores else 0.0
        relevant_set = set(relevant)
        retrieved_set = set(retrieved)
        matched = len(relevant_set.intersection(retrieved_set))
        return matched / len(relevant_set)
    
    def _answer_tokens(self, text: str) -> List[str]:
        """答案 token：去标准号后取中文 2-gram + 数值/英文词，用于标准 token-F1。"""
        cleaned = _STD_TOKEN_RE.sub(" ", text or "")
        cleaned = re.sub(r"\[\d+\]", " ", cleaned)
        cleaned = cleaned.lower()
        toks: List[str] = []
        toks.extend(re.findall(r"\d+(?:\.\d+)?(?:%|mm|cm|m|mpa|pa|hz|v|a|ω|℃|μm|um|db|次|倍)?", cleaned, flags=re.I))
        toks.extend(re.findall(r"[a-z][a-z0-9_\-/]{1,}", cleaned))
        for run in re.findall(r"[\u4e00-\u9fff]+", cleaned):
            if len(run) <= 2:
                if len(run) == 2 and run not in _GENERIC_EVAL_TOKENS:
                    toks.append(run)
                continue
            for i in range(len(run) - 1):
                bg = run[i : i + 2]
                if bg not in _GENERIC_EVAL_TOKENS:
                    toks.append(bg)
        return toks

    def _token_precision_recall(self, answer: str, ground_truth: str) -> tuple:
        """返回 (precision, recall)，供 Acc 近一致判定。"""
        from collections import Counter

        pred = self._answer_tokens(self._clean_answer_for_eval(answer))
        gold = self._answer_tokens(self._clean_answer_for_eval(ground_truth))
        if not gold and not pred:
            return 1.0, 1.0
        if not gold or not pred:
            return 0.0, 0.0
        pc, gc = Counter(pred), Counter(gold)
        overlap = sum((pc & gc).values())
        precision = overlap / max(sum(pc.values()), 1)
        recall = overlap / max(sum(gc.values()), 1)
        return float(precision), float(recall)

    def _calculate_token_f1(self, answer: str, ground_truth: str) -> float:
        """经典 token-level F1（预测 vs 金标），非 Acc/覆盖率再调和。"""
        precision, recall = self._token_precision_recall(answer, ground_truth)
        if precision + recall <= 0:
            return 0.0
        return float(max(0.0, min(1.0, 2 * precision * recall / (precision + recall))))

    def _clean_answer_for_eval(self, text: str) -> str:
        """Remove citation/format artifacts before answer-vs-reference scoring."""
        if not text:
            return ""
        cleaned = re.sub(r'【(?:证据|Evidence)】.*$', '', str(text), flags=re.DOTALL)
        cleaned = re.sub(r'【(?:答案|Answer)】', '', cleaned)
        cleaned = re.sub(r'\[\d+\]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _semantic_similarity(self, a: str, b: str) -> float:
        """语义相似度近似：使用 SequenceMatcher 比例作为简易语义相似度（0-1）"""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    
    def _calculate_hallucination_rate(self, answer: str, system_result: Dict) -> float:
        retrieved_entities = self._extract_entities_from_retrieval(system_result)
        answer_entities = self._extract_entities(answer)
        
        if not answer_entities:
            return 0.0
        
        hallucinated = sum(1 for entity in answer_entities if entity not in retrieved_entities)
        return hallucinated / len(answer_entities)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        phrases = []
        
        number_pattern = r'\d+\.?\d*\s*(mm|MPa|℃|°C|g/cm³|kN|小时|分钟)'
        phrases.extend(re.findall(number_pattern, text))
        
        standard_pattern = r'(HB|GB/T|GJB)\s*\d+[-\s]\d{4}'
        phrases.extend(re.findall(standard_pattern, text))
        
        material_pattern = r'[A-Z]{2,}\d+[A-Z]*'
        phrases.extend(re.findall(material_pattern, text))
        
        return list(set(phrases))
    
  
    def _extract_entities(self, text: str) -> List[str]:
        text = text or ""
        entities = []
        
        entities.extend(re.findall(r'(HB|GB/T|GJB)\s*\d+[-\s]\d{4}', text))
        entities.extend(re.findall(r'[A-Z]{2,}\d+[A-Z]*', text))
        entities.extend(re.findall(r'\d+\.?\d*', text))
        
        return list(set(entities))
    
    def _extract_entities_from_retrieval(self, system_result: Dict) -> Set[str]:
        entities = set()
        
        for chunk in system_result.get("retrieval", {}).get("vector_results", []):
            chunk_text = chunk.get("chunk_text", "")
            entities.update(self._extract_entities(chunk_text))
        
        kg_entities = system_result.get("retrieval", {}).get("kg_results", {}).get("entities", [])
        entities.update(kg_entities)
        
        return entities

    def _aggregate_metrics(self, all_metrics: Dict[str, List]) -> EvaluationResult:
        eval_result = EvaluationResult()
        
        retrieval_metrics = all_metrics["retrieval_metrics"]
        if retrieval_metrics:
            eval_result.hit_at_1 = np.mean([m.get("hit_at_1", 0) for m in retrieval_metrics])
            eval_result.hit_at_3 = np.mean([m.get("hit_at_3", 0) for m in retrieval_metrics])
            eval_result.hit_at_5 = np.mean([m.get("hit_at_5", 0) for m in retrieval_metrics])
            # 段落级指标
            eval_result.para_hit_at_1 = np.mean([m.get("para_hit_at_1", 0) for m in retrieval_metrics])
            eval_result.para_hit_at_5 = np.mean([m.get("para_hit_at_5", 0) for m in retrieval_metrics])
            eval_result.para_hit_at_10 = np.mean([m.get("para_hit_at_10", 0) for m in retrieval_metrics])
            eval_result.para_recall_at_1 = np.mean([m.get("para_recall_at_1", 0) for m in retrieval_metrics])
            eval_result.para_recall_at_5 = np.mean([m.get("para_recall_at_5", 0) for m in retrieval_metrics])
            eval_result.para_recall_at_10 = np.mean([m.get("para_recall_at_10", 0) for m in retrieval_metrics])
            eval_result.recall_at_1 = np.mean([m.get("recall_at_1", 0) for m in retrieval_metrics])
            eval_result.recall_at_3 = np.mean([m.get("recall_at_3", 0) for m in retrieval_metrics])
            eval_result.recall_at_5 = np.mean([m.get("recall_at_5", 0) for m in retrieval_metrics])
            eval_result.mrr = np.mean([m.get("mrr", 0) for m in retrieval_metrics])
            eval_result.context_precision = np.mean([m.get("context_precision", 0) for m in retrieval_metrics])
            eval_result.context_recall = np.mean([m.get("context_recall", 0) for m in retrieval_metrics])
            eval_result.context_relevance = np.mean([m.get("context_relevance", 0) for m in retrieval_metrics])
            for name in (
                "clause_hit_at_1", "clause_hit_at_5", "clause_hit_at_10",
                "clause_recall_at_1", "clause_recall_at_5", "clause_recall_at_10",
                "same_std_recall_at_1", "same_std_mrr", "gold_all_in_context",
            ):
                setattr(eval_result, name, np.mean([m.get(name, 0) for m in retrieval_metrics]))
        
        generation_metrics = all_metrics["generation_metrics"]
        if generation_metrics:
            eval_result.em_score = np.mean([m.get("em_score", 0) for m in generation_metrics])
            eval_result.f1_score = np.mean([m.get("f1_score", 0) for m in generation_metrics])
            judge_scores = [
                m.get("answer_judge_score")
                for m in generation_metrics
                if m.get("answer_judge_available") and m.get("answer_judge_score") is not None
            ]
            if any("answer_judge_available" in m for m in generation_metrics):
                eval_result.answer_judge_coverage = len(judge_scores) / len(generation_metrics)
                eval_result.answer_judge_score = (
                    float(np.mean(judge_scores)) if judge_scores else 0.0
                )
            # 主 Acc 只平均每题 0/1 阈值分，不用 LLM 连续分覆盖
            eval_result.accuracy = np.mean([m.get("accuracy", 0) for m in generation_metrics])
            eval_result.judge_acc = np.mean([m.get("judge_acc", 0) for m in generation_metrics])
            eval_result.citation_precision = np.mean([m.get("citation_precision", 0) for m in generation_metrics])
            eval_result.citation_recall = np.mean([m.get("citation_recall", 0) for m in generation_metrics])
            eval_result.citation_f1 = np.mean([m.get("citation_f1", 0) for m in generation_metrics])
            eval_result.exact_clause_match = np.mean([m.get("exact_clause_match", 0) for m in generation_metrics])
            eval_result.penalized_accuracy = np.mean([m.get("penalized_accuracy", 0) for m in generation_metrics])
            eval_result.hallucination_rate = np.mean([m.get("hallucination_rate", 0) for m in generation_metrics])
        
        performance_metrics = all_metrics["performance_metrics"]
        if performance_metrics:
            eval_result.retrieval_time = np.mean([m.get("retrieval_time", 0) for m in performance_metrics])
            eval_result.generation_time = np.mean([m.get("generation_time", 0) for m in performance_metrics])
            eval_result.total_time = np.mean([m.get("total_time", 0) for m in performance_metrics])
        return eval_result
    
    def _save_evaluation_results(self, result: EvaluationResult, sample_size: int,
                                 confusion_matrix: dict = None, per_item_details: List[Dict] = None,
                                 method_name: str = "SCT-RAG", output_subdir: str = "compare"):
        try:
            from project_config import RAG_RESULTS_DIR
            base_results = Path(RAG_RESULTS_DIR)
        except ImportError:
            base_results = Path(__file__).resolve().parent.parent / "results"

        output_dir = base_results / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_method = re.sub(r"[^a-zA-Z0-9_\-]", "_", method_name)
        offset_tag = f"_o{getattr(self, '_last_sample_offset', 0)}" if getattr(self, "_last_sample_offset", 0) > 0 else ""
        filename = f"{safe_method}_evaluation_{timestamp}_n{sample_size}{offset_tag}.json"
        filepath = output_dir / filename

        result_dict = result.to_dict()
        result_dict["sample_size"] = sample_size
        result_dict["timestamp"] = timestamp
        if confusion_matrix:
            result_dict["confusion_matrix"] = confusion_matrix
        if per_item_details:
            result_dict["per_item_details"] = per_item_details

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        self._generate_report(result_dict, filepath)

        # 生成按 QA 类型的单独报告：优先按 gt_type 分组；若没有 gt_type，则按 pred_type 分组
        try:
            if per_item_details:
                self._generate_type_reports(result_dict, filepath, per_item_details)
        except Exception as e:
            logger.warning(f"生成按类型报告失败: {e}")
    
    def _generate_report(self, result: Dict, filepath: Path):
        # 构建更详细的 Markdown 报告，包含段落级指标与混淆矩阵
        lines = []
        lines.append("# RAG系统评估报告")
        lines.append("")
        lines.append("## 基本信息")
        lines.append(f"- 评估时间: {result.get('timestamp', '')}")
        lines.append(f"- 样本大小: {result.get('sample_size', 0)}")
        lines.append("")
        lines.append("## 检索性能（文档级）")
        lines.append(f"- Hit@1: {result.get('hit_at_1', 0):.3f}")
        lines.append(f"- Hit@3: {result.get('hit_at_3', 0):.3f}")
        lines.append(f"- Hit@5: {result.get('hit_at_5', 0):.3f}")
        lines.append(f"- Recall@1: {result.get('recall_at_1', 0):.3f}")
        lines.append(f"- Recall@3: {result.get('recall_at_3', 0):.3f}")
        lines.append(f"- Recall@5: {result.get('recall_at_5', 0):.3f}")
        lines.append(f"- MRR: {result.get('mrr', 0):.3f}")
        lines.append("")
        lines.append("## 检索性能（段落级）")
        lines.append(f"- Para Hit@1: {result.get('para_hit_at_1', result.get('retrieval', {}).get('para_hit_at_1', 0)):.3f}")
        lines.append(f"- Para Hit@5: {result.get('para_hit_at_5', result.get('retrieval', {}).get('para_hit_at_5', 0)):.3f}")
        lines.append(f"- Para Hit@10: {result.get('para_hit_at_10', result.get('retrieval', {}).get('para_hit_at_10', 0)):.3f}")
        lines.append(f"- Para Recall@1: {result.get('para_recall_at_1', result.get('retrieval', {}).get('para_recall_at_1', 0)):.3f}")
        lines.append(f"- Para Recall@5: {result.get('para_recall_at_5', result.get('retrieval', {}).get('para_recall_at_5', 0)):.3f}")
        lines.append(f"- Para Recall@10: {result.get('para_recall_at_10', result.get('retrieval', {}).get('para_recall_at_10', 0)):.3f}")
        lines.append("")
        lines.append("## 检索上下文质量")
        lines.append(f"- 上下文精度: {result.get('context_precision', result.get('retrieval', {}).get('context_precision', 0)):.3f}")
        lines.append(f"- 上下文召回率: {result.get('context_recall', result.get('retrieval', {}).get('context_recall', 0)):.3f}")
        lines.append(f"- 上下文相关性: {result.get('context_relevance', result.get('retrieval', {}).get('context_relevance', 0)):.3f}")
        lines.append("")
        lines.append("## 生成质量")
        lines.append(f"- 精确匹配 (EM): {result.get('em_score', result.get('generation', {}).get('em_score', 0)):.3f}")
        lines.append(f"- F1分数: {result.get('f1_score', result.get('generation', {}).get('f1_score', 0)):.3f}")
        lines.append(f"- 准确率: {result.get('accuracy', result.get('generation', {}).get('accuracy', 0)):.3f}")
        if result.get("judge_acc") is not None:
            lines.append(f"- JudgeAcc(LLM≥0.7): {result.get('judge_acc', 0):.3f}")
        lines.append("")
        lines.append("## 可信度")
        lines.append(f"- Citation F1: {result.get('citation_f1', result.get('generation', {}).get('citation_f1', 0)):.3f}")
        lines.append(f"- 幻觉率: {result.get('hallucination_rate', result.get('generation', {}).get('hallucination_rate', 0)):.3f}")
        lines.append("")
        lines.append("## 路由混淆矩阵")
        cm = result.get('confusion_matrix', {})
        if cm:
            lines.append("")
            lines.append("| GT \\ Pred | " + " | ".join(sorted({p for preds in cm.values() for p in preds.keys()})) + " |")
            lines.append("|---" + "|---" * len({p for preds in cm.values() for p in preds.keys()}) + "|")
            preds_sorted = sorted({p for preds in cm.values() for p in preds.keys()})
            for gt in sorted(cm.keys()):
                row = [str(cm[gt].get(p, 0)) for p in preds_sorted]
                lines.append("| {} | {} |".format(gt, " | ".join(row)))
        else:
            lines.append("- 无混淆矩阵数据")
        lines.append("")
        lines.append("## 性能指标")
        lines.append(f"- 平均检索时间: {result.get('retrieval_time', result.get('performance', {}).get('retrieval_time', 0)):.3f}秒")
        lines.append(f"- 平均生成时间: {result.get('generation_time', result.get('performance', {}).get('generation_time', 0)):.3f}秒")
        lines.append(f"- 平均总时间: {result.get('total_time', result.get('performance', {}).get('total_time', 0)):.3f}秒")

        report = "\n".join(lines)
        report_path = filepath.with_suffix('.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"评估报告已生成: {report_path}")


    def _generate_type_reports(self, result: Dict, filepath: Path, per_item_details: List[Dict]):
        """为每个 QA 类型生成单独的评估报告（优先使用 gt_type）。
        同时生成按 pred_type 的诊断报告以便分析路由/分类误差。
        """
        base_dir = filepath.parent
        timestamp = result.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))

        # 准备按 gt_type / pred_type 分组
        groups_by_gt = {}
        groups_by_pred = {}
        for item in per_item_details:
            gt = item.get('gt_type') or ''
            groups_by_gt.setdefault(gt, []).append(item)
            pred = item.get('pred_type') or ''
            groups_by_pred.setdefault(pred, []).append(item)

        # 论文 per-type 分析以数据集标注类型为准
        primary_groups = groups_by_gt if groups_by_gt else groups_by_pred
        primary_key_name = 'gt_type' if groups_by_gt else 'pred_type'

        # 辅助函数：计算一组 items 的聚合指标
        def aggregate_items(items: List[Dict]) -> Dict:
            agg = {
                'count': len(items),
                'hit_at_1': 0.0, 'hit_at_3': 0.0, 'hit_at_5': 0.0,
                'recall_at_1': 0.0, 'recall_at_3': 0.0, 'recall_at_5': 0.0, 'mrr': 0.0,
                'para_hit_at_1': 0.0, 'para_hit_at_5': 0.0, 'para_hit_at_10': 0.0,
                'para_recall_at_1': 0.0, 'para_recall_at_5': 0.0, 'para_recall_at_10': 0.0,
                'clause_hit_at_1': 0.0, 'clause_hit_at_5': 0.0, 'clause_hit_at_10': 0.0,
                'clause_recall_at_1': 0.0, 'clause_recall_at_5': 0.0, 'clause_recall_at_10': 0.0,
                'context_precision': 0.0, 'context_recall': 0.0, 'context_relevance': 0.0,
                'em_score': 0.0, 'f1_score': 0.0, 'accuracy': 0.0,
                'citation_f1': 0.0, 'hallucination_rate': 0.0,
                'retrieval_time': 0.0, 'generation_time': 0.0, 'total_time': 0.0,
            }
            if not items:
                return agg

            n = len(items)
            for it in items:
                m_retr = it.get('metrics', {}).get('retrieval', {})
                m_gen = it.get('metrics', {}).get('generation', {})
                m_perf = it.get('metrics', {}).get('performance', {})

                agg['hit_at_1'] += m_retr.get('hit_at_1', 0)
                agg['hit_at_3'] += m_retr.get('hit_at_3', 0)
                agg['hit_at_5'] += m_retr.get('hit_at_5', 0)
                agg['recall_at_1'] += m_retr.get('recall_at_1', 0)
                agg['recall_at_3'] += m_retr.get('recall_at_3', 0)
                agg['recall_at_5'] += m_retr.get('recall_at_5', 0)
                agg['mrr'] += m_retr.get('mrr', 0)
                agg['para_hit_at_1'] += m_retr.get('para_hit_at_1', 0)
                agg['para_hit_at_5'] += m_retr.get('para_hit_at_5', 0)
                agg['para_hit_at_10'] += m_retr.get('para_hit_at_10', 0)
                agg['para_recall_at_1'] += m_retr.get('para_recall_at_1', 0)
                agg['para_recall_at_5'] += m_retr.get('para_recall_at_5', 0)
                agg['para_recall_at_10'] += m_retr.get('para_recall_at_10', 0)
                agg['clause_hit_at_1'] += m_retr.get('clause_hit_at_1', 0)
                agg['clause_hit_at_5'] += m_retr.get('clause_hit_at_5', 0)
                agg['clause_hit_at_10'] += m_retr.get('clause_hit_at_10', 0)
                agg['clause_recall_at_1'] += m_retr.get('clause_recall_at_1', 0)
                agg['clause_recall_at_5'] += m_retr.get('clause_recall_at_5', 0)
                agg['clause_recall_at_10'] += m_retr.get('clause_recall_at_10', 0)
                agg['context_precision'] += m_retr.get('context_precision', 0)
                agg['context_recall'] += m_retr.get('context_recall', 0)
                agg['context_relevance'] += m_retr.get('context_relevance', 0)

                agg['em_score'] += m_gen.get('em_score', 0)
                agg['f1_score'] += m_gen.get('f1_score', 0)
                agg['accuracy'] += m_gen.get('accuracy', 0)
                agg['citation_f1'] += m_gen.get('citation_f1', 0)
                agg['hallucination_rate'] += m_gen.get('hallucination_rate', 0)

                agg['retrieval_time'] += m_perf.get('retrieval_time', 0)
                agg['generation_time'] += m_perf.get('generation_time', 0)
                agg['total_time'] += m_perf.get('total_time', 0)

            # 平均化
            for k in list(agg.keys()):
                if k != 'count':
                    agg[k] = agg[k] / n

            return agg

        # 为分组生成 md 文件
        for t, items in primary_groups.items():
            label = t if t else 'UNKNOWN'
            stats = aggregate_items(items)
            md_lines = []
            md_lines.append(f"# 类型评估报告 — {primary_key_name}: {label}")
            md_lines.append("")
            md_lines.append(f"- 总样本数: {stats['count']}")
            md_lines.append("")
            md_lines.append("## 检索（论文主口径：条款身份）")
            md_lines.append(f"- Clause Hit@1: {stats['clause_hit_at_1']:.3f}")
            md_lines.append(f"- Clause Hit@5: {stats['clause_hit_at_5']:.3f}")
            md_lines.append(f"- Clause Hit@10: {stats['clause_hit_at_10']:.3f}")
            md_lines.append(f"- Clause Recall@1: {stats['clause_recall_at_1']:.3f}")
            md_lines.append(f"- Clause Recall@5: {stats['clause_recall_at_5']:.3f}")
            md_lines.append(f"- Clause Recall@10: {stats['clause_recall_at_10']:.3f}")
            md_lines.append("")
            md_lines.append("## Retrieval diagnostics (passage and source levels)")
            md_lines.append(f"- Hit@1: {stats['hit_at_1']:.3f}")
            md_lines.append(f"- Hit@3: {stats['hit_at_3']:.3f}")
            md_lines.append(f"- Hit@5: {stats['hit_at_5']:.3f}")
            md_lines.append(f"- Recall@1: {stats['recall_at_1']:.3f}")
            md_lines.append(f"- Recall@3: {stats['recall_at_3']:.3f}")
            md_lines.append(f"- Recall@5: {stats['recall_at_5']:.3f}")
            md_lines.append(f"- MRR: {stats['mrr']:.3f}")
            md_lines.append("")
            md_lines.append("## 检索 (段落级)")
            md_lines.append(f"- Para Hit@1: {stats['para_hit_at_1']:.3f}")
            md_lines.append(f"- Para Hit@5: {stats['para_hit_at_5']:.3f}")
            md_lines.append(f"- Para Hit@10: {stats['para_hit_at_10']:.3f}")
            md_lines.append(f"- Para Recall@1: {stats['para_recall_at_1']:.3f}")
            md_lines.append(f"- Para Recall@5: {stats['para_recall_at_5']:.3f}")
            md_lines.append(f"- Para Recall@10: {stats['para_recall_at_10']:.3f}")
            md_lines.append("")
            md_lines.append("## 上下文质量")
            md_lines.append(f"- 上下文精度: {stats['context_precision']:.3f}")
            md_lines.append(f"- 上下文召回: {stats['context_recall']:.3f}")
            md_lines.append(f"- 上下文相关性: {stats['context_relevance']:.3f}")
            md_lines.append("")
            md_lines.append("## 生成质量")
            md_lines.append(f"- EM: {stats['em_score']:.3f}")
            md_lines.append(f"- F1: {stats['f1_score']:.3f}")
            md_lines.append(f"- 准确率: {stats['accuracy']:.3f}")
            md_lines.append(f"- Citation F1: {stats['citation_f1']:.3f}")
            md_lines.append(f"- 幻觉率: {stats['hallucination_rate']:.3f}")
            md_lines.append("")
            md_lines.append("## 性能")
            md_lines.append(f"- 平均检索时间: {stats['retrieval_time']:.3f}s")
            md_lines.append(f"- 平均生成时间: {stats['generation_time']:.3f}s")
            md_lines.append(f"- 平均总时间: {stats['total_time']:.3f}s")
            type_report_path = base_dir / f"evaluation_{timestamp}_{label}.md"
            with open(type_report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))
