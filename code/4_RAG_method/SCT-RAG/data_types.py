"""系统配置和数据类型定义"""
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import os
from pathlib import Path

# Load only the public, relocatable configuration.
try:
    import sys
    _cfg_root = Path(__file__).resolve().parent.parent.parent  # code/
    if str(_cfg_root) not in sys.path:
        sys.path.insert(0, str(_cfg_root))
    from project_config import (
        NEO4J_URI as _NEO4J_URI, NEO4J_USER as _NEO4J_USER, NEO4J_PASSWORD as _NEO4J_PASSWORD,
        KG_CLAUSE_VECTOR_DB_DIR, KG_CLAUSE_CACHE_DIR,
        TEXT2VEC_MODEL, CROSS_ENCODER_MODEL, LLM_SERVICE_URL, LLM_API_KEY,
        LLM_MODEL, RAG_RESULTS_DIR, HOTPOT_VECTOR_DB_DIR, HOTPOT_KG_JSON,
    )
    _USE_PROJECT_CONFIG = True
except ImportError:
    _USE_PROJECT_CONFIG = False

@dataclass
class SystemConfig:
    """系统配置"""
    # Neo4j配置
    neo4j_uri: str = _NEO4J_URI if _USE_PROJECT_CONFIG else os.environ.get("SCT_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _NEO4J_USER if _USE_PROJECT_CONFIG else os.environ.get("SCT_NEO4J_USER", "neo4j")
    neo4j_password: str = _NEO4J_PASSWORD if _USE_PROJECT_CONFIG else os.environ.get("SCT_NEO4J_PASSWORD", "")
    
    # 向量库配置
    vector_db_path1: str = os.environ.get(
        "SCT_VECTOR_DB_PATH",
        # 条款级评测必须默认使用带 standard/clause/evidence_id 的可追溯索引。
        str(KG_CLAUSE_VECTOR_DB_DIR) if _USE_PROJECT_CONFIG else "",
    )
    vector_db_path2: str = str(HOTPOT_VECTOR_DB_DIR) if _USE_PROJECT_CONFIG else ""
    # OpenAI-compatible chat API. Values are required at runtime.
    llm_service_url: str = LLM_SERVICE_URL if _USE_PROJECT_CONFIG else os.environ.get("SCT_API_BASE_URL", "")
    llm_api_key: str = LLM_API_KEY if _USE_PROJECT_CONFIG else os.environ.get("SCT_API_KEY", "")
    llm_model: str = LLM_MODEL if _USE_PROJECT_CONFIG else os.environ.get("SCT_CHAT_MODEL", "")

    # Embedding API model identifier. The same identifier must be used to build
    # and query the FAISS index.
    semantic_model_path: str = TEXT2VEC_MODEL if _USE_PROJECT_CONFIG else os.environ.get("SCT_EMBEDDING_MODEL", "")
    # CrossEncoder 重排序模型（必须与 SentenceTransformer 区分，不能用 text2vec）
    cross_encoder_model: str = CROSS_ENCODER_MODEL if _USE_PROJECT_CONFIG else os.environ.get("SCT_RERANKER_MODEL", "")

    # HotpotQA英文证据库：英文MPNet双编码器。msmarco-*-cos-v5也是
    # SentenceTransformer而非CrossEncoder，不得强制转为随机初始化的分类头。
    semantic_model_path2: str = os.environ.get("SCT_HOTPOT_EMBEDDING_MODEL", TEXT2VEC_MODEL if _USE_PROJECT_CONFIG else "")
    cross_encoder_model2: str = ""

    # Generation limits used by the locked experiment protocol.
    llm_max_context_chars: int = 24000
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.0
    llm_seed: Optional[int] = None
    
    # Clause-linked triples and their entity index share one reproducible cache.
    kg_cache_path: str = str(KG_CLAUSE_CACHE_DIR) if _USE_PROJECT_CONFIG else ""
    kg_triplet_cache_path: str = os.environ.get(
        "SCT_KG_TRIPLET_CACHE_PATH",
        str(KG_CLAUSE_CACHE_DIR) if _USE_PROJECT_CONFIG else "",
    )
    hotpot_kg_cache_path: str = str(HOTPOT_VECTOR_DB_DIR) if _USE_PROJECT_CONFIG else ""

    # HotPotQA 专用配置
    hotpot_kg_json_path: str = str(HOTPOT_KG_JSON) if _USE_PROJECT_CONFIG else ""
    hotpot_neo4j_uri: str = os.environ.get("SCT_HOTPOT_NEO4J_URI", "bolt://localhost:7688")
    hotpot_neo4j_user: str = "neo4j"
    hotpot_neo4j_password: str = os.environ.get("SCT_HOTPOT_NEO4J_PASSWORD", "")
    
    # 输出配置
    output_path: str = str(RAG_RESULTS_DIR) if _USE_PROJECT_CONFIG else "outputs"
    
    # Retrieval pool sizes. Final SCT-RAG evidence uses K_tau=10/10/20.
    top_k_kg: int = 20
    top_k_vector: int = 20
    top_k_rerank: int = 20
    # dev 集选定：CrossEncoder 与原始 Dense/BM25/RRF 候选顺序的融合权重。
    rerank_ce_alpha: float = 0.5
    similarity_threshold: float = 0.6
    # 答案准确率：token-F1 / 语义相似度达到该阈值记 1，否则记 0
    accuracy_similarity_threshold: float = 0.6


def evidence_top_k(config=None) -> int:
    """Return the outer evidence cap; SCT-RAG applies K_tau before this cap."""
    if config is None:
        return 20
    return max(1, int(getattr(config, "top_k_vector", 20) or 20))
    
class QuestionType(Enum):
    """Paper-defined industrial question categories."""
    SINGLE_STANDARD = "single"
    CROSS_STANDARD = "cross"
    CORRELATION = "correlation"
    
    @classmethod
    def from_int(cls, value: int) -> 'QuestionType':
        mapping = {
            1: cls.SINGLE_STANDARD,
            2: cls.CROSS_STANDARD,
            3: cls.CORRELATION,
        }
        return mapping.get(value, cls.SINGLE_STANDARD)

@dataclass
class KGTriple:
    """知识图谱三元组"""
    head: str
    relation: str
    tail: str
    source: str
    paragraph: str = ""
    confidence: float = 1.0
    standard_id: str = ""
    clause_id: str = ""
    evidence_id: str = ""
    
    def to_dict(self):
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "source": self.source,
            "paragraph": self.paragraph,
            "confidence": self.confidence,
            "standard_id": self.standard_id,
            "clause_id": self.clause_id,
            "evidence_id": self.evidence_id,
        }

@dataclass
class KGResult:
    """知识图谱查询结果"""
    triples: List[KGTriple]
    entities: List[str]
    query_time: float = 0.0
    
    def to_natural_language(self) -> str:
        if not self.triples:
            return ""
        
        descriptions = []
        for triple in self.triples[:10]:
            if triple.paragraph:
                descriptions.append(f"{triple.head} {triple.relation} {triple.tail}（来源：{triple.source}，段落：{triple.paragraph[:100]}...）")
            else:
                descriptions.append(f"{triple.head} {triple.relation} {triple.tail}（来源：{triple.source}）")
        
        return "\n".join(descriptions)
    
    def to_dict(self):
        return {
            "triples": [t.to_dict() for t in self.triples],
            "entities": self.entities,
            "query_time": self.query_time
        }

@dataclass
class RetrievedChunk:
    """检索到的文档块"""
    chunk_id: str
    source: str
    chunk_text: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_source: str = ""
    
    def to_dict(self):
        metadata = dict(self.metadata or {})
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "chunk_text": self.chunk_text,
            "similarity_score": round(self.similarity_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "retrieval_source": self.retrieval_source,
            "metadata": metadata,
            "standard_id": metadata.get("standard_id", ""),
            "clause_id": metadata.get("clause_id", ""),
            "evidence_id": metadata.get("evidence_id", ""),
        }

@dataclass
class RAGContext:
    """RAG上下文"""
    question: str
    question_type: QuestionType
    kg_results: Optional[KGResult] = None
    vector_chunks: List[RetrievedChunk] = field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = field(default_factory=list)
    retrieval_time: float = 0.0
    def get_enhanced_context(self, doc_chunks=None) -> str:
        context_parts = []
        reports = getattr(self, "_community_reports", None) or []
        if reports:
            context_parts.append("【图谱社区摘要】")
            for i, rep in enumerate(reports, start=1):
                context_parts.append(f"G{i}. {rep}")
            context_parts.append("")
        # 1) KG 信息：按 source -> paragraph 分组，列出每个三元组（完整段落）
        if self.kg_results and self.kg_results.triples:
            from collections import defaultdict
            context_parts.append("【知识图谱信息】")
            grouped = defaultdict(lambda: defaultdict(list))
            for t in self.kg_results.triples:
                src = t.source or "unknown"
                para = t.paragraph or ""
                grouped[src][para].append(t)

            # KG 来源不再占用 [n]，避免与文档条款编号冲突；身份仍写出 ev_
            for src, para_map in grouped.items():
                context_parts.append(f"KG来源: {src}")
                for para_text, triples in para_map.items():
                    if para_text:
                        context_parts.append(f"段落: {para_text}")
                    for tri in triples:
                        identity = ""
                        if tri.evidence_id:
                            identity = (
                                f" [条款身份: {tri.standard_id}; Clause {tri.clause_id}; "
                                f"Evidence {tri.evidence_id}]"
                            )
                        context_parts.append(
                            f"- {tri.head} | {tri.relation} | {tri.tail} "
                            f"(confidence={tri.confidence}){identity}"
                        )
                    context_parts.append("")
        
        # 2) 文档段落：先使用重排序结果（若有），否则使用向量检索结果。
        # doc_chunks 仅用于生成侧重排（如真公式置顶），不改 Hit@5 检索集合。
        if doc_chunks is not None:
            docs = doc_chunks
        else:
            docs = self.reranked_chunks if self.reranked_chunks else self.vector_chunks
        if docs:
            context_parts.append("\n【相关文档段落】")
            # [n] 从 1 起只编号文档条款，与评测 [n]→evidence_id 映射一致
            for i, chunk in enumerate(docs, start=1):
                meta_info = chunk.metadata or {}
                eid = str(meta_info.get("evidence_id") or "").strip()
                ident = ""
                if eid:
                    ident = (
                        f" | {meta_info.get('standard_id', '')}; "
                        f"Clause {meta_info.get('clause_id', '')}; "
                        f"Evidence {eid}"
                    )
                context_parts.append(f"[{i}] 来源: {chunk.source}{ident}")
                # 单段截断，避免上下文过长触发 LLM 400
                chunk_text = chunk.chunk_text or ""
                if len(chunk_text) > 2000:
                    chunk_text = chunk_text[:2000] + "…"
                context_parts.append(f"内容: {chunk_text}")
                # 添加元数据引用（如存在）
                try:
                    if meta_info:
                        if eid:
                            context_parts.append(
                                "条款身份: "
                                f"{meta_info.get('standard_id', '')}; "
                                f"Clause {meta_info.get('clause_id', '')}; "
                                f"Evidence {eid}"
                            )
                        meta_str = ", ".join(f"{k}:{v}" for k, v in list(meta_info.items())[:6])
                        context_parts.append(f"元信息: {meta_str}")
                except Exception:
                    pass
                context_parts.append("---")

        return "\n".join(context_parts) if context_parts else "无相关上下文信息。"

@dataclass 
class LLMResponse:
    """LLM响应"""
    answer: str
    evidence_citations: List[str]
    raw_response: str
    generation_time: float

@dataclass
class EvaluationResult:
    """评估结果"""
    # 检索指标
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    # 段落级检索指标
    para_hit_at_1: float = 0.0
    para_hit_at_5: float = 0.0
    para_hit_at_10: float = 0.0
    para_recall_at_1: float = 0.0
    para_recall_at_5: float = 0.0
    para_recall_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    # Deterministic clause-ID retrieval metrics (new protocol).
    clause_hit_at_1: float = 0.0
    clause_hit_at_5: float = 0.0
    clause_hit_at_10: float = 0.0
    clause_recall_at_1: float = 0.0
    clause_recall_at_5: float = 0.0
    clause_recall_at_10: float = 0.0
    # 同标准内排序（A）：金标是否压过邻条、是否全进上下文
    same_std_recall_at_1: float = 0.0
    same_std_mrr: float = 0.0
    gold_all_in_context: float = 0.0
    
    # 上下文质量
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevance: float = 0.0
    
    # 生成质量
    em_score: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    answer_judge_score: float = 0.0
    answer_judge_coverage: float = 0.0
    judge_acc: float = 0.0  # 与 accuracy 相同：LLM 分 >0.7 记 1
    # 性能
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    
    # 可信度
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    exact_clause_match: float = 0.0
    penalized_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    def to_dict(self):
        return asdict(self)
