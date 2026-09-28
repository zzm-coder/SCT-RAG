"""Public, relocatable configuration for SCT-RAG.

Only settings required by the released pipeline are exposed. Secrets and
service addresses must be supplied through environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"

MINERU_ROOT = CODE_ROOT / "0_mineru_pdf"
DATA_GB_MD = MINERU_ROOT / "data_GB_md"
DATA_HB_MD = MINERU_ROOT / "data_HB_md"
DATA_MD_FILTERED = MINERU_ROOT / "data_md_filtered"
KG_DATA_DIR = CODE_ROOT / "1_extract_data" / "kg_data"
KG_ROOT = CODE_ROOT / "2_kg_construction"
KG_CLAUSE_CACHE_DIR = KG_ROOT / "kg_clause_cache"
KG_CLAUSE_VECTOR_DB_DIR = KG_ROOT / "clause_vector_db"
KG_VECTOR_DB_DIR = KG_CLAUSE_VECTOR_DB_DIR
HOTPOT_VECTOR_DB_DIR = KG_ROOT / "hotpot_evidence_assets"
HOTPOT_KG_JSON = HOTPOT_VECTOR_DB_DIR / "hotpot_knowledge_graph.json"

QA_DATA_ROOT = CODE_ROOT / "3_QA_creation" / "3_QA_data"
INDSTD_QA_DIR = QA_DATA_ROOT / "indstd_design_qa_clause"
INDSTD_QA_TRAIN = INDSTD_QA_DIR / "train.json"
INDSTD_QA_DEV = INDSTD_QA_DIR / "dev.json"
INDSTD_QA_TEST = INDSTD_QA_DIR / "test.json"
INDSTD_QA_RERANKER_TRAIN = INDSTD_QA_DIR / "train.json"
HOTPOT_DATA_DIR = DATA_ROOT / "hotpot"
HOTPOT_QA_50 = HOTPOT_DATA_DIR / "hotpotqa_50.json"

RAG_ROOT = CODE_ROOT / "4_RAG_method" / "SCT-RAG"
RAG_RESULTS_DIR = PROJECT_ROOT / "outputs"
SCT_RERANKER_MODEL = RAG_ROOT / "models" / "sct_reranker_500q"

# OpenAI-compatible APIs; deliberately no built-in private endpoint or key.
LLM_SERVICE_URL = os.environ.get("SCT_API_BASE_URL", "")
LLM_API_KEY = os.environ.get("SCT_API_KEY", "")
LLM_MODEL = os.environ.get("SCT_CHAT_MODEL", "")
EMBEDDING_API_BASE_URL = os.environ.get("SCT_EMBEDDING_API_BASE_URL", LLM_SERVICE_URL)
EMBEDDING_API_KEY = os.environ.get("SCT_EMBEDDING_API_KEY", LLM_API_KEY)
TEXT2VEC_MODEL = os.environ.get("SCT_EMBEDDING_MODEL", "")
CROSS_ENCODER_MODEL = os.environ.get("SCT_RERANKER_MODEL", str(SCT_RERANKER_MODEL))

NEO4J_URI = os.environ.get("SCT_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("SCT_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("SCT_NEO4J_PASSWORD", "")

