# -*- coding: utf-8 -*-
"""无检索 closed-book 基线：仅用骨干模型参数知识作答，用于估计记忆贡献。"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from data_types import SystemConfig
from llm_generator import LLMGenerator
from compare_rag.utils import baseline_router_stub

logger = logging.getLogger(__name__)


class ClosedBookSystem:
    """空上下文生成。Hit@5 按定义应为 0；非零 Acc/CiteAcc 视为参数记忆。"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.llm_generator = LLMGenerator(config)

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> Dict[str, Any]:
        start = time.time()
        gen_start = time.time()
        llm_response = self.llm_generator.generate_closed_book(
            question, dataset_class=dataset_class
        )
        generation_time = time.time() - gen_start
        citations = list(getattr(llm_response, "evidence_citations", None) or [])
        result = {
            "question": question,
            "router_analysis": baseline_router_stub("Closed-book"),
            "retrieval": {
                "vector_results": [],
                "reranked_results": [],
                "retrieval_time": 0.0,
            },
            "generation": {
                "answer": llm_response.answer,
                "citation_llm_use": citations,
                "generation_time": generation_time,
                "raw_response": llm_response.raw_response,
            },
            "performance": {
                "retrieval_time": 0.0,
                "generation_time": generation_time,
                "total_time": time.time() - start,
            },
        }
        from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
        append_run_jsonl(resolve_run_jsonl(
            self,
            getattr(self, "_run_method_name", "Closed-book"),
            getattr(self, "_output_subdir", ""),
        ), result)
        return result
