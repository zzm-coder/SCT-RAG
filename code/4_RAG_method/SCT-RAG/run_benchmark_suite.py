#!/usr/bin/env python3
"""
统一运行 SCT-RAG 与 compare_rag 基线实验。

示例：
  python run_benchmark_suite.py --dataset indstd --sample-size 200 --methods SCT-RAG
  python run_benchmark_suite.py --dataset hotpot --sample-size 50 --methods SCT-RAG,BM25,Hybrid-RAG
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import traceback
from pathlib import Path

# 路径初始化
SCT_RAG_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCT_RAG_ROOT.parent.parent
sys.path.insert(0, str(SCT_RAG_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from data_types import SystemConfig  # noqa: E402
from sct_rag_system import SCTRAGSystem
from evaluator import RAGEvaluator  # noqa: E402
from benchmark_paths import DATASET_QA_MAP, method_log_dir, results_dir  # noqa: E402
from benchmark_log_utils import bind_system_run_log  # noqa: E402
from experiment_preflight import validate_formal_dataset, write_run_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("benchmark_suite")

DATASET_MAP = {k: v for k, v in DATASET_QA_MAP.items()}

METHOD_REGISTRY = {
    # 最终 SCT-RAG：StdDirect + WithinStd + clause-linked KG
    "SCT-RAG": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_global_dense=False,
        use_std_direct=True, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=False, use_gold_type=False,
    ),
    "SCT-RAG-oracle": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_global_dense=False,
        use_std_direct=True, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True,
    ),
    "SCT-RAG-noRerank": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_global_dense=False,
        use_std_direct=True, use_within_std_dense=True, use_rerank_boost=False,
    ),
    "GraphRAG": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_global_dense=False,
        use_std_direct=False, use_within_std_dense=False, use_rerank_boost=False,
        use_graphrag_paper=True,
    ),
    "DPR": lambda cfg: SCTRAGSystem(
        cfg, use_kg=False, use_global_dense=True,
        use_std_direct=False, use_within_std_dense=False, use_rerank_boost=False,
    ),
    "BM25": lambda cfg: __import__("compare_rag.BM25.BM25", fromlist=["BM25system"]).BM25system(cfg),
    "Hybrid-RAG": lambda cfg: __import__("compare_rag.Hybrid_rag.hybrid_rag", fromlist=["HybridRAGsystem"]).HybridRAGsystem(cfg),
    "Adaptive-RAG": lambda cfg: _load_adaptive_rag().AdaptiveRAGSystem(
        cfg, use_kg=True, use_vector=True,
        adaptive_model_dir=os.environ.get(
            "SCT_ADAPTIVE_ROUTER_MODEL",
            str(SCT_RAG_ROOT / "models" / "adaptive_router"),
        ),
    ),
    "Self-RAG": lambda cfg: __import__(
        "compare_rag.Self_rag.self_rag_system", fromlist=["SelfRAGSystem"]
    ).SelfRAGSystem(cfg, use_vector=True),
    "FLARE": lambda cfg: __import__(
        "compare_rag.FLARE.flare_system", fromlist=["FLARESystem"]
    ).FLARESystem(cfg, use_vector=True),
    "CRAG": lambda cfg: __import__(
        "compare_rag.CRAG.crag_system", fromlist=["CRAGSystem"]
    ).CRAGSystem(cfg, use_vector=True),
    "Closed-book": lambda cfg: __import__(
        "compare_rag.ClosedBook.closed_book", fromlist=["ClosedBookSystem"]
    ).ClosedBookSystem(cfg),
}


def _load_adaptive_rag():
    return __import__(
        "compare_rag.Adaptive_rag.adaptive_rag_system", fromlist=["AdaptiveRAGSystem"]
    )


def build_system(method: str, config: SystemConfig):
    factory = METHOD_REGISTRY.get(method)
    if not factory:
        raise ValueError(f"未知方法: {method}")
    return factory(config)


def _attach_method_log(method: str, output_subdir: str, truncate: bool = True) -> Path:
    """为当前方法绑定独立文件日志与 jsonl 路径。"""
    import re
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", method)
    log_dir = method_log_dir(output_subdir)
    log_path = log_dir / f"{safe}.log"
    jsonl_path = log_dir / f"{safe}.jsonl"
    if truncate:
        log_path.write_text("", encoding="utf-8")
        jsonl_path.write_text("", encoding="utf-8")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    return jsonl_path


def run_one(
    method: str,
    qa_path: Path,
    sample_size: int,
    dataset_class: str,
    output_subdir: str,
    cross_encoder_model: str = "",
    semantic_model_path: str = "",
    llm_service_url: str = "",
    llm_model: str = "",
    sample_offset: int = 0,
    sample_mode: str = "auto",
    sample_seed: int = 42,
    llm_seed: int | None = None,
    llm_temperature: float = 0.0,
    top_k_kg: int | None = None,
    top_k_vector: int | None = None,
    top_k_rerank: int | None = None,
    rerank_ce_alpha: float | None = None,
):
    logger.info("=" * 60)
    logger.info(
        "方法: %s | 数据集: %s | 样本: %d | offset: %d | mode: %s",
        method, qa_path.name, sample_size, sample_offset, sample_mode,
    )
    jsonl_path = _attach_method_log(method, output_subdir, truncate=True)
    config = SystemConfig()
    config.llm_seed = llm_seed
    config.llm_temperature = float(llm_temperature)
    if top_k_kg is not None:
        config.top_k_kg = int(top_k_kg)
    if top_k_vector is not None:
        config.top_k_vector = int(top_k_vector)
    if top_k_rerank is not None:
        config.top_k_rerank = int(top_k_rerank)
    if rerank_ce_alpha is not None:
        config.rerank_ce_alpha = float(rerank_ce_alpha)
    config.output_path = str(results_dir() / output_subdir / "method_logs")
    if cross_encoder_model:
        if dataset_class == "ht":
            config.cross_encoder_model = cross_encoder_model
        else:
            config.cross_encoder_model2 = cross_encoder_model
        logger.info("使用覆盖 reranker: %s", cross_encoder_model)
    if semantic_model_path:
        if dataset_class == "ht":
            config.semantic_model_path = semantic_model_path
        else:
            config.semantic_model_path2 = semantic_model_path
        logger.info("使用覆盖 embedding model: %s", semantic_model_path)
    if llm_service_url:
        config.llm_service_url = llm_service_url
        logger.info("使用覆盖 LLM 服务: %s", llm_service_url)
    if llm_model:
        config.llm_model = llm_model
        logger.info("使用覆盖 LLM 模型: %s", llm_model)
    if not config.llm_service_url or not config.llm_model:
        raise RuntimeError(
            "Configure SCT_API_BASE_URL and SCT_CHAT_MODEL, or pass "
            "--llm-service-url and --llm-model."
        )
    chosen_embedding = (
        config.semantic_model_path if dataset_class == "ht"
        else config.semantic_model_path2
    )
    if method != "Closed-book" and not chosen_embedding:
        raise RuntimeError("Configure SCT_EMBEDDING_MODEL before retrieval experiments.")
    system = build_system(method, config)
    bind_system_run_log(system, jsonl_path, output_subdir, method)
    evaluator = RAGEvaluator(str(qa_path))
    result = evaluator.evaluate_system(
        system,
        sample_size=sample_size,
        dataset_class=dataset_class,
        method_name=method,
        output_subdir=output_subdir,
        sample_offset=sample_offset,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
    )
    logger.info(
        "%s 完成 | Acc=%.3f | PenAcc=%.3f | CiteF1=%.3f | ECM=%.3f | ClauseHit@5=%.3f",
        method,
        getattr(result, "accuracy", result.f1_score),
        result.penalized_accuracy,
        result.citation_f1,
        result.exact_clause_match,
        result.clause_hit_at_5,
    )
    if hasattr(system, "close"):
        try:
            system.close()
        except Exception:
            pass
    return result


def parse_methods(raw: str):
    if raw.lower() in {"all", "*"}:
        return [
            "SCT-RAG", "DPR", "BM25", "Hybrid-RAG", "GraphRAG",
            "Adaptive-RAG", "Self-RAG", "FLARE", "CRAG", "Closed-book",
        ]
    return [m.strip() for m in raw.split(",") if m.strip()]


def already_evaluated(
    method: str,
    output_subdir: str,
    sample_size: int,
    sample_offset: int = 0,
) -> bool:
    """检查是否已有该方法的 n{sample_size} 评估结果"""
    out_dir = CODE_ROOT / "4_RAG_method" / "results" / output_subdir
    if not out_dir.exists():
        return False
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", method)
    offset_pat = f"_o{sample_offset}" if sample_offset > 0 else ""
    matches = list(out_dir.glob(f"{safe}_evaluation_*_n{sample_size}{offset_pat}.json"))
    if matches:
        return True
    # 合并后的全量结果
    if sample_offset == 0:
        matches = list(out_dir.glob(f"{safe}_evaluation_*_n{sample_size}_merged.json"))
    return len(matches) > 0


def main():
    parser = argparse.ArgumentParser(description="SCT-RAG reproducible benchmark runner")
    parser.add_argument("--dataset", choices=list(DATASET_MAP.keys()), default="indstd")
    parser.add_argument("--qa-path", default="", help="直接指定 QA json 路径（优先于 --dataset）")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--sample-offset", type=int, default=0, help="分片起始下标（配合 --sample-mode slice）")
    parser.add_argument("--sample-mode", default="auto", choices=["auto", "slice", "head"],
                        help="auto=随机抽样(42); slice=按id切片; head=前N条")
    parser.add_argument("--methods", default="all", help="逗号分隔或 all")
    parser.add_argument("--output-subdir", default="", help="results 子目录，默认按 dataset 自动")
    parser.add_argument("--dataset-class", default="", help="ht 或 hotpot，默认可自动推断")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有同规模结果的方法")
    parser.add_argument("--cross-encoder-model", default="", help="覆盖 CrossEncoder/reranker 路径")
    parser.add_argument("--semantic-model-path", default="", help="override embedding API model identifier")
    parser.add_argument("--llm-service-url", default="", help="覆盖 OpenAI-compatible LLM 服务地址")
    parser.add_argument("--llm-model", default="", help="覆盖请求中的 model 字段（第二骨干）")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--top-k-kg", type=int, default=None)
    parser.add_argument("--top-k-vector", type=int, default=None)
    parser.add_argument("--top-k-rerank", type=int, default=None)
    parser.add_argument("--rerank-ce-alpha", type=float, default=None)
    parser.add_argument("--allow-unvalidated-dataset", action="store_true",
                        help="仅诊断运行；正式实验必须通过条款数据门禁")
    args = parser.parse_args()

    qa_path = Path(args.qa_path) if args.qa_path else DATASET_MAP[args.dataset]
    if not qa_path.exists():
        raise FileNotFoundError(f"数据集不存在: {qa_path}")

    output_subdir = args.output_subdir or f"compare_{args.dataset}"
    dataset_class = args.dataset_class or ("hotpot" if args.dataset == "hotpot" else "ht")
    methods = parse_methods(args.methods)

    gate = (
        validate_formal_dataset(qa_path, args.allow_unvalidated_dataset)
        if dataset_class == "ht"
        else {"status": "not_applicable", "reason": "non-industrial benchmark"}
    )
    write_run_manifest(
        results_dir() / output_subdir, qa_path, gate,
        {"suite": "benchmark", "sample_size": args.sample_size,
         "sample_offset": args.sample_offset, "sample_mode": args.sample_mode,
         "methods": methods, "dataset_class": dataset_class,
         "sample_seed": args.sample_seed, "llm_seed": args.llm_seed,
         "llm_temperature": args.llm_temperature,
         "top_k_kg": args.top_k_kg, "top_k_vector": args.top_k_vector,
         "top_k_rerank": args.top_k_rerank,
         "rerank_ce_alpha": args.rerank_ce_alpha},
    )

    summary = []
    for method in methods:
        if args.skip_existing and already_evaluated(
            method, output_subdir, args.sample_size, args.sample_offset
        ):
            logger.info("跳过已完成: %s", method)
            continue
        try:
            result = run_one(
                method,
                qa_path,
                args.sample_size,
                dataset_class,
                output_subdir,
                cross_encoder_model=args.cross_encoder_model,
                semantic_model_path=args.semantic_model_path,
                llm_service_url=args.llm_service_url,
                llm_model=args.llm_model,
                sample_offset=args.sample_offset,
                sample_mode=args.sample_mode,
                sample_seed=args.sample_seed,
                llm_seed=args.llm_seed,
                llm_temperature=args.llm_temperature,
                top_k_kg=args.top_k_kg,
                top_k_vector=args.top_k_vector,
                top_k_rerank=args.top_k_rerank,
                rerank_ce_alpha=args.rerank_ce_alpha,
            )
            summary.append(
                {
                    "method": method,
                    "accuracy": round(getattr(result, "accuracy", 0), 4),
                    "judge_coverage": round(result.answer_judge_coverage, 4),
                    "penalized_accuracy": round(result.penalized_accuracy, 4),
                    "citation_precision": round(result.citation_precision, 4),
                    "citation_recall": round(result.citation_recall, 4),
                    "citation_f1": round(result.citation_f1, 4),
                    "exact_clause_match": round(result.exact_clause_match, 4),
                    "clause_hit_at_5": round(result.clause_hit_at_5, 4),
                }
            )
        except Exception as e:
            logger.error("方法 %s 失败: %s", method, e)
            traceback.print_exc()
            summary.append({"method": method, "error": str(e)})

    out_dir = Path(CODE_ROOT) / "4_RAG_method" / "results" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime

    summary_path = out_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("汇总已保存: %s", summary_path)


if __name__ == "__main__":
    # 不覆盖外部传入的 CUDA_VISIBLE_DEVICES（支持双卡并行）
    main()
