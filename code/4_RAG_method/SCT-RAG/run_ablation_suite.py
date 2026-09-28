#!/usr/bin/env python3
"""SCT-RAG 消融实验：在指定测试集上对比各模块贡献。

支持：
- 串行跑全部变体（默认）
- --variant / --variants 只跑指定变体（供多进程并行调度）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

SCT_RAG_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCT_RAG_ROOT.parent.parent
sys.path.insert(0, str(SCT_RAG_ROOT))
sys.path.insert(0, str(CODE_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ablation")

from benchmark_paths import resolve_qa_path, default_indstd_dataset, results_dir, method_log_dir
from benchmark_log_utils import bind_system_run_log
from experiment_preflight import validate_formal_dataset, write_run_manifest

QA_PATH = resolve_qa_path(default_indstd_dataset())
OUT_SUBDIR = "ablation_indstd"

# 精简架构消融：仅公开 StdDirect + WithinStd + KG（及重排）
ABLATION_VARIANTS = {
    "SCT-RAG (oracle)": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=True, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True,
    ),
    "SCT-RAG (predicted)": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=True, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=False, use_gold_type=False,
    ),
    "w/o KG": lambda cfg: SCTRAGSystem(
        cfg, use_kg=False, use_std_direct=True, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True,
    ),
    "w/o WithinStd": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=True, use_within_std_dense=False, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True,
    ),
    "w/o StdDirect": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=False, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True,
    ),
    "w/o Rerank": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=True, use_within_std_dense=True, use_rerank_boost=False,
        oracle_routing=True, use_gold_type=True,
    ),
    "w/o StdDirect+WithinStd": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=False, use_within_std_dense=False, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True,
    ),
    "Unified-cross": lambda cfg: SCTRAGSystem(
        cfg, use_kg=True, use_std_direct=True, use_within_std_dense=True, use_rerank_boost=True,
        oracle_routing=True, use_gold_type=True, force_paper_type="cross",
    ),
}


def _safe_name(name: str) -> str:
    """与 evaluator / method_logs 文件名一致：非字母数字统一为下划线。"""
    import re
    s = name.replace(" ", "_").replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def _find_variant_eval(out_dir: Path, name: str, sample_size: int) -> Path | None:
    """Locate the evaluation JSON written for one named ablation."""
    safe = _safe_name(name)
    cands = sorted(out_dir.glob(f"{safe}_evaluation_*_n{sample_size}.json"))
    return cands[-1] if cands else None


def _select_variants(variant: str = "", variants: str = "") -> dict:
    """按名称筛选变体；空则返回全部。"""
    if not variant and not variants:
        return dict(ABLATION_VARIANTS)
    names: list[str] = []
    if variant.strip():
        names.append(variant.strip())
    if variants.strip():
        names.extend([x.strip() for x in variants.split(",") if x.strip()])
    # 去重且保序
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    out = {}
    for n in ordered:
        if n not in ABLATION_VARIANTS:
            raise SystemExit(f"未知消融变体: {n}; 可选: {list(ABLATION_VARIANTS)}")
        out[n] = ABLATION_VARIANTS[n]
    return out


def _load_existing_summary(path: Path, name: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "variant": name,
        "f1_score": data.get("f1_score", 0),
        "citation_precision": data.get("citation_precision", 0),
        "citation_recall": data.get("citation_recall", 0),
        "citation_f1": data.get("citation_f1", 0),
        "exact_clause_match": data.get("exact_clause_match", 0),
        "clause_hit_at_5": data.get("clause_hit_at_5", 0),
        "para_hit_at_5": data.get("para_hit_at_5", data.get("hit_at_5", 0)),
        "para_recall_at_5": data.get("para_recall_at_5", data.get("recall_at_5", 0)),
        "f1_score": data.get("f1_score", 0),
        "accuracy": data.get("accuracy", data.get("generation", {}).get("accuracy", 0)),
        "answer_judge_coverage": data.get("answer_judge_coverage", 0),
        "penalized_accuracy": data.get("penalized_accuracy", 0),
        "total_time": data.get("total_time", 0),
    }


def write_ablation_summary(summary: list[dict], sample_size: int, out_name: str) -> Path:
    """写出消融汇总 JSON + Markdown 表。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir() / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ablation_summary_{ts}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [
        f"# SCT-RAG 消融实验 (n={sample_size})",
        "",
        "> Total(s) 为每题平均端到端耗时（秒）。",
        "",
        "| 变体 | Hit@5 | Recall@5 | Acc | F1 | Citation F1 | ECM | PenAcc | Time |",
        "|------|------:|---------:|----:|----:|------------:|----:|-------:|-----:|",
    ]
    for r in summary:
        if "error" in r:
            md.append(f"| {r['variant']} | ERR | | | | | | | {r['error'][:40]} |")
        else:
            md.append(
                f"| {r['variant']} | {float(r.get('para_hit_at_5') or r.get('clause_hit_at_5') or 0):.3f} | "
                f"{float(r.get('para_recall_at_5') or 0):.3f} | {r['accuracy']:.3f} | "
                f"{float(r.get('f1_score') or 0):.3f} | {r['citation_f1']:.3f} | "
                f"{r['exact_clause_match']:.3f} | {r['penalized_accuracy']:.3f} | "
                f"{float(r.get('total_time') or 0):.2f} |"
            )
    md_path = out_dir / f"ABLATION_TABLE_{ts}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("消融表: %s", md_path)
    print("\n".join(md))
    return md_path


def collect_ablation_summary(out_name: str, sample_size: int) -> list[dict]:
    """从已完成的 evaluation JSON 汇总全部变体（用于并行跑完后汇总）。"""
    out_dir = results_dir() / out_name
    summary = []
    for name in ABLATION_VARIANTS:
        path = _find_variant_eval(out_dir, name, sample_size)
        if not path:
            summary.append({"variant": name, "error": "missing"})
            continue
        summary.append(_load_existing_summary(path, name))
    return summary


def run_ablation(
    sample_size: int = 200,
    skip_existing: bool = False,
    qa_path: str | None = None,
    output_subdir: str | None = None,
    cross_encoder_model: str | None = None,
    variant: str = "",
    variants: str = "",
    write_summary: bool = True,
    allow_unvalidated_dataset: bool = False,
):
    # Gate data before importing GPU/Neo4j/model dependencies.  This keeps data
    # integrity failures explicit and lets summary-only use remain lightweight.
    qa_file = Path(qa_path) if qa_path else QA_PATH
    out_name = output_subdir or OUT_SUBDIR
    gate = validate_formal_dataset(qa_file, allow_unvalidated_dataset)
    write_run_manifest(
        results_dir() / out_name,
        qa_file,
        gate,
        {"suite": "ablation", "sample_size": sample_size, "variants": list(_select_variants(variant, variants))},
    )
    global SCTRAGSystem
    from data_types import SystemConfig
    from evaluator import RAGEvaluator
    from sct_rag_system import SCTRAGSystem
    selected = _select_variants(variant, variants)
    evaluator = RAGEvaluator(str(qa_file))
    summary = []
    for name, factory in selected.items():
        safe = _safe_name(name)
        out_dir = results_dir() / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = list(out_dir.glob(f"{safe}_evaluation_*_n{sample_size}.json"))
        if skip_existing and existing:
            logger.info("跳过已完成: %s", name)
            summary.append(_load_existing_summary(existing[-1], name))
            continue
        logger.info("=" * 50 + " 消融: %s", name)
        try:
            cfg = SystemConfig()
            cfg.top_k_vector = 20
            cfg.top_k_rerank = 20
            # Forward the explicitly configured OpenAI-compatible endpoint.
            llm_url = os.environ.get("SCT_API_BASE_URL")
            if llm_url:
                cfg.llm_service_url = llm_url
            if cross_encoder_model:
                cfg.cross_encoder_model = cross_encoder_model
            if not cfg.llm_service_url or not cfg.llm_model or not cfg.semantic_model_path:
                raise RuntimeError(
                    "SCT_API_BASE_URL, SCT_CHAT_MODEL, and SCT_EMBEDDING_MODEL are required"
                )
            system = factory(cfg)
            jsonl_path = method_log_dir(out_name) / f"{safe}.jsonl"
            jsonl_path.write_text("", encoding="utf-8")
            bind_system_run_log(system, jsonl_path, out_name, safe)
            result = evaluator.evaluate_system(
                system, sample_size=sample_size, dataset_class="ht",
                method_name=safe, output_subdir=out_name, sample_mode="head",
            )
            summary.append({
                "variant": name,
                "citation_precision": round(result.citation_precision, 4),
                "citation_recall": round(result.citation_recall, 4),
                "citation_f1": round(result.citation_f1, 4),
                "exact_clause_match": round(result.exact_clause_match, 4),
                "clause_hit_at_5": round(result.clause_hit_at_5, 4),
                "para_hit_at_5": round(getattr(result, "para_hit_at_5", 0) or 0, 4),
                "para_recall_at_5": round(getattr(result, "para_recall_at_5", 0) or 0, 4),
                "f1_score": round(result.f1_score, 4),
                "accuracy": round(getattr(result, "accuracy", result.f1_score), 4),
                "answer_judge_coverage": round(result.answer_judge_coverage, 4),
                "penalized_accuracy": round(result.penalized_accuracy, 4),
                "total_time": round(result.total_time, 4),
            })
            if hasattr(system, "close"):
                system.close()
        except Exception as e:
            logger.error("消融失败 %s: %s", name, e)
            traceback.print_exc()
            summary.append({"variant": name, "error": str(e)})

    if write_summary and (not variant and not variants):
        # 全量串行跑完才写总表；单变体由并行调度器汇总
        return write_ablation_summary(summary, sample_size, out_name)
    if write_summary and summary:
        # 单变体也可写局部小表，便于排查
        logger.info("单变体结果: %s", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--qa-path", default="")
    p.add_argument("--output-subdir", default="")
    p.add_argument("--cross-encoder-model", default="")
    p.add_argument("--variant", default="", help="只跑一个变体，名称需与 ABLATION_VARIANTS 键一致")
    p.add_argument("--variants", default="", help="逗号分隔的多个变体")
    p.add_argument("--write-summary", action="store_true", help="强制写出汇总表（含单变体模式）")
    p.add_argument("--summarize-only", action="store_true", help="仅从已有 JSON 汇总消融表")
    p.add_argument(
        "--allow-unvalidated-dataset", action="store_true",
        help="仅用于诊断：允许数据门禁失败，并在 manifest 标记非正式运行",
    )
    args = p.parse_args()
    out_name = args.output_subdir or OUT_SUBDIR
    if args.summarize_only:
        summary = collect_ablation_summary(out_name, args.sample_size)
        write_ablation_summary(summary, args.sample_size, out_name)
        return
    run_ablation(
        args.sample_size,
        args.skip_existing,
        qa_path=args.qa_path or None,
        output_subdir=args.output_subdir or None,
        cross_encoder_model=args.cross_encoder_model or None,
        variant=args.variant,
        variants=args.variants,
        write_summary=args.write_summary or (not args.variant and not args.variants),
        allow_unvalidated_dataset=args.allow_unvalidated_dataset,
    )


if __name__ == "__main__":
    main()
