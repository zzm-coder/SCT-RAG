#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用官方 ragas 包离线重算 RAGAS-F/C/A（不重跑检索/生成）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "outputs"

METHODS = [
    "SCT-RAG", "DPR", "BM25", "Hybrid-RAG", "GraphRAG",
    "Adaptive-RAG", "Self-RAG", "FLARE", "CRAG", "Closed-book",
]


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"【(?:证据|Evidence)】.*$", "", str(text), flags=re.DOTALL)
    cleaned = re.sub(r"【(?:答案|Answer)】", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def retrieval_chunks(record: dict) -> list:
    ret = record.get("retrieval") or {}
    for key in ("reranked_results", "combined_results", "vector_results", "bm25_results"):
        chunks = ret.get(key)
        if chunks:
            return list(chunks)
    return []


def chunk_text(chunk: dict) -> str:
    if not isinstance(chunk, dict):
        return str(chunk or "").strip()
    return str(
        chunk.get("chunk_text")
        or chunk.get("original_text")
        or chunk.get("text_preview")
        or chunk.get("text")
        or ""
    ).strip()


def latest_eval(eval_dir: Path, method: str, n: int) -> Path | None:
    safe = method.replace("/", "_")
    cands = sorted(eval_dir.glob(f"{safe}_evaluation_*_n{n}.json"))
    return cands[-1] if cands else None


def load_eval_rows(eval_path: Path) -> list[dict]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    return list(data.get("per_item_details") or [])


def load_log_by_question(jsonl: Path) -> dict:
    out = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        q = str(rec.get("question") or "").strip()
        if q:
            out[q] = rec
    return out


def build_dataset_rows(eval_rows: list[dict], logs: dict, top_k: int) -> tuple[list[dict], dict]:
    rows = []
    missing_ctx = 0
    empty_ctx = 0
    for item in eval_rows:
        q = str(item.get("question") or "").strip()
        rec = logs.get(q) or {}
        texts = []
        seen = set()
        for ch in retrieval_chunks(rec)[:top_k]:
            t = chunk_text(ch)
            if not t or t in seen:
                continue
            seen.add(t)
            texts.append(t)
        if not rec:
            missing_ctx += 1
        if not texts:
            empty_ctx += 1
            texts = [""]
        ans = _clean_answer(item.get("generated_answer") or "")
        gt = _clean_answer(item.get("ground_truth") or "")
        rows.append({
            "question": q,
            "answer": ans,
            "contexts": texts,
            "ground_truth": gt,
            "id": item.get("id"),
        })
    meta = {"n": len(rows), "missing_log": missing_ctx, "empty_context": empty_ctx}
    return rows, meta


def make_hf_dataset(rows: list[dict]):
    from datasets import Dataset
    return Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })


def init_judge(llm_base: str, llm_model: str, embed_path: str):
    """Bind official RAGAS metrics to OpenAI-compatible chat and embedding APIs."""
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("SCT_API_KEY") or "")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI
    request_timeout = int(os.environ.get("SCT_RAGAS_REQUEST_TIMEOUT", "180") or 180)
    # Bound intermediate statement extraction. Without this cap, malformed or
    # repetitive generations can make the following faithfulness NLI prompt
    # exceed the judge model's context window.
    max_tokens = int(os.environ.get("SCT_RAGAS_MAX_TOKENS", "2048") or 2048)
    common = dict(temperature=0.0, max_retries=2, max_tokens=max_tokens)
    try:
        llm = ChatOpenAI(
            openai_api_base=llm_base.rstrip("/"),
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_name=llm_model,
            request_timeout=request_timeout,
            model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            **common,
        )
    except TypeError:
        llm = ChatOpenAI(
            base_url=llm_base.rstrip("/"),
            api_key=os.environ["OPENAI_API_KEY"],
            model=llm_model,
            timeout=request_timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **common,
        )

    from api_embedding import APIEmbeddingModel
    embeddings = APIEmbeddingModel(embed_path) if embed_path else None
    from ragas.llms.output_parser import RagasoutputParser
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.metrics._faithfulness import Faithfulness

    def _normalize_json_result(text: str) -> str:
        raw = (text or "").strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except Exception:
            m2 = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if not m2:
                return text
            data = json.loads(m2.group(1))
        if isinstance(data, dict):
            for key in ("statements", "__root__", "analysis", "items", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                if "verdict" in data or "sentence_index" in data:
                    data = [data]
        return json.dumps(data, ensure_ascii=False)

    _orig_parse = RagasoutputParser.parse

    def _parse(self, text):  # noqa: ANN001
        try:
            text = _normalize_json_result(text)
        except Exception:
            pass
        return _orig_parse(self, text)

    RagasoutputParser.parse = _parse

    def _create_statements_prompt_cjk(self, row):
        """官方 faithfulness 原实现只保留英文句号结尾；工业标准答案多为中文句号，这里按官方分段逻辑补上 CJK 终止符。"""
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"
        answer, question = row["answer"], row["question"]
        sentences = self.sentence_segmenter.segment(answer)
        kept = []
        for sentence in sentences:
            s = str(sentence or "").strip()
            if not s:
                continue
            if s[-1] in ".。!！?？;；" or len(s) >= 8:
                kept.append(s)
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(kept)])
        return self.statement_prompt.format(
            question=question, answer=answer, sentences=sentences
        )

    Faithfulness._create_statements_prompt = _create_statements_prompt_cjk
    extra = (
        " The answer, context, and statements may be in Chinese. "
        "Keep extracted statements in the same language as the answer; do not translate. "
        "Judge semantic entailment, not language identity."
    )
    if extra not in str(getattr(faithfulness.statement_prompt, "instruction", "")):
        faithfulness.statement_prompt.instruction = (
            faithfulness.statement_prompt.instruction + extra
        )
        faithfulness.nli_statements_message.instruction = (
            faithfulness.nli_statements_message.instruction + extra
        )
    metrics = [faithfulness, context_precision, answer_relevancy]
    names = ["faithfulness", "context_precision", "answer_relevancy"]
    return llm, embeddings, metrics, names


def run_ragas(ds, metrics, names, llm, embeddings, *, timeout: int = 180, max_workers: int = 12):
    from ragas import evaluate
    from ragas.run_config import RunConfig

    result = evaluate(
        ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(timeout=timeout, max_retries=2, max_workers=max_workers),
    )
    df = result.to_pandas() if hasattr(result, "to_pandas") else None
    scores = {}
    if hasattr(result, "scores") and isinstance(result.scores, dict):
        scores = {k: float(v) for k, v in result.scores.items() if v is not None}
    elif isinstance(result, dict):
        scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    per_item = []
    if df is not None:
        recs = df.to_dict(orient="records")
        for rec in recs:
            per_item.append({k: rec.get(k) for k in names if k in rec})
        if not scores:
            for name in names:
                vals = [r.get(name) for r in recs if isinstance(r.get(name), (int, float))]
                if vals:
                    scores[name] = sum(vals) / len(vals)
    return scores, per_item


def map_fca(scores: dict, names: list[str]) -> dict:
    """映射到主表 RAGAS-F/C/A。"""
    f_key = "faithfulness"
    a_key = "answer_relevancy" if "answer_relevancy" in scores else "answer_relevance"
    c_key = None
    for k in ("context_relevancy", "nv_context_relevance", "context_precision", "context_relevance"):
        if k in scores:
            c_key = k
            break
    return {
        "ragas_faithfulness": scores.get(f_key),
        "ragas_context_relevance": scores.get(c_key) if c_key else None,
        "ragas_answer_relevance": scores.get(a_key),
        "official_keys": {"F": f_key, "C": c_key, "A": a_key, "metric_names": names},
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_eval_json(eval_path: Path, mapped: dict, per_item: list, meta: dict) -> None:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    data["ragas_faithfulness"] = mapped.get("ragas_faithfulness")
    data["ragas_context_relevance"] = mapped.get("ragas_context_relevance")
    data["ragas_answer_relevance"] = mapped.get("ragas_answer_relevance")
    data["official_ragas_meta"] = meta
    details = data.get("per_item_details") or []
    for i, row in enumerate(details):
        if i < len(per_item):
            row["official_ragas"] = per_item[i]
    eval_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-subdir", default="main_indstd")
    parser.add_argument(
        "--log-subdir", default="",
        help="directory containing method_logs; defaults to --eval-subdir",
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help=">0 时只评前 N 题，用于试跑")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--llm-base", default=os.environ.get("SCT_API_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.environ.get("SCT_CHAT_MODEL", ""))
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("SCT_EMBEDDING_MODEL", ""),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not args.llm_base or not args.llm_model or not args.embed_model:
        parser.error("SCT_API_BASE_URL, SCT_CHAT_MODEL, and SCT_EMBEDDING_MODEL are required")

    import ragas
    llm, embeddings, metrics, names = init_judge(args.llm_base, args.llm_model, args.embed_model)
    eval_dir = RESULTS / args.eval_subdir
    frozen_dir = RESULTS / (args.log_subdir or args.eval_subdir) / "method_logs"
    out_dir = Path(args.output) if args.output else eval_dir / "official_ragas"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        eval_path = latest_eval(eval_dir, method, args.n)
        log_path = frozen_dir / f"{method}.jsonl"
        if not eval_path or not log_path.exists():
            print(f"[skip] {method}: eval={eval_path} log={log_path.exists()}", flush=True)
            continue
        sidecar = out_dir / f"{method}_official_ragas.json"
        if (
            not args.limit
            and sidecar.exists()
        ):
            prev = json.loads(sidecar.read_text(encoding="utf-8"))
            if len(prev.get("per_item") or []) >= args.n:
                print(f"[resume] {method}", flush=True)
                mapped = prev.get("mapped") or map_fca(prev.get("scores") or {}, names)
                table.append({
                    "method": method,
                    "ragas_faithfulness": mapped.get("ragas_faithfulness"),
                    "ragas_context_relevance": mapped.get("ragas_context_relevance"),
                    "ragas_answer_relevance": mapped.get("ragas_answer_relevance"),
                    "raw_scores": prev.get("scores"),
                })
                continue
        eval_rows = load_eval_rows(eval_path)
        if args.limit:
            eval_rows = eval_rows[: args.limit]
        logs = load_log_by_question(log_path)
        rows, row_meta = build_dataset_rows(eval_rows, logs, args.top_k)
        print(f"[run] {method} n={len(rows)} empty_ctx={row_meta['empty_context']}", flush=True)
        ds = make_hf_dataset(rows)
        scores, per_item = run_ragas(ds, metrics, names, llm, embeddings)
        mapped = map_fca(scores, names)
        meta = {
            "package": "ragas",
            "version": getattr(ragas, "__version__", ""),
            "judge_model": args.llm_model,
            "judge_base": args.llm_base,
            "embedding_model": args.embed_model,
            "top_k": args.top_k,
            "n": len(rows),
            "eval_file": eval_path.name,
            "eval_sha256": sha256_file(eval_path),
            "log_file": str(log_path),
            "log_sha256": sha256_file(log_path),
            "row_meta": row_meta,
            "raw_scores": scores,
            "mapped": mapped["official_keys"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "官方 ragas 包；答案来自评测 JSON，上下文来自冻结检索日志。",
        }
        (out_dir / f"{method}_official_ragas.json").write_text(
            json.dumps({"method": method, "scores": scores, "mapped": mapped, "meta": meta, "per_item": per_item},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not args.limit:
            patch_eval_json(eval_path, mapped, per_item, meta)
        table.append({
            "method": method,
            "ragas_faithfulness": mapped.get("ragas_faithfulness"),
            "ragas_context_relevance": mapped.get("ragas_context_relevance"),
            "ragas_answer_relevance": mapped.get("ragas_answer_relevance"),
            "raw_scores": scores,
        })
        print(f"[done] {method} {mapped}", flush=True)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ragas_version": getattr(ragas, "__version__", ""),
        "table": table,
    }
    (out_dir / "official_ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
