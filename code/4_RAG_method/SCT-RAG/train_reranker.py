#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a clause-level CrossEncoder reranker for industrial standard QA.

Positive pairs come from supporting clauses in the 500-question training
split. Negative pairs are sampled from the training corpus, prioritizing
same-standard dense candidates before random cross-standard candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

try:
    import sys

    CODE_ROOT = Path(__file__).resolve().parents[2]
    SCT_RAG_ROOT = Path(__file__).resolve().parent
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    if str(SCT_RAG_ROOT) not in sys.path:
        sys.path.insert(0, str(SCT_RAG_ROOT))
    from project_config import CROSS_ENCODER_MODEL, KG_VECTOR_DB_DIR, DATA_MD_FILTERED
except Exception:
    CROSS_ENCODER_MODEL = os.environ.get("SCT_RERANKER_BASE_MODEL", "hfl/chinese-macbert-base")
    KG_VECTOR_DB_DIR = Path(__file__).resolve().parents[2] / "2_kg_construction" / "kg_vector_db_filtered"
    DATA_MD_FILTERED = Path(__file__).resolve().parents[2] / "0_mineru_pdf" / "data_md_filtered"

# Keep training and inference at the same sentence granularity.
try:
    from standard_boost import split_evidence_sentences
except Exception:
    def split_evidence_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[。；;！？\n])", text or "")
        out = []
        for p in parts:
            s = (p or "").strip()
            compact = re.sub(r"\s+", "", s)
            if 12 <= len(compact) <= 420:
                out.append(s)
        return out or ([text.strip()] if text and len(re.sub(r"\s+", "", text)) >= 12 else [])



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_reranker")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_integrity_manifest(manifest_path: str, train_path: str, corpus_chunks: str) -> dict:
    """Fail closed unless the declared reranker inputs have zero test overlap."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = manifest.get("verification") or {}
    required = (
        "test_question_overlap",
        "test_evidence_id_overlap",
        "test_evidence_text_overlap",
        "within_train_dev_question_duplicates",
        "within_train_dev_id_duplicates",
    )
    failed = {key: checks.get(key) for key in required if checks.get(key) != 0}
    if failed:
        raise ValueError(f"reranker split-integrity verification failed: {failed}")
    outputs = manifest.get("outputs") or {}
    declared_qa = Path(outputs.get("qa", "")).resolve()
    declared_chunks = Path(outputs.get("chunks", "")).resolve()
    if declared_qa != Path(train_path).resolve() or declared_chunks != Path(corpus_chunks).resolve():
        raise ValueError("training inputs do not match the integrity manifest outputs")
    if file_sha256(train_path) != outputs.get("qa_sha256"):
        raise ValueError("training QA SHA-256 differs from the integrity manifest")
    if file_sha256(corpus_chunks) != outputs.get("chunks_sha256"):
        raise ValueError("training corpus SHA-256 differs from the integrity manifest")
    return manifest


@dataclass
class CorpusClause:
    idx: int
    source: str
    text: str


def normalize_source(src: str) -> str:
    src = (src or "").replace("\\", "/").split("/")[-1]
    return src.strip()


def iter_json_items(path: Path) -> Iterable[dict]:
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            yield from iter_json_items(file)
        return
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        # Accept both a single item and {id: item} maps.
        if "question" in data or "query" in data:
            yield data
        else:
            for v in data.values():
                if isinstance(v, dict):
                    yield v


def get_supporting_facts(item: dict) -> list[dict]:
    sfs = list(item.get("supporting_facts") or item.get("support_facts") or [])
    # Extended facts are optional supervision. They are useful positives for
    # clause-level training, while the evaluator still uses only supporting_facts
    # as the required gold evidence.
    sfs.extend(item.get("extended_supporting_facts") or [])
    out = []
    seen = set()
    for sf in sfs:
        if isinstance(sf, str):
            key = ("", sf)
            if key not in seen:
                seen.add(key)
                out.append({"source": "", "chunk": sf})
        elif isinstance(sf, dict):
            text = (
                sf.get("chunk")
                or sf.get("paragraph")
                or sf.get("text")
                or sf.get("context")
                or sf.get("para")
                or sf.get("chunk_text")
                or sf.get("paragraph_text")
                or ""
            )
            if text:
                source = normalize_source(sf.get("source") or sf.get("doc") or "")
                key = (source, text)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"source": source, "chunk": text})
    return out


def attach_side_query(question: str, source: str) -> str:
    """训练查询带侧维度，与推理 ce_query_for_chunk 对齐。"""
    try:
        from standard_boost import ce_query_for_chunk
        return ce_query_for_chunk(question, source)
    except Exception:
        return question


def expand_to_sentence_positives(
    pairs: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """将 SF 整段切成与推理一致的短证据句正例。"""
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for q, text, src in pairs:
        sents = split_evidence_sentences(text)
        if not sents:
            compact = re.sub(r"\s+", "", text or "")
            if len(compact) >= 12:
                sents = [text.strip()]
        for sent in sents:
            key = (q, sent, src)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def load_positive_pairs(
    train_path: str,
    max_pairs: int | None = None,
    sentence_level: bool = True,
) -> list[tuple[str, str, str]]:
    pairs = []
    for item in iter_json_items(Path(train_path)):
        q = item.get("question") or item.get("query") or ""
        if not q:
            continue
        for sf in get_supporting_facts(item):
            text = sf["chunk"].strip()
            if len(text) < 8:
                continue
            src = sf["source"]
            pairs.append((attach_side_query(q, src), text, src))
            if max_pairs and len(pairs) >= max_pairs:
                break
        if max_pairs and len(pairs) >= max_pairs:
            break
    if sentence_level:
        pairs = expand_to_sentence_positives(pairs)
        if max_pairs:
            pairs = pairs[:max_pairs]
    return pairs


def count_training_questions(train_path: str) -> tuple[int, int]:
    """Return total questions and questions that yield usable support pairs."""
    total = 0
    usable = 0
    for item in iter_json_items(Path(train_path)):
        total += 1
        facts = get_supporting_facts(item)
        if any(len((fact.get("chunk") or "").strip()) >= 8 for fact in facts):
            usable += 1
    return total, usable


def expand_corpus_to_sentences(corpus: list[CorpusClause]) -> list[CorpusClause]:
    """语料切句，供同标准难负例挖掘（与推理句级一致）。"""
    out: list[CorpusClause] = []
    idx = 0
    seen: set[tuple[str, str]] = set()
    for c in corpus:
        sents = split_evidence_sentences(c.text)
        if not sents:
            sents = [c.text]
        for sent in sents:
            compact = re.sub(r"\s+", "", sent)
            key = (c.source, compact[:240])
            if key in seen or len(compact) < 12:
                continue
            seen.add(key)
            out.append(CorpusClause(idx=idx, source=c.source, text=sent))
            idx += 1
    return out


def load_corpus(corpus_chunks: str) -> list[CorpusClause]:
    path = Path(corpus_chunks)
    if path.is_dir():
        path = path / "chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    clauses = []
    iterable = enumerate(data) if isinstance(data, list) else ((int(k), v) for k, v in data.items())
    for idx, chunk in iterable:
        meta = chunk.get("metadata", {}) or {}
        text = chunk.get("context_text") or chunk.get("original_text") or chunk.get("chunk_text") or chunk.get("text") or ""
        source = normalize_source(meta.get("file_name") or meta.get("source") or chunk.get("source") or "")
        if len(text.strip()) >= 8:
            clauses.append(CorpusClause(idx=int(idx), source=source, text=text.strip()))
    return clauses


def split_markdown_clauses(text: str) -> list[str]:
    text = re.sub(r"!\[.*?\]\([^)]*\)", " ", text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"#+", " ", text)
    raw_parts = re.split(r"(?<=[。；;])|(?=^#{2,5}\s*)|\n+", text, flags=re.M)
    clauses: list[str] = []
    for part in raw_parts:
        part = re.sub(r"\s+", " ", part).strip(" ；;，,。")
        if not part:
            continue
        if len(part) > 620:
            buf = ""
            for sub in re.split(r"(?<=[，,、])", part):
                if len(buf) + len(sub) <= 520:
                    buf += sub
                else:
                    if 18 <= len(buf.strip()) <= 620:
                        clauses.append(buf.strip(" ；;，,。"))
                    buf = sub
            if 18 <= len(buf.strip()) <= 620:
                clauses.append(buf.strip(" ；;，,。"))
        elif 18 <= len(part) <= 620:
            clauses.append(part)
    return clauses


def load_md_clause_corpus(md_root: str) -> list[CorpusClause]:
    root = Path(md_root)
    if not root.exists():
        return []
    clauses: list[CorpusClause] = []
    idx = 10_000_000
    for path in sorted(root.glob("*/auto/*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        source = normalize_source(path.name)
        for clause in split_markdown_clauses(text):
            compact = re.sub(r"\s+", "", clause)
            if len(compact) < 18:
                continue
            if re.fullmatch(r"(?:HB|GJB|GB/T|GB|QJ|SJ)[A-Z0-9＋+\-—－./（）() ]{2,}", compact, flags=re.I):
                continue
            if any(k in compact for k in ("目录", "前言", "出版", "发行", "书号", "定价")) and len(compact) < 420:
                continue
            clauses.append(CorpusClause(idx=idx, source=source, text=clause))
            idx += 1
    return clauses


def _filter_same_source_candidates(
    same_source: list[CorpusClause],
    pos: str,
) -> list[CorpusClause]:
    pos_compact = re.sub(r"\s+", "", pos)
    out = []
    for c in same_source:
        if c.text == pos:
            continue
        cc = re.sub(r"\s+", "", c.text)
        if pos_compact in cc or cc in pos_compact:
            continue
        out.append(c)
    return out


def _precompute_embeddings(
    texts: list[str],
    embedder: SentenceTransformer,
    batch_size: int = 256,
) -> torch.Tensor:
    """批量预编码，避免难负例阶段逐条 encode。"""
    with torch.no_grad():
        emb = embedder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    return emb


def build_examples(
    positives: list[tuple[str, str, str]],
    corpus: list[CorpusClause],
    neg_per_pos: int,
    seed: int,
    embedder: SentenceTransformer | None = None,
    same_src_neg_ratio: float = 0.85,
) -> list[InputExample]:
    """Build positive pairs and corpus-derived negative pairs."""
    rng = random.Random(seed)
    by_source: dict[str, list[CorpusClause]] = {}
    for c in corpus:
        by_source.setdefault(c.source, []).append(c)

    # 预编码语料与问题，显著加速 hard-neg
    corpus_emb = None
    q_emb_map: dict[str, torch.Tensor] = {}
    idx_to_row: dict[int, int] = {}
    if embedder is not None and corpus:
        logger.info("pre-encoding corpus sentences for dense hard-neg: %d", len(corpus))
        corpus_emb = _precompute_embeddings([c.text for c in corpus], embedder)
        idx_to_row = {c.idx: i for i, c in enumerate(corpus)}
        uniq_q = sorted({q for q, _, _ in positives})
        logger.info("pre-encoding unique questions: %d", len(uniq_q))
        q_mat = _precompute_embeddings(uniq_q, embedder)
        for i, q in enumerate(uniq_q):
            q_emb_map[q] = q_mat[i]

    examples: list[InputExample] = []
    all_clauses = corpus[:]
    n_same = max(1, int(round(neg_per_pos * same_src_neg_ratio)))
    total = len(positives)
    for pi, (q, pos, src) in enumerate(positives, start=1):
        if pi == 1 or pi % 200 == 0 or pi == total:
            logger.info("hard-neg progress %d/%d", pi, total)
        examples.append(InputExample(texts=[q, pos], label=1.0))

        negs: list[CorpusClause] = []
        same_source_full = _filter_same_source_candidates(by_source.get(src, []), pos)
        # Add adjacent clauses before dense same-standard candidates.
        if same_source_full:
            pos_idx = next((i for i, c in enumerate(same_source_full) if c.text == pos), None)
            if pos_idx is None:
                pos_compact = re.sub(r"\s+", "", pos)
                pos_idx = next(
                    (i for i, c in enumerate(same_source_full) if pos_compact[:40] in re.sub(r"\s+", "", c.text)),
                    None,
                )
            if pos_idx is not None:
                for j in (pos_idx - 1, pos_idx + 1, pos_idx - 2, pos_idx + 2):
                    if 0 <= j < len(same_source_full) and same_source_full[j] not in negs:
                        negs.append(same_source_full[j])
                    if len(negs) >= n_same:
                        break
        same_source = same_source_full
        if len(same_source) > 80:
            same_source = rng.sample(same_source, 80)

        if embedder is not None and corpus_emb is not None and same_source and q in q_emb_map:
            rows = [idx_to_row[c.idx] for c in same_source if c.idx in idx_to_row]
            cands = [c for c in same_source if c.idx in idx_to_row]
            if rows:
                with torch.no_grad():
                    qe = q_emb_map[q].unsqueeze(0)
                    ce = corpus_emb[rows]
                    scores = torch.mm(qe, ce.transpose(0, 1)).squeeze(0).detach().cpu().tolist()
                    if isinstance(scores, float):
                        scores = [scores]
                ranked = sorted(zip(cands, scores), key=lambda x: float(x[1]), reverse=True)
                negs.extend([c for c, _ in ranked[:n_same]])
        else:
            negs.extend(same_source[:n_same])

        # Fill remaining slots with corpus-level random negatives.
        tries = 0
        while len(negs) < neg_per_pos and all_clauses and tries < neg_per_pos * 40:
            tries += 1
            c = rng.choice(all_clauses)
            if src and c.source == src and len(negs) < n_same + 1:
                # 同标准位优先留给 dense；偶发允许
                if rng.random() < 0.7:
                    continue
            if c.text == pos or c in negs:
                continue
            negs.append(c)

        for n in negs[:neg_per_pos]:
            examples.append(InputExample(texts=[q, n.text], label=0.0))
    rng.shuffle(examples)
    return examples


def train(
    train_path: str,
    model_name: str,
    output_dir: str,
    corpus_chunks: str,
    epochs: int = 2,
    batch_size: int = 16,
    max_pairs: int | None = None,
    neg_per_pos: int = 6,
    seed: int = 42,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    log_every: int = 50,
    md_root: str | None = None,
    include_md_clauses: bool = True,
    sentence_level: bool = True,
    embed_model: str | None = None,
    expand_corpus_sentences: bool = True,
    integrity_manifest: str | None = None,
):
    integrity = None
    if integrity_manifest:
        integrity = verify_integrity_manifest(integrity_manifest, train_path, corpus_chunks)
        logger.info("verified zero-overlap integrity manifest: %s", integrity_manifest)
    positives = load_positive_pairs(
        train_path, max_pairs=max_pairs, sentence_level=sentence_level
    )
    if not positives:
        raise ValueError(f"No positive clause pairs found in {train_path}")
    corpus = load_corpus(corpus_chunks)
    if include_md_clauses and md_root:
        md_clauses = load_md_clause_corpus(md_root)
        if md_clauses:
            logger.info("markdown clause corpus: %d", len(md_clauses))
            existing = {(c.source, re.sub(r"\s+", "", c.text)[:180]) for c in corpus}
            for c in md_clauses:
                key = (c.source, re.sub(r"\s+", "", c.text)[:180])
                if key not in existing:
                    corpus.append(c)
                    existing.add(key)
    if not corpus:
        raise ValueError(f"No corpus clauses found in {corpus_chunks}")

    if expand_corpus_sentences and sentence_level:
        before = len(corpus)
        corpus = expand_corpus_to_sentences(corpus)
        logger.info("corpus sentence expand: %d -> %d", before, len(corpus))

    logger.info("positive pairs: %d (sentence_level=%s)", len(positives), sentence_level)
    logger.info("corpus clauses: %d", len(corpus))

    # dense 难负例：整问 embedding，不用关键词重叠
    emb_name = embed_model or os.environ.get("SCT_RERANKER_MINING_MODEL", "hfl/chinese-macbert-base")
    embedder = None
    try:
        logger.info("loading dense embedder for hard-neg: %s", emb_name)
        embedder = SentenceTransformer(emb_name)
    except Exception as e:
        logger.warning("dense embedder load failed (%s), fallback to random same-src negs", e)

    examples = build_examples(
        positives, corpus, neg_per_pos=neg_per_pos, seed=seed, embedder=embedder
    )
    positive_example_count = sum(float(example.label) >= 0.5 for example in examples)
    negative_example_count = len(examples) - positive_example_count
    question_count, usable_question_count = count_training_questions(train_path)
    logger.info("training examples: %d", len(examples))
    logger.info(
        "pair inventory: questions=%d usable_questions=%d positive=%d negative=%d",
        question_count,
        usable_question_count,
        positive_example_count,
        negative_example_count,
    )
    # 释放 embedder 显存，留给 CrossEncoder 训练
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = CrossEncoder(model_name, max_length=max_length)
    train_cross_encoder_torch(
        model=model,
        examples=examples,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        seed=seed,
        log_every=log_every,
        pairwise=True,
        pairwise_margin=0.3,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    metadata = {
        "train_path": str(train_path),
        "base_model": str(model_name),
        "corpus_chunks": str(corpus_chunks),
        "training_questions": question_count,
        "questions_with_usable_support": usable_question_count,
        "positive_pairs": len(positives),
        "positive_training_examples": positive_example_count,
        "negative_training_examples": negative_example_count,
        "training_examples": len(examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "neg_per_pos": neg_per_pos,
        "seed": seed,
        "max_length": max_length,
        "log_every": log_every,
        "md_root": str(md_root or ""),
        "include_md_clauses": include_md_clauses,
        "sentence_level": sentence_level,
        "expand_corpus_sentences": expand_corpus_sentences,
        "embed_model": emb_name,
        "negative_sampling": "adjacent_plus_same_standard_dense_plus_random_cross_standard",
        "pairwise": True,
        "integrity_manifest": str(integrity_manifest or ""),
        "integrity_manifest_sha256": file_sha256(integrity_manifest) if integrity_manifest else "",
        "train_sha256": file_sha256(train_path),
        "corpus_sha256": file_sha256(corpus_chunks),
        "integrity_verification": (integrity or {}).get("verification", {}),
    }
    (out / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("saved reranker to %s", out)


def train_cross_encoder_torch(
    model: CrossEncoder,
    examples: list[InputExample],
    epochs: int,
    batch_size: int,
    max_length: int,
    warmup_ratio: float,
    seed: int,
    log_every: int,
    pairwise: bool = False,
    pairwise_margin: float = 0.3,
):
    """Small dependency-free CrossEncoder training loop.

    Newer sentence-transformers CrossEncoder.fit depends on the optional
    `datasets` package. This loop keeps training local to torch/transformers.
    """
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("training device: %s", device)
    model.model.to(device)
    model.model.train()

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = max(1, (len(examples) + batch_size - 1) // batch_size * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        return 1.0

    step = 0
    num_labels = getattr(model.model.config, "num_labels", 1)
    epoch_steps = max(1, (len(examples) + batch_size - 1) // batch_size)
    for epoch in range(epochs):
        epoch_start = time.time()
        rng.shuffle(examples)
        losses = []
        for batch_idx, start in enumerate(range(0, len(examples), batch_size), start=1):
            batch = examples[start:start + batch_size]
            texts_a = [ex.texts[0] for ex in batch]
            texts_b = [ex.texts[1] for ex in batch]
            labels = torch.tensor([float(ex.label) for ex in batch], dtype=torch.float32, device=device)
            features = model.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            features = {k: v.to(device) for k, v in features.items()}
            outputs = model.model(**features)
            logits = outputs.logits
            if num_labels == 1:
                loss = BCEWithLogitsLoss()(logits.view(-1), labels)
                if pairwise:
                    pos_logit = logits.view(-1)[labels > 0.5]
                    neg_logit = logits.view(-1)[labels <= 0.5]
                    if pos_logit.numel() and neg_logit.numel():
                        # 边际：正例分应高于负例至少 margin
                        gap = pos_logit.mean() - neg_logit.mean()
                        loss = loss + torch.relu(torch.tensor(pairwise_margin, device=device) - gap)
            else:
                class_labels = labels.long()
                loss = CrossEntropyLoss()(logits, class_labels)
            loss.backward()
            scale = lr_scale(step)
            for group in optimizer.param_groups:
                group["lr"] = 2e-5 * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu()))
            if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == epoch_steps):
                recent = losses[-log_every:]
                avg_loss = sum(recent) / max(1, len(recent))
                logger.info(
                    "epoch %d/%d step %d/%d total_step %d/%d loss=%.4f lr=%.2e elapsed=%.1fs",
                    epoch + 1,
                    epochs,
                    batch_idx,
                    epoch_steps,
                    step,
                    total_steps,
                    avg_loss,
                    optimizer.param_groups[0]["lr"],
                    time.time() - epoch_start,
                )
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, sum(losses) / max(1, len(losses)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "3_QA_creation"
            / "3_QA_data"
            / "indstd_design_qa_clause"
            / "train.json"
        ),
        help="500-question training split; do not pass the development or test split",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("SCT_RERANKER_BASE_MODEL", "hfl/chinese-macbert-base"),
        help="MacBERT base checkpoint used to initialize the cross-encoder",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models" / "sct_reranker_500q"))
    parser.add_argument("--corpus-chunks", default=str(Path(KG_VECTOR_DB_DIR) / "chunks.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--neg-per-pos", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--md-root", default=str(DATA_MD_FILTERED))
    parser.add_argument("--no-md-clauses", action="store_true")
    parser.add_argument(
        "--no-sentence-level",
        action="store_true",
        help="关闭句级正例/语料切句（回退整段 SF）",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("SCT_RERANKER_MINING_MODEL", "hfl/chinese-macbert-base"),
        help="同标准 dense 难负例用的 bi-encoder",
    )
    parser.add_argument(
        "--integrity-manifest",
        default="",
        help="zero-overlap manifest; when provided, hashes and all overlap checks must pass",
    )
    args = parser.parse_args()
    train(
        train_path=args.train_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        corpus_chunks=args.corpus_chunks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        log_every=args.log_every,
        md_root=args.md_root,
        include_md_clauses=not args.no_md_clauses,
        sentence_level=not args.no_sentence_level,
        embed_model=args.embed_model,
        expand_corpus_sentences=not args.no_sentence_level,
        integrity_manifest=args.integrity_manifest or None,
    )


if __name__ == "__main__":
    main()
