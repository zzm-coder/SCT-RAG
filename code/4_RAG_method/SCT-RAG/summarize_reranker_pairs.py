#!/usr/bin/env python3
"""Recompute the paper's reranker pair inventory without training a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_reranker import count_training_questions, file_sha256, load_positive_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--neg-per-pos", type=int, default=6)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    positives = load_positive_pairs(args.train_path, sentence_level=True)
    total_questions, usable_questions = count_training_questions(args.train_path)
    positive_count = len(positives)
    negative_count = positive_count * args.neg_per_pos
    result = {
        "schema_version": "sct-reranker-pair-statistics-v1",
        "train_file": Path(args.train_path).name,
        "train_sha256": file_sha256(args.train_path),
        "training_questions": total_questions,
        "questions_with_usable_support": usable_questions,
        "questions_without_usable_support": total_questions - usable_questions,
        "sentence_level_positive_pairs": positive_count,
        "negative_pairs_per_positive": args.neg_per_pos,
        "negative_pairs": negative_count,
        "total_question_clause_pairs": positive_count + negative_count,
        "sentence_level": True,
        "negative_sampling": "adjacent_plus_same_standard_dense_plus_random_cross_standard",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
