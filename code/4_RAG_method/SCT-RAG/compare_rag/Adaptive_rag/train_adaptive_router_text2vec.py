#!/usr/bin/env python3
"""Train the Adaptive-RAG baseline router on the released category labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SCT_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(SCT_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(SCT_RAG_ROOT))

from api_embedding import APIEmbeddingModel


CATEGORIES = ["single", "cross", "correlation"]


def load_split(path: Path) -> tuple[list[str], list[int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    label_to_id = {label: index for index, label in enumerate(CATEGORIES)}
    questions: list[str] = []
    labels: list[int] = []
    for item in records:
        question = str(item.get("question") or "").strip()
        label = str(item.get("type") or item.get("question_type") or "").strip().lower()
        if question and label in label_to_id:
            questions.append(question)
            labels.append(label_to_id[label])
    if not questions:
        raise ValueError(f"No labelled questions found in {path}")
    return questions, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/adaptive_router"))
    args = parser.parse_args()

    model_name = os.environ.get("SCT_EMBEDDING_MODEL", "").strip()
    if not model_name:
        parser.error("SCT_EMBEDDING_MODEL is required")
    train_questions, train_labels = load_split(args.train)
    dev_questions, dev_labels = load_split(args.dev)
    encoder = APIEmbeddingModel(model_name)
    train_embeddings = encoder.encode(train_questions, normalize_embeddings=True)
    dev_embeddings = encoder.encode(dev_questions, normalize_embeddings=True)

    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(train_embeddings, train_labels)
    dev_accuracy = accuracy_score(dev_labels, classifier.predict(dev_embeddings))

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.output / "classifier.joblib")
    (args.output / "labels.json").write_text(
        json.dumps({"labels": CATEGORIES}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "training_metadata.json").write_text(
        json.dumps(
            {
                "train_size": len(train_questions),
                "dev_size": len(dev_questions),
                "dev_accuracy": float(dev_accuracy),
                "seed": 42,
                "embedding_model": model_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved router to {args.output}; dev accuracy={dev_accuracy:.4f}")


if __name__ == "__main__":
    main()
