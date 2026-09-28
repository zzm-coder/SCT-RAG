#!/usr/bin/env python3
"""Extract clause triples through an OpenAI-compatible chat API."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_config import DATA_MD_FILTERED, KG_DATA_DIR
from prompt import TRIPLE_EXTRACTION_PROMPT
from text_chunking import split_text_into_chunks


def parse_triples(raw: str, source: str) -> list[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.I).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        records = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    output = []
    for record in records:
        if not isinstance(record, dict):
            continue
        output.append({
            "head": record.get("head", {"name": "", "type": ""}),
            "relation": record.get("relation", ""),
            "tail": record.get("tail", {"name": "", "type": ""}),
            "paragraph": record.get("paragraph", ""),
            "source": source,
            "confidence": record.get("confidence", 0.5),
            "extraction_type": "chat_api",
        })
    return output


class APITripleExtractor:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model

    def extract(self, text: str, source: str) -> list[dict]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.endpoint,
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": TRIPLE_EXTRACTION_PROMPT
                        + "\n\nDocument clause:\n"
                        + text,
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            timeout=300,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return parse_triples(raw, source)


def collect_files(input_dir: Path, output_dir: Path):
    for source in sorted(input_dir.rglob("*.md")):
        yield source, output_dir / source.relative_to(input_dir).with_suffix(".json")


def write_aggregate_cache(output_dir: Path) -> Path:
    """Combine per-document extraction files into the cache consumed by linkage."""
    cache: dict[str, list[dict]] = {}
    for path in sorted(output_dir.rglob("*.json")):
        if path.name == "triplet_cache.json":
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for record in records:
            source = str(record.get("source") or path.with_suffix(".md").name)
            cache.setdefault(source, []).append(record)
    destination = output_dir / "triplet_cache.json"
    destination.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DATA_MD_FILTERED)
    parser.add_argument("--output-dir", type=Path, default=KG_DATA_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=1500)
    args = parser.parse_args()

    base_url = os.environ.get("SCT_API_BASE_URL", "")
    model = os.environ.get("SCT_CHAT_MODEL", "")
    if not base_url or not model:
        raise SystemExit("Set SCT_API_BASE_URL and SCT_CHAT_MODEL")
    extractor = APITripleExtractor(base_url, os.environ.get("SCT_API_KEY", ""), model)
    files = list(collect_files(args.input_dir, args.output_dir))[args.start:]
    if args.limit:
        files = files[:args.limit]
    for source, destination in tqdm(files, desc="triple extraction"):
        triples = []
        text = source.read_text(encoding="utf-8", errors="ignore")
        for chunk in split_text_into_chunks(text, args.max_chars, 100):
            triples.extend(extractor.extract(chunk, source.name))
        unique = {json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in triples}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(list(unique.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate = write_aggregate_cache(args.output_dir)
    print(f"Wrote aggregate triple cache: {aggregate}")


if __name__ == "__main__":
    main()
