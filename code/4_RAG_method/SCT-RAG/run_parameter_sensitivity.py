#!/usr/bin/env python3
"""Paper-aligned one-factor-at-a-time sensitivity study on dev n=100.

The seven runs reproduce the manuscript's three knobs: standard-number boost
(off/default 1.5/2.0), grouping budget (5/default 10/20), and WithinStd
multiplier (0.5/default 1/2). The default run is shared by all three panels.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PAPER_VARIANTS = (
    ("default", {}),
    ("boost_off", {"SCT_RERANK_BETA_OVERRIDE": "1.0"}),
    ("boost_2.0", {"SCT_RERANK_BETA_OVERRIDE": "2.0"}),
    ("grouping_5", {"SCT_ASSEMBLE_TOP_K": "5"}),
    ("grouping_20", {"SCT_ASSEMBLE_TOP_K": "20"}),
    ("withinstd_0.5x", {"SCT_WITHIN_BUDGET_SCALE": "0.5"}),
    ("withinstd_2x", {"SCT_WITHIN_BUDGET_SCALE": "2.0"}),
)

CONTROLLED_ENV = {
    "SCT_RERANK_BETA_OVERRIDE",
    "SCT_ASSEMBLE_TOP_K",
    "SCT_WITHIN_BUDGET_SCALE",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--base-url", default=os.environ.get("SCT_API_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("SCT_CHAT_MODEL", ""))
    parser.add_argument("--output-prefix", default="sensitivity_paper_dev")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.base_url or not args.model:
        parser.error("set SCT_API_BASE_URL and SCT_CHAT_MODEL, or pass --base-url/--model")
    if args.sample_size != 100:
        parser.error("the paper sensitivity protocol requires --sample-size 100")

    records: list[dict] = []
    for index, (name, overrides) in enumerate(PAPER_VARIANTS, 1):
        output_subdir = f"{args.output_prefix}_{name}_n100"
        command = [
            sys.executable,
            "run_benchmark_suite.py",
            "--qa-path", str(args.qa),
            "--sample-size", "100",
            "--sample-mode", "head",
            "--methods", "SCT-RAG-oracle",
            "--output-subdir", output_subdir,
            "--dataset-class", "ht",
            "--llm-service-url", args.base_url,
            "--llm-model", args.model,
            "--llm-seed", "42",
            "--allow-unvalidated-dataset",
        ]
        reranker = os.environ.get("SCT_RERANKER_MODEL", "").strip()
        if reranker:
            command += ["--cross-encoder-model", reranker]
        if args.resume:
            command.append("--skip-existing")

        environment = os.environ.copy()
        for key in CONTROLLED_ENV:
            environment.pop(key, None)
        environment.update(overrides)

        print(f"[{index}/{len(PAPER_VARIANTS)}] {name}: {overrides or 'paper defaults'}", flush=True)
        started_at = datetime.now(timezone.utc).isoformat()
        process = subprocess.run(command, env=environment, check=False)
        records.append(
            {
                "name": name,
                "environment": overrides,
                "output_subdir": output_subdir,
                "returncode": process.returncode,
                "started_at": started_at,
            }
        )
        Path("sensitivity_progress.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
