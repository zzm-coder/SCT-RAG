# SCT-RAG: public reproduction package

This repository contains the executable research pipeline for SCT-RAG:

1. parse authorized standards into Markdown;
2. create canonical clause records and clause identities;
3. build the within-standard FAISS index and clause-linked KG cache;
4. run category-conditioned `StdDirect + WithinStd + ClauseKG` retrieval;
5. rerank evidence, generate answers, validate citations, and evaluate outputs.

Paper figures, manuscript sources, internal service addresses, run archives,
audit utilities, and development-only tests are intentionally excluded.

## Released data

| Path | Content | Size |
|---|---|---:|
| `code/3_QA_creation/3_QA_data/indstd_design_qa_clause/train.json` | training split | 500 |
| `code/3_QA_creation/3_QA_data/indstd_design_qa_clause/dev.json` | development split | 100 |
| `code/3_QA_creation/3_QA_data/indstd_design_qa_clause/test.json` | locked test split | 200 |
| `data/hotpot/hotpotqa_50.json` | fixed HotPotQA subset | 50 |

The source standards, extracted Markdown, and derived industrial indexes are
not redistributed. They may be rebuilt from documents that the user is
authorized to process. Do not commit restricted standard text or API keys.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/4_RAG_method/SCT-RAG/requirements.txt
cp code/4_RAG_method/SCT-RAG/.env.example .env
set -a; source .env; set +a
```

All generative and embedding calls use OpenAI-compatible APIs. No private
network address, machine-specific checkpoint path, or credential is embedded.
The optional local Neo4j defaults can be overridden through environment
variables. The paper-aligned MacBERT cross-encoder is trained locally with the
released 500-question training split.

Required environment variables:

```bash
export SCT_API_BASE_URL="https://provider.example/v1"
export SCT_API_KEY="replace-with-your-key"
export SCT_CHAT_MODEL="provider-chat-model"
export SCT_EMBEDDING_API_BASE_URL="https://provider.example/v1"
export SCT_EMBEDDING_API_KEY="replace-with-your-key"
export SCT_EMBEDDING_MODEL="provider-embedding-model"
```

The embedding model used at inference must be identical to the model used to
build the FAISS index.

## Rebuild the evidence assets

### 1. PDF to Markdown

```bash
bash code/0_mineru_pdf/batch_convert.sh \
  /path/to/authorized_pdfs code/0_mineru_pdf/data_GB_md
```

Place the Markdown files for the fixed, authorized 200-standard corpus under
`code/0_mineru_pdf/data_md_filtered`. Corpus membership must be recorded before
indexing and must remain unchanged across compared methods. The release does
not select standards from question text or from a domain-keyword list.

### 2. Extract triples through the chat API

```bash
python code/1_extract_data/extract_triples_api.py \
  --input-dir code/0_mineru_pdf/data_md_filtered \
  --output-dir code/1_extract_data/kg_data
```

### 3. Build canonical clauses and vector index

Use `build_clause_kg_cache.py` to construct the source--clause--triple linkage,
then build the clause index:

```bash
python code/2_kg_construction/clause_ground_truth.py \
  --corpus-root code/0_mineru_pdf/data_md_filtered \
  --qa-dir code/3_QA_creation/3_QA_data/indstd_design_qa_clause \
  --output-dir code/2_kg_construction/clause_records
python code/2_kg_construction/build_clause_kg_cache.py \
  --triples code/1_extract_data/kg_data/triplet_cache.json \
  --clauses code/2_kg_construction/clause_records/clauses.jsonl \
  --output-dir code/2_kg_construction/kg_clause_cache
python code/2_kg_construction/build_kg_entity_index.py \
  --input code/2_kg_construction/kg_clause_cache/triplet_clause_cache.json \
  --output-dir code/2_kg_construction/kg_clause_cache \
  --model "$SCT_EMBEDDING_MODEL"
python code/2_kg_construction/build_clause_vector_db.py \
  --clauses code/2_kg_construction/clause_records/clauses.jsonl \
  --model "$SCT_EMBEDDING_MODEL" \
  --output-dir code/2_kg_construction/clause_vector_db
```

If Neo4j is used, set `SCT_NEO4J_URI`, `SCT_NEO4J_USER`, and
`SCT_NEO4J_PASSWORD`. The retrieval code can also consume the exported caches.

## Run the experiments

```bash
cd code/4_RAG_method/SCT-RAG

# Main SCT-RAG experiment (predicted routing)
python run_benchmark_suite.py \
  --dataset indstd --dataset-class ht --sample-size 200 \
  --methods SCT-RAG --output-subdir main_indstd

# All comparison methods
python run_benchmark_suite.py \
  --dataset indstd --dataset-class ht --sample-size 200 \
  --methods DPR,BM25,Hybrid-RAG,GraphRAG,Adaptive-RAG,Self-RAG,FLARE,CRAG,Closed-book \
  --output-subdir baselines_indstd

# Component ablation
python run_ablation_suite.py \
  --qa-path ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/test.json \
  --sample-size 200

# Development-set sensitivity study
python run_parameter_sensitivity.py \
  --qa ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/dev.json \
  --sample-size 100

# HotPotQA subset
python run_benchmark_suite.py \
  --dataset hotpot --dataset-class hotpot --sample-size 50 \
  --methods SCT-RAG,DPR,BM25,Hybrid-RAG,GraphRAG,Adaptive-RAG,Self-RAG,FLARE,CRAG,Closed-book \
  --output-subdir hotpot_n50
```

Outputs are written to `outputs/`. Each run records its configuration and
per-question retrieval, generation, citation, and evaluation fields.

Before running the Adaptive-RAG comparison, train its classifier on the same
released category labels:

```bash
python compare_rag/Adaptive_rag/train_adaptive_router_text2vec.py \
  --train ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/train.json \
  --dev ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/dev.json \
  --output models/adaptive_router
```

## Official RAGAS 0.1.21

`run_official_ragas.py` consumes completed experiment logs and uses the same
configured chat and embedding APIs:

```bash
python run_official_ragas.py \
  --eval-subdir main_indstd --n 200 --methods SCT-RAG
```

## Train the paper-aligned reranker

The paper protocol uses only the 500-question training split. The 100-question
development split is reserved for parameter selection, and the 200-question
test split is never used for reranker training. The defaults
match the manuscript: MacBERT, two epochs, batch size 16, maximum length 512,
AdamW learning rate `2e-5`, weight decay `0.01`, warm-up ratio `0.1`, pairwise
margin `0.3`, and seed `42`.

The released deterministic pair constructor finds usable supporting clauses in
412 of the 500 training questions. It produces 2,068 sentence-level positive
pairs and samples six negatives per positive, yielding 12,408 negative pairs
and 14,476 question--clause pairs in total. The counts, construction policy,
and training-file hash are recorded in
`code/3_QA_creation/3_QA_data/indstd_design_qa_clause/reranker_pair_statistics.json`.

```bash
python train_reranker.py \
  --train-path ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/train.json \
  --output-dir models/sct_reranker_500q \
  --epochs 2 --batch-size 16 --max-length 512 \
  --seed 42
```

Run `python train_reranker.py --help` for corpus and training arguments. The
test split must not be supplied to reranker training.

The released pair counts can be regenerated without fitting the model:

```bash
python summarize_reranker_pairs.py \
  --train-path ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/train.json \
  --neg-per-pos 6
```

## Paper sensitivity study

The sensitivity runner executes the seven development-set runs reported in the
paper: boost off/default 1.5/2.0, grouping 5/default 10/20, and WithinStd
0.5x/default 1x/2x. The shared default configuration is executed once. Gold
development-set categories are held fixed in this diagnostic so that each run
changes only the selected retrieval or reranking knob.

```bash
python run_parameter_sensitivity.py \
  --qa ../../../code/3_QA_creation/3_QA_data/indstd_design_qa_clause/dev.json \
  --sample-size 100
```

## Output fields

Each benchmark directory contains the run manifest, method log, per-question
JSONL records, and aggregate evaluation. The per-question records preserve the
predicted route, parsed identifiers, retrieved candidates, post-reranking
evidence, generated answer, emitted citations, and clause-level metrics.
Clause Hit@5 is computed from the first five unique canonical clause IDs in the
post-reranking evidence collector. Multiple fragments of the same clause are
merged, and parsed answer citations are not written back to the retrieval list.

## Paper protocol summary

- Router: predicted category for the main comparison; gold category only for
  the explicitly labeled oracle-route ablation and frozen-evidence backbone
  replay.
- Retrieval: `StdDirect + WithinStd + ClauseKG`; no independent full-corpus
  dense or sparse channel is used by SCT-RAG.
- Initial candidate budget: `B=20` for each active retrieval operation.
- Final generator evidence budget: `K_tau=10/10/20` for single-standard,
  cross-standard, and correlation questions.
- Standard-number reranking boost: `beta=1.5`.
- Generation: temperature `0.0`, context safety limit `24,000` characters,
  and maximum output length `1,200` tokens.
- Industrial answer threshold: `0.7`; HotPotQA answer threshold: `0.5`.
- Industrial QA categories: single-standard, cross-standard, and correlation.
- RAGAS scores are produced only by `run_official_ragas.py` with
  `ragas==0.1.21`; they are not approximated inside the main evaluator.

## Prompt reproducibility

The executable router, triple-extraction, generator, and answer-judge prompts
are maintained in their corresponding source modules. When preparing an
archival release, export the exact executed prompt strings and their SHA-256
hashes with the run manifest so that translated manuscript renderings are not
mistaken for the runtime Chinese prompts.
The source-function map is provided in `docs/PROMPTS.md`.
The paper-to-code parameter map is provided in `docs/REPRODUCIBILITY.md`.
The exact public artifact scope is recorded in `RELEASE_MANIFEST.md`.

## Data and licensing

The repository distributes QA annotations and identifiers. Copyrighted source
standards and their full clause text are excluded. Users must supply standards
they are authorized to process and rebuild the 14,440 canonical clause units
with the released scripts. Add the project license and citation metadata before
publishing the repository.

## Reproducibility checks

```bash
python -m compileall -q code
python - <<'PY'
import json
from pathlib import Path
for name, expected in [('train', 500), ('dev', 100), ('test', 200)]:
    path = Path('code/3_QA_creation/3_QA_data/indstd_design_qa_clause') / f'{name}.json'
    assert len(json.loads(path.read_text(encoding='utf-8'))) == expected
print('dataset sizes verified')
PY
```

Dataset hashes are recorded in `data/SHA256SUMS.json`.
