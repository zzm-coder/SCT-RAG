# Dataset construction

`build_dataset_splits.py` validates clause-grounded QA records and creates the
paper splits: 500 training, 100 development, and 200 test questions. Every
record must use one of `single`, `cross`, or `correlation` and must contain at
least one canonical `evidence_id`. Duplicate keys are rejected before the
seeded split.

```bash
python code/3_QA_creation/build_dataset_splits.py \
  --input /path/to/curated_800.json \
  --output-dir code/3_QA_creation/3_QA_data/indstd_design_qa_clause \
  --seed 42
```

The released split files are stored under `3_QA_data/`. Source-standard text
is not redistributed; the records retain canonical clause identifiers and the
supporting excerpts required for evaluation and reranker training.
