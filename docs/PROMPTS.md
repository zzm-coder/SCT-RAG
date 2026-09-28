# Runtime prompt map

The manuscript presents English renderings for readability. The locked
experiments execute the Chinese prompt strings defined in the source files
listed below.

| Prompt | Runtime source |
|---|---|
| Three-category standard-aware router | `code/4_RAG_method/SCT-RAG/query_router.py::_build_analysis_prompt` |
| Triple extraction | `code/1_extract_data/prompt.py` and `extract_triples_api.py` |
| Identity-preserving answer generation | `code/4_RAG_method/SCT-RAG/llm_generator.py::_build_ht_prompt` |
| Category-conditioned answer contract | `code/indstd_answer_style.py` |
| Answer Judge v2 | `code/4_RAG_method/SCT-RAG/llm_generator.py::build_answer_judge_prompt` |

For an archival experiment, preserve the source revision, run manifest, model
identifier, decoding settings, and SHA-256 hashes of these files. The router
predicts the category and key entities. The deterministic parser independently
extracts and normalizes explicit standard identifiers used by retrieval.

The generator receives only the selected evidence context and may emit only
the clause identifiers exposed in that context. Citation validation intersects
the emitted identifiers with the identifiers in the selected evidence set.

Answer Judge v2 uses four non-overlapping score bands: 0.85--1.00 for a fully
correct response, 0.70--0.84 for a correct decisive conclusion with only minor
omissions, 0.40--0.69 for a substantive but incomplete response, and
0.00--0.39 for a response without sufficient decisive content or with a
contradictory decisive value. Industrial Acc applies the fixed 0.70 threshold.
