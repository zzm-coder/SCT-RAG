# Code layout

| Directory | Released purpose |
|---|---|
| `0_mineru_pdf` | PDF-to-Markdown conversion for a fixed corpus |
| `1_extract_data` | API-based triple extraction and prompts |
| `2_kg_construction` | clause linkage, FAISS, KG cache, and HotPot assets |
| `3_QA_creation` | dataset validation, deterministic splitting, and released industrial QA data |
| `4_RAG_method/SCT-RAG` | method, baselines, training, experiments, and metrics |

`project_config.py` contains relocatable paths and environment-backed service
settings only. Complete commands are in the repository root README.
