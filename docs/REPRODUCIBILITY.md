# Paper-to-code configuration map

| Manuscript item | Public implementation |
|---|---|
| Predicted-routing main run | `run_benchmark_suite.py --methods SCT-RAG` |
| Oracle-route ceiling | `run_benchmark_suite.py --methods SCT-RAG-oracle` |
| Component ablation | `run_ablation_suite.py` |
| Initial per-channel budget `B=20` | `SystemConfig.top_k_vector`, `SystemConfig.top_k_kg`, and StdDirect limit |
| Final evidence `K_tau=10/10/20` | `assemble_context_by_paper_type` |
| Standard boost `beta=1.5` | `standard_boost.py::STANDARD_RERANK_BETA` |
| Category-conditioned assembly | `standard_boost.py::assemble_context_by_paper_type` |
| WithinStd sensitivity | `SCT_WITHIN_BUDGET_SCALE` |
| Grouping sensitivity | `SCT_ASSEMBLE_TOP_K` |
| Official RAGAS | `run_official_ragas.py`, `ragas==0.1.21` |
| MacBERT reranker | `train_reranker.py` |
| Category-conditioned generation contract | `code/indstd_answer_style.py` |
| Answer Judge v2 and 0.70 threshold | `llm_generator.py`, `accuracy_protocol.py` |

The paper reranker uses `train.json` (500 questions), two epochs, batch size
16, maximum length 512, AdamW learning rate `2e-5`, weight decay `0.01`,
warm-up ratio `0.1`, pairwise margin `0.3`, and seed `42`. Development data are
used for parameter selection, and test data are reserved for final evaluation.
Of the 500 training questions, 412 contain usable supporting clauses under the
released pair constructor. Clause-to-sentence expansion produces 2,068 positive
pairs. Six negatives per positive produce 12,408 negative pairs and 14,476
question--clause pairs in total. The auditable inventory is stored in
`code/3_QA_creation/3_QA_data/indstd_design_qa_clause/reranker_pair_statistics.json`; every new
training run also writes the realized label counts to `training_metadata.json`.

The main comparison uses predicted routing. Gold categories are used only in
the explicitly labeled oracle-route ablation, the one-factor sensitivity
diagnostic, and frozen-evidence backbone replay.
