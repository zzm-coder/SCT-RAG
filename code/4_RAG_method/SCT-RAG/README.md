# SCT-RAG implementation

- `query_router.py`: three-category LLM routing and standard-ID parsing.
- `standard_boost.py`: StdDirect, WithinStd, fixed beta=1.5, and ID fusion.
- `kg_retriever.py`: clause-linked graph traversal and clause mapping.
- `vector_retriever.py`: API embeddings, FAISS retrieval, and reranking.
- `sct_rag_system.py`: category-conditioned orchestration.
- `llm_generator.py`: API generation and citation validation.
- `evaluator.py`, `clause_metrics.py`, `accuracy_protocol.py`: the final
  clause-level, citation, answer, and latency metrics used in the paper.
- `run_benchmark_suite.py`: SCT-RAG and baseline experiments.
- `run_ablation_suite.py`, `run_parameter_sensitivity.py`: controlled studies.
- `train_reranker.py`: MacBERT cross-encoder training on the 500-question
  training split with BCE relevance loss and a margin-based pairwise term.
- `run_official_ragas.py`: the only RAGAS implementation in the release,
  using official `ragas==0.1.21` metrics.

See the repository root `README.md` for setup and commands.

The paper configuration uses an initial per-channel candidate budget of 20,
final evidence budgets of 10/10/20 for single-standard/cross-standard/
correlation queries, and a standard-number boost of 1.5. The main comparison
uses predicted routing; gold categories are restricted to the explicitly
labeled oracle-route ablation and backbone replay.

The released industrial dataset contains the three open-answer categories used
by the final paper: single-standard, cross-standard, and correlation.
Industrial Acc is the frozen answer-judge pass rate at 0.7;
official RAGAS metrics are computed afterward with `run_official_ragas.py`.

The executable SCT-RAG order is: explicit standard-ID parsing; StdDirect,
WithinStd, and clause-linked KG retrieval; `MergeByID` over canonical clause
identities; cross-encoder scoring; the fixed standard-number multiplier
`beta=1.5`; and category-conditioned `SelectTop` with final budgets 10/10/20.
