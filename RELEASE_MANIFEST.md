# Public release manifest

## Included

- PDF-to-Markdown batch entry and manifest-defined corpus preparation.
- API-based triple extraction prompt and parser.
- Canonical clause/triple linkage and FAISS/KG asset builders.
- SCT-RAG routing, StdDirect, WithinStd, ClauseKG, identity-preserving fusion,
  MacBERT reranking, generation, and citation validation.
- Executable category-conditioned generation and Answer Judge v2 prompts.
- DPR, GraphRAG, BM25, Hybrid-RAG, Adaptive-RAG, Self-RAG, FLARE, CRAG,
  and closed-book comparison implementations.
- Main benchmark, component ablation, sensitivity, official RAGAS 0.1.21,
  reranker training, and metric implementations.
- IndStd 500/100/200 splits and the fixed HotPotQA subset.
- Reranker training code configured for the paper's 500-question training
  split, two epochs, batch size 16, and fixed seed 42.

## Excluded

- Source standards, extracted standard text, and derived industrial indexes.
- Private IP addresses, local checkpoint paths, credentials, and machine-specific
  environment files.
- Manuscript sources, paper figures, plotting scripts, intermediate result files,
  frozen replay responses, audit scripts, test-only utilities, and UI code.

## External assets required for a full run

1. Authorized source standards or a compatible clause corpus.
2. A clause file (`clauses.jsonl`) and matching FAISS/KG assets built with the
   released scripts.
3. OpenAI-compatible chat and embedding APIs configured through `.env`.
4. Optional Neo4j credentials when graph retrieval is not served from cache.
5. A paper-aligned reranker checkpoint produced with the released training
   command.

The repository root README defines the commands and environment variables.
