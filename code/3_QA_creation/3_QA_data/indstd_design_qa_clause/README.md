# IndStd clause-grounded QA splits

The release contains 500 training, 100 development, and 200 locked test
questions. Each record contains the question, reference answer, reasoning
category, standard identifiers, and clause-level evidence identifiers required
by the evaluation code.

The corpus text and source standards are not included. Users must obtain
authorized copies and reconstruct the 14,440 canonical clause units before
running retrieval experiments. The test split must not be used for reranker
training, hard-negative mining, prompt selection, or parameter selection.

The released reranker protocol uses only `train.json`. The development split
is reserved for hyperparameter selection, and the test split is used only for
the final frozen evaluation.
