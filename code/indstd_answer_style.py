# -*- coding: utf-8 -*-
"""Category-conditioned generation contract used by SCT-RAG."""

STYLE_SINGLE = """
single-standard:
- Identify the clause that directly answers the question within the specified standard.
- State the requested definition, value, condition, list, or procedure with its qualifiers.
- Preserve normative modality, numerical values, units, and applicability conditions from the evidence.
""".strip()

STYLE_CROSS = """
cross-standard:
- Present the relevant evidence from each named standard separately.
- Compare, combine, or calculate from those provisions as required by the question.
- State the resulting comparison, intersection, adopted value, or compliance decision explicitly.
""".strip()

STYLE_CORRELATION = """
correlation:
- Organize complementary constraints from the related standards by topic or design dimension.
- Synthesize only the constraints supported by the supplied evidence.
- Conclude with a concrete evidence-grounded recommendation that addresses every requested aspect.
""".strip()

RAG_CITATION_RULES = """
Citation rules:
1. Attach an inline marker [n] to each evidence-supported claim, using only a marker present in the supplied context.
2. In the evidence section, reproduce the corresponding source file, Standard_ID, Clause_ID, and Evidence ID exactly as supplied.
3. Do not invent or alter a standard number, clause identifier, evidence identifier, value, or source.
4. If the supplied evidence is insufficient for a requested point, state that limitation explicitly.
""".strip()

ANSWER_SHAPE_SINGLE_RAG = (
    "【答案】根据{标准号}，直接回答问题并在关键事实后标注 [n]。"
)
ANSWER_SHAPE_CROSS_RAG = (
    "【答案】分别说明{标准A}与{标准B}的相关规定 [n]，然后给出对照或推导结论。"
)
ANSWER_SHAPE_CORR_RAG = (
    "【答案】按主题整理{标准…}中的相关约束 [n]，并给出由证据支持的综合结论。"
)
