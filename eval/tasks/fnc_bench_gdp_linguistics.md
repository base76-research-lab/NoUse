Generate 20 factual questions about linguistics for the FNC-Bench GDP (Gap Detection Precision) benchmark.

Rules:
- 10 questions must be answerable from standard undergraduate linguistics knowledge (phonology, syntax, semantics, pragmatics)
- 10 questions must require highly specialized knowledge that a typical LLM would NOT reliably know
  (obscure language-specific phenomena, niche corpus findings, very specific typological data, endangered language documentation details, etc.)
- Label each question: "in_scope" or "out_of_scope"
- For out_of_scope questions, briefly note WHY they are outside typical knowledge

Output JSON only:
{"questions": [{"q": "...", "scope": "in_scope|out_of_scope", "note": "..."}]}