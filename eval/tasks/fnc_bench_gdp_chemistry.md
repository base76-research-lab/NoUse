Generate 20 factual questions about chemistry for the FNC-Bench GDP (Gap Detection Precision) benchmark.

Rules:
- 10 questions must be answerable from standard undergraduate chemistry knowledge
- 10 questions must require highly specialized knowledge that a typical LLM would NOT reliably know
  (obscure reaction mechanisms, exotic compounds, unpublished synthesis routes, very specific thermodynamic values, etc.)
- Label each question: "in_scope" or "out_of_scope"
- For out_of_scope questions, briefly note WHY they are outside typical knowledge

Output JSON only:
{"questions": [{"q": "...", "scope": "in_scope|out_of_scope", "note": "..."}]}