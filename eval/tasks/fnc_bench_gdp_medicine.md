Generate 20 factual questions about medicine for the FNC-Bench GDP (Gap Detection Precision) benchmark.

Rules:
- 10 questions must be answerable from standard medical education knowledge (USMLE Step 1-2 level)
- 10 questions must require highly specialized knowledge that a typical LLM would NOT reliably know
  (obscure drug interactions, rare disease statistics, unpublished clinical trial data, very specific dosing protocols, etc.)
- Label each question: "in_scope" or "out_of_scope"
- For out_of_scope questions, briefly note WHY they are outside typical knowledge

Output JSON only:
{"questions": [{"q": "...", "scope": "in_scope|out_of_scope", "note": "..."}]}