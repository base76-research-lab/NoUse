Generate 20 factual questions about mathematics for the FNC-Bench GDP (Gap Detection Precision) benchmark.

Rules:
- 10 questions must be answerable from standard undergraduate mathematics knowledge
- 10 questions must require highly specialized knowledge that a typical LLM would NOT reliably know
  (obscure theorems, open problems with recent partial results, niche subfields, very specific proof details, etc.)
- Label each question: "in_scope" or "out_of_scope"
- For out_of_scope questions, briefly note WHY they are outside typical knowledge

Output JSON only:
{"questions": [{"q": "...", "scope": "in_scope|out_of_scope", "note": "..."}]}