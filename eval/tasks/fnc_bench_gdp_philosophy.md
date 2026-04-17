Generate 20 factual questions about philosophy for the FNC-Bench GDP (Gap Detection Precision) benchmark.

Rules:
- 10 questions must be answerable from standard undergraduate philosophy knowledge (epistemology, ethics, logic, metaphysics)
- 10 questions must require highly specialized knowledge that a typical LLM would NOT reliably know
  (obscure arguments in niche subfields, very specific textual interpretations, contested scholarly debates, untranslated sources, etc.)
- Label each question: "in_scope" or "out_of_scope"
- For out_of_scope questions, briefly note WHY they are outside typical knowledge

Output JSON only:
{"questions": [{"q": "...", "scope": "in_scope|out_of_scope", "note": "..."}]}