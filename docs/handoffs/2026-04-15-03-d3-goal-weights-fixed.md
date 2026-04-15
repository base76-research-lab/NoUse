# Handoff 2026-04-15-03: D3 Goal Weights Fixed

## Vad hände

D3 goal-directed execution-pipelinen var implementerad men **goal_weights applicerades aldrig på grafen**. Anledningen: `main.py` anropade `field.decay_goal_weights()` och `field.apply_goal_weights()` med `hasattr()`-guard, men `FieldSurface` saknade dessa metoder — de fanns bara på `Brain`.

## Fix

Lade till `apply_goal_weights()` och `decay_goal_weights()` i `src/nouse/field/surface.py`:
- `apply_goal_weights(goals)`: sätter goal_weight på noder som matchar aktiva måls target_concepts
- `decay_goal_weights(rate=0.05)`: minskar goal_weight varje cykel så tillfredsställda mål fades

Verifierat med direkttest: goal_weight=0.9 sattes korrekt, decay fungerar (0.9->0.85).

## Resultat (daemon live)

- NightRun körde och applicerade goal_weights på **39 noder** (första gången!)
- Percolation-mål skapas korrekt: 5 bridge domains (neuroscience prio=0.93, philosophy of mind prio=0.91, analogi/immunologi/kreativitetsteori prio=0.80) + 5 loose node goals (NoUse/kognition/LLMs/samhälle/konfiguration prio=1.00)
- Self-knowledge logging: "50 isolerade, 50 lösa. Targeting: AI-forskning, NoUse, STATUS, Teori"
- Loose node-identifiering: 23,739 isolerade koncept (67% av alla koncept har 0 cross-domain connections)

## Modellkonfig ändrad

Bytte EXTRACT_MODEL från `minimax-m2.7:cloud` till `gemma4:e2b` (lokal) för snabbare extraktion. Ökade timeout till 60s. Lade till model candidates med fallback.

## Nästa steg

- Verifiera att huvudloopen (inte bara NightRun) kör D3 goal_weights
- D4: Satisfaction & Feedback — stäng målslingan
- D5: Policy Integration — goal metrics -> cognitive_policy