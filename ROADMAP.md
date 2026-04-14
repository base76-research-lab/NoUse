# Nous ROADMAP

> Single source of truth for project state. Next LLM session: read THIS file + latest handoff, then start working.
> Updated: 2026-04-14

---

## Current Focus

**Daemon source loop fixed — running full cognitive cycles**

All fixar implementerade och daemonen når nu alla discovery-stadier. Varje cykel producerar nervbanor, curiosity loops och cycle reflections. Modellkonfiguration uppdaterad till `minimax-m2.7:cloud` som primär extraktionsmodell (5.8s response time).

Next: D3 (goal-directed execution), PDF ingestion for Stanford AI Index.

---

## System Health

| Check | Status |
|-------|--------|
| Daemon | active (running) since 2026-04-14 22:52, full cognitive cycles |
| PyPI | v0.4.0 published |
| Core imports | brain, surface, inject, limbic, mcp — OK |
| Persona module | `nouse.persona` — stub created, all imports working |
| Source loop | Fixed: 50 doc cap/cycle, 600s timeout, batched state saves, DOC_EVERY=20 |
| Extract model | `minimax-m2.7:cloud` (primary), `gemma4:e2b`/`glm-5.1:cloud` (fallback) |
| Test suite | 8 collection errors (web modules) — persona errors fixed |
| Git remote | github.com/base76-research-lab/Nous.git |
| Total commits | 90 |

### Fixed: `nouse.persona` import

Created `src/nouse/persona.py` stub with sensible defaults:
- `persona_identity_seed()` returns dict (name, greeting, mission, personality, values, boundaries)
- `assistant_entity_name()` returns "Nous"
- `agent_identity_policy()`, `assistant_greeting()`, `persona_prompt_fragment()` all implemented
- All values configurable via `NOUSE_*` environment variables

---

## P1-P5 Roadmap (cognitive self-regulation)

- [x] P1 Contradiction Detection (commit 8e30315)
- [x] P2 Evidence Accumulation (commit 8e30315)
- [x] P3 Causal Reflection — reflection-to-policy bridge (commit 435af2d)
- [x] P4 Substrate→LLM Direction — focus agenda, gap questions, hallucination block (commit c67641d)
- [x] P5 Evalving Harness + Operator Feedback (commits 8e30315, b717b25)

## Drive Engine (D1-D6) — autonomous goal system

- [x] D1 Goal Registry — `daemon/goal_registry.py` (419 lines)
- [x] D2 Goal Generator — `daemon/goal_generator.py` (585 lines)
- [ ] D3 Goal-Directed Execution — direct Ghost Q + curiosity toward active goals
  - [ ] D3.1 goal-directed Ghost Q (ghost_q.py)
  - [ ] D3.2 goal-directed curiosity (initiative.py)
  - [ ] D3.3 goal_weight dynamics (brain.py)
  - [ ] D3.4 NightRun integration
- [ ] D4 Satisfaction & Feedback — close the goal loop
  - [ ] D4.1 evaluate_satisfaction() in goal_registry.py
  - [ ] D4.2 CLI: `nouse goal add/list/status`
  - [ ] D4.3 eval_log goal metrics
- [ ] D5 Policy Integration — goal metrics drive cognitive_policy
  - [ ] D5.1 new trigger rules in cognitive_policy.py
  - [ ] D5.2 goal-driven living_core drives
- [ ] D6 Hierarchical goals + multi-step plans (future)

## Frontier Plan (external positioning)

- [ ] Fas 0: System ready — `pytest tests/` passes clean
  - blocker: persona import errors, 8 collection errors
- [ ] Fas 1: Intellectual priority — Larynx Problem on Zenodo (DOI)
  - sub: also Academia.edu + PhilPapers
  - sub: sister paper (Creative Free Energy / F_bisoc)
  - sub: GitHub README presentation
- [ ] Fas 2: Empirical validation — TruthfulQA benchmark
  - 8B without Nous: ~46% | 8B with Nous: ~96% (small test set, not universal)
  - need: lm-eval integration, proper benchmark run
- [ ] Fas 3: Institutional presence — ESA paper + HuggingFace Space
- [ ] Fas 4: Frontier radar — conference submission, researcher outreach

## Publications

- 18 papers on PhilPapers
- Larynx Problem: complete draft, pending Nous DOI
- Age of No Resistance: R&R revision submitted to Acta Sociologica (2026-03-31)

---

## Architecture Quick Reference

```
LLM (Larynx) + Nous (Brain) = Bisociationsmotor

Residual stream edges: w (structural weight) + r (residual signal) + u (uncertainty)
path_signal = w + 0.45*r - 0.25*u
Crystallization: w>0.55 AND u<0.35 → permanent
Decay: r *= 0.89 per step

Memory levels: working → episodic → semantic → procedural
18-step daemon loop in daemon/main.py
Limbic: arousal = 0.4*DA + 0.4*NA + 0.2*ACh (Yerkes-Dodson inverted-U)
F_bisoc = prediction_error + lam * complexity_blend; threshold=0.45
```

## Key Conventions

- brain.py and field/surface.py are the core — change carefully
- daemon/main.py is complex but works — be conservative
- Lab notes: `docs/lab-notes/YYYY-MM-DD-slug.md`
- Handoffs: `docs/handoffs/YYYY-MM-DD-NN.md`
- Code is intertwined with FNC theory — every change has philosophical implications
- Language: code + docs in English, strategic docs in Swedish

## Open Questions / Blockers

1. Ollama model `deepseek-r1:1.5b` not installed (404) — `minimax-m2.7:cloud` works as primary
2. TruthfulQA benchmark — needs GPU time and lm-eval adapter (`src/nouse/eval/lm_eval_adapter.py` does not exist yet)
3. Fas 0 (pytest clean) — web modules still have collection errors
4. `bisoc_candidates=0` — TDA threshold not yet met, normal during growth phase

---

## Recent Decisions

| Date | Decision |
|------|----------|
| 2026-04-14 | Fixed daemon source loop: 50 doc cap, 600s timeout, batched state saves, DOC_EVERY=20 |
| 2026-04-14 | Created nouse.persona stub module (fixes 5-file import blocker) |
| 2026-04-14 | Switched extraction model to minimax-m2.7:cloud (5.8s vs 26s for gemma4) |
| 2026-04-14 | Reset model router state, daemon now reaches bridge_synthesis + curiosity stages |
| 2026-04-14 | Added handoff system (ROADMAP.md + docs/handoffs/) |
| 2026-04-14 | Added Stop hooks: handoff reminder + git push check |
| 2026-04-14 | Added MCP servers: nous-sqlite, arxiv, nouse-mcp |
| 2026-04-14 | Scheduled morning research agent (weekdays 06:17) |
| 2026-04-12 | Rename NoUse → Nous |
| 2026-04-13 | Drive Engine workplan (D1-D5) created |
| 2026-04-13 | P1-P5 implementation documented + 41 tests added |
| 2026-04-13 | DeepMind research note drafted |