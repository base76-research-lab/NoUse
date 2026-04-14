---
title: "Daemon Cognitive Cycle Fix"
subtitle: "From Stuck Ingestion to Full Bisociative Discovery"
author: "Björn Wikström"
date: "2026-04-14"
abstract: |
  The Nous daemon was stuck in the `source_progress` journal stage and never reached discovery stages (BFS nervbanor, TDA bisociation, curiosity loop, bridge synthesis). Root cause: an unbounded source ingestion loop combined with per-file disk persistence. Fix: (1) cap source docs per cycle to 50, (2) add per-source timeout of 600s, (3) batch state persistence from per-file to every-50-files, (4) increase SOURCE_PROGRESS_DOC_EVERY from 3 to 20. Additionally, the `nouse.persona` module (imported by 5 files) did not exist, preventing daemon startup after restart — fixed with a stub module returning a properly structured identity dict. Model configuration updated from unavailable `deepseek-r1:1.5b` to `minimax-m2.7:cloud` (5.8s extraction time). After all fixes, the daemon now completes full cognitive cycles: source_ingest → bridge_synthesis → curiosity_loop → policy_change → cycle_reflection, producing ~100 new relations and 1–4 nervbanor per cycle.
---

# Daemon Cognitive Cycle Fix

**From Stuck Ingestion to Full Bisociative Discovery**

Björn Wikström\
Base76 Research Lab\
bjorn@base76research.com | ORCID: 0009-0000-4015-2357

*14 April 2026*

---

## 1. Problem

The Nous daemon (v0.4.0) had been running continuously since 2026-04-13, accumulating 90+ commits and a knowledge graph of 31k concepts / 33k relations. However, the daily journal (`journal/2026-04-14.md`, 3576 lines) contained **only** `source_progress` entries. The eval log had a single entry (cycle 100). The daemon was stuck in source ingestion and never reached its discovery stages:

- BFS nervbanor (multi-hop domain path finding)
- TDA bisociation candidates (Koestler Step B)
- Bridge discovery and synthesis
- Curiosity loop + gap queue
- Policy adaptation

This meant the system was *accumulating* knowledge but never *connecting* or *discovering* — the core bisociative capability was dormant.

## 2. Root Cause Analysis

### 2.1 Unbounded source loop

`FileSource.read_new()` (in `daemon/sources.py`) iterates over all files in watch directories via `rglob("*")`. For each new/modified file, `extract_relations_with_diagnostics()` calls the LLM with a 45s timeout per attempt. With fallback models, a single file can take 90–135 seconds. The loop has no document cap — it processes ALL new files before yielding to the next cognitive stage.

### 2.2 Per-file state persistence

`_save_state()` writes the entire state dict (20,812 tracked file mtimes) to disk as JSON **after every single file** (line 139 of `sources.py`). For 2000 new files per cycle, this means 2000 JSON serializations and disk writes of a growing 20k-entry dictionary.

### 2.3 Model not found

`deepseek-r1:1.5b` (configured as the primary extraction model) returned HTTP 404 from Ollama — the model was not installed. All extraction attempts failed immediately, and the system fell through to `gemma4:e2b` (26s response time) or `glm-5.1:cloud` (13s but inconsistent), both of which frequently timed out at the 45s limit.

### 2.4 Missing `nouse.persona` module

Five files imported `nouse.persona` which did not exist:

- `self_layer/living_core.py`
- `cli/run.py`
- `cli/chat.py`
- `cli/ask.py`
- `web/server.py`

This prevented the daemon from restarting with the new code — the import error crashed immediately.

## 3. Fixes Applied

### Fix 1: Source document cap per cycle

Added `MAX_SOURCE_DOCS_PER_CYCLE` (default 50, env: `NOUSE_MAX_SOURCE_DOCS_PER_CYCLE`) with a break condition in the source loop:

```python
if source_docs_processed >= MAX_SOURCE_DOCS_PER_CYCLE:
    log.info("Source doc cap reached, deferring remaining docs")
    break
```

### Fix 2: Per-source timeout

Added `SOURCE_TIMEOUT_SEC` (default 600, env: `NOUSE_SOURCE_TIMEOUT_SEC`) with a break condition:

```python
if time.monotonic() - source_start > SOURCE_TIMEOUT_SEC:
    log.warning("Source %s timed out after %ds", source_name, SOURCE_TIMEOUT_SEC)
    break
```

### Fix 3: Batched state persistence

Changed `FileSource.read_new()` and `ConversationSource.read_new()` from per-file `_save_state()` to batched (every 50 files + end-of-source):

```python
_batch_count = 0
for path in self.root.rglob("*"):
    # ... process file ...
    _batch_count += 1
    if _batch_count % 50 == 0:
        _save_state(self._state)
if _batch_count % 50 != 0:
    _save_state(self._state)
```

This reduces disk writes from O(N) to O(N/50).

### Fix 4: Reduced journal verbosity

Changed `SOURCE_PROGRESS_DOC_EVERY` from 3 to 20 (both in code default and systemd service config). This reduces the `source_progress` journal flood from ~200 entries per source to ~50.

### Fix 5: Persona stub module

Created `src/nouse/persona.py` with all required exports:

- `persona_identity_seed()` → returns dict with `name`, `greeting`, `mission`, `personality`, `values`, `boundaries`
- `assistant_entity_name()` → returns "Nous"
- `agent_identity_policy()`, `assistant_greeting()`, `persona_prompt_fragment()` → sensible defaults
- All configurable via `NOUSE_*` environment variables

The critical detail: `persona_identity_seed()` must return a **dict** (not a string) because `living_core._normalize_identity()` treats it as the default identity base and assigns to its keys.

### Fix 6: Model configuration

- Primary extraction model changed from `deepseek-r1:1.5b` (404) to `minimax-m2.7:cloud` (5.8s response time)
- Candidates: `minimax-m2.7:cloud, gemma4:e2b, glm-5.1:cloud`
- Model router state reset (zeroed timeouts, capped history) to favor minimax

## 4. Results

After all fixes, the daemon journal shows all cognitive stages:

| Stage | Count | Content |
|-------|-------|---------|
| `source_progress` | 972 | Ingest progress (reduced from ~900/cycle) |
| `cycle_start` | 7 | Cycle initialization |
| `system_events` | 7 | System event processing |
| `source_ingest` | 5 | Ingest summary |
| **`bridge_synthesis`** | **4** | **BFS nervbanor + TDA bisociation** |
| **`curiosity_loop`** | **4** | **Autonomous curiosity** |
| **`policy_change`** | **4** | **Cognitive policy adaptation** |
| **`cycle_reflection`** | **4** | **Cycle reflection** |

Per cycle metrics (cycles 2447–2450):

- New relations: ~100/cycle
- Nervbanor discovered: 1–4/cycle
- Bisociation candidates: 0 (TDA threshold not yet met — normal during growth phase)
- Graph size: 32k concepts / 34k relations

## 5. Implications

The daemon was not broken — it was **trapped in ingestion**. The cognitive loop architecture is sound, but the source ingestion stage dominated each cycle because it had no exit conditions. The fixes establish proper boundaries:

1. **Temporal boundary**: 600s per source, ensuring no source monopolizes a cycle
2. **Volume boundary**: 50 documents per cycle, ensuring ingestion is bounded
3. **I/O boundary**: Batched persistence, ensuring disk writes don't become the bottleneck

These changes shift the daemon's behavior from "ingest everything first, then think" to "ingest some, think, repeat" — a much healthier operational pattern that allows discovery and adaptation within each cycle rather than only after exhausting all sources.

---

*Lab note for research traceability. All changes committed to `main` branch.*