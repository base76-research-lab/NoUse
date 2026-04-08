---
name: "LLM domain for session memory & handoff"
about: "Implement LLM domain for session memory, agent handoff, and knowledge extraction"
title: "[core] LLM domain for session memory, handoff & knowledge extraction"
labels: [enhancement, core, session, agent, llm]
assignees: ['base76-research-lab']
---

## Problem
NoUse saknar idag en domänspecifik LLM-session som explicit stödjer:
- Session memory mellan agentkörningar och modellskiften
- Standardiserad handoff mellan agenter/modeller
- Kunskapsutvinning ur sessionhistorik

## Förslag
1. Skapa en "llm_domain"-struktur i session state/event-logg
2. Definiera handoff-protokoll (vem, när, varför, state)
3. Lägg till stöd för att extrahera insikter/kunskap ur sessionhistorik

## Motivation
- Kontinuitet och spårbarhet mellan agentkörningar
- Möjliggör audit trail och kunskapsutvinning
- Robusthet vid modell/agent-skiften

## Acceptanskriterier
- [ ] "llm_domain"-fält i session state
- [ ] Handoff-protokoll mellan agenter/modeller
- [ ] Funktion för att extrahera insikter ur sessionhistorik

## Relaterat
- session_state
- session_events
- agent/handoff

---
*Skapad av Jasper (Copilot) på uppdrag av Björn, 2026-04-07*