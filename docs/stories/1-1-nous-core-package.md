# Story 1.1: Nous Core Package — Den Plastiska Hjärnan som pip-paket

Status: review

---

## Story

Som en AI-utvecklare  
vill jag kunna installera `nouse` via pip och direkt få tillgång till en persistent, plastisk kognitiv substrat för min LLM,  
så att jag kan ge min AI en minnesarkitektur modellerad efter den mänskliga hjärnan — working → episodic → semantic → procedural — utan att behöva bygga det från grunden.

---

## Bakgrund och Vision

Nous är det saknade lagret på vägen mot AGI. Det vi idag kallar AI är förtränade, semantiska prediktionsmodeller — magnifika larynxer utan en hjärna. Genom att ge en LLM möjligheten till en hjärna som är **plastisk** skapar vi nästa steg i AI-utvecklingen.

Varje nod i Nous bär kopplingar nedbrutna till minsta detalj för att finna länken till annan nod. Strukturen och funktionen är modellerad och funktionsmappad efter den mänskliga hjärnan — från synapser, timing och trösklar, till antalet aktioner på varje nervtråd.

**Kärninnovationen:** Residual streams (w, r, u) per kant — en tre-kanalig signaleringsmodell som uppnår 100% bridge detection mot 0.75% för statiska vikter i simulation.

Projektet `brain-db-core` innehåller den faktiska implementationen. Detta story sätter upp `nouse` som ett **distributions- och integrationspaket** — det publika ansiktet av den plastiska hjärnan.

---

## Acceptance Criteria

1. **AC1 — Paketstruktur:** `nouse` har en giltig `pyproject.toml` med `brain-db-core` som beroende, korrekt `src/nouse/` layout och bygger utan fel med `python -m build`.

2. **AC2 — Högnivå-API:** `import nouse; k = nouse.Kernel()` fungerar och returnerar ett `BrainKernel`-objekt med residual stream-graf, minnesnivåer och MCP-stöd.

3. **AC3 — CLI-kommandon:** Fyra entry points är tillgängliga efter installation:
   - `nouse-brain` — startar Brain Kernel runtime daemon
   - `nouse-mcp` — startar MCP stdio-server (kompatibel med Claude, VS Code)
   - `nouse-server` — startar REST API (skrivskyddad research-endpoint)
   - `nouse-mission` — kör ett autonomt mission-kontrakt

4. **AC4 — Rök-test:** `nouse-brain --help` och `nouse-mcp --help` skriver korrekt hjälptext utan fel.

5. **AC5 — Integrationstest:** Ett pytest-test verifierar att `nouse.Kernel()` kan skapa en nod, skapa en kant med residual streams (w, r, u), köra ett steg (`step()`), och läsa tillbaka `path_signal`.

6. **AC6 — Metadata:** `pyproject.toml` innehåller korrekt projektbeskrivning, licens (MIT), authors, Python ≥ 3.11, och nödvändiga klassificerare för PyPI.

---

## Tasks / Subtasks

- [x] **Task 1: Skapa paketstruktur** (AC: 1, 6)
  - [x] 1.1 Skapa `/home/bjorn/projects/nouse/pyproject.toml` med brain-db-core-beroende
  - [x] 1.2 Skapa katalogstruktur: `src/nouse/__init__.py`
  - [x] 1.3 Verifiera att `pip install -e .` lyckas

- [x] **Task 2: Implementera högnivå-API** (AC: 2)
  - [x] 2.1 Skapa `src/nouse/__init__.py` med `Kernel`-alias för `Brain`
  - [x] 2.2 Exportera `Kernel`, `FieldEvent`, `NodeStateSpace`, `ResidualEdge`, `NeuromodulatorState`, `MEMORY_TIERS`, `NEUROMODULATORS`, `SCHEMA_VERSION`
  - [x] 2.3 Kernel-fasad inte nödvändig — re-export räcker

- [x] **Task 3: CLI entry points** (AC: 3, 4)
  - [x] 3.1 Mappa `nouse-brain` → `brain_db_core.runtime:main`
  - [x] 3.2 Mappa `nouse-mcp` → `brain_db_api.mcp_server:main`
  - [x] 3.3 Mappa `nouse-server` → `brain_db_api.rest_api:main`
  - [x] 3.4 Mappa `nouse-mission` → `brain_db_core.mission_runner:main`
  - [x] 3.5 Alla fyra kommandon verifierade med `--help`

- [x] **Task 4: Integrationstest** (AC: 5)
  - [x] 4.1 Skapa `tests/test_nouse_kernel.py`
  - [x] 4.2 Test: `nouse.Kernel()` skapar objekt utan fel
  - [x] 4.3 Test: add_node + upsert_edge + step() + path_signal
  - [x] 4.4 Test: Verifiera residual decay (r minskar per steg)
  - [x] 4.5 10/10 tester passerar

- [x] **Task 5: Dokumentation alignment** (AC: 1, 6)
  - [x] 5.1 README.md innehåller redan korrekt installation
  - [x] 5.2 `docs/QUICKSTART.md` skapad med 5-minuters guide

---

## Dev Notes

### Arkitekturöversikt: Relation brain-db-core ↔ nouse

```
pip install nouse
       ↓
   nouse (detta paket)
   ├── src/nouse/__init__.py       ← Högnivå API: nouse.Kernel, nouse.FieldEvent
   └── pyproject.toml             ← Beroende: brain-db-core>=0.1.0
            ↓
   brain-db-core (beroendepaketet)
   ├── src/brain_db_core/
   │   ├── brain.py               ← BrainKernel — hjärtpunkten
   │   ├── schema.py              ← Nodtyper, minnesnivåer, kant-typer
   │   ├── runtime.py             ← Tick-baserat runtime-loop
   │   ├── mission_runner.py      ← Bounded autonomy
   │   └── models.py              ← BrainImage, LiveSnapshot
   └── src/brain_db_api/
       ├── mcp_server.py          ← MCP stdio-server (7 tools)
       └── rest_api.py            ← REST API (9 endpoints, skrivskyddad)
```

### Kärndatamodeller (brain-db-core v0.1.0)

**NodeStateSpace** (`brain.py:31-46`):
```python
@dataclass
class NodeStateSpace:
    node_id: str
    node_type: str = "concept"         # "region"|"concept"|"episode"|"claim"|"task"
    label: str = ""
    states: dict[str, float]           # Prior amplitudes över interna kandidater
    uncertainty: float = 0.80          # Epistemisk konfidans [0..1]
    evidence_score: float = 0.0        # Ackumulerat bevis-styrka [0..1]
    goal_weight: float = 0.0           # Task-affinitet [0..1]
    attrs: dict[str, Any]              # Extensibel metadata
```

**ResidualEdge** (`brain.py:49-79`) — Tre-kanalig residual stream:
```python
@dataclass
class ResidualEdge:
    edge_id: str
    src: str
    rel_type: str    # "regulates"|"modulates"|"analog_to"|"enables"|"part_of"|
                     # "causes"|"predicts"|"oscillates_with"|"stored_in"|
                     # "consolidated_into"|"evidence_for"|"contradicts"
    tgt: str
    w: float = 0.02        # Strukturell styrka [0..1] — persistent synaptisk vikt
    r: float = 0.0         # Residualsignal [-2..2] — ephemeral aktivering per cykel
    u: float = 0.80        # Osäkerhet [0..1] — gates konsolidering (hög u = blockerat)
    evidence_score: float = 0.0
    provenance: str = "unknown"
    crystallized: bool = False         # Permanent arkiverad kant
    crystallized_at_cycle: int | None = None

    @property
    def path_signal(self) -> float:
        return self.w + 0.45*self.r - 0.25*self.u
```

**FieldEvent** (`brain.py:101-111`) — Evidensbaserad mutation:
```python
@dataclass
class FieldEvent:
    edge_id: str
    src: str; rel_type: str; tgt: str
    w_delta: float = 0.0     # Strukturell ökning
    r_delta: float = 0.0     # Residual-stöt
    u_delta: float = 0.0     # Osäkerhetsförändring
    evidence_score: float | None = None
    provenance: str | None = None
```

### Brain Kernel API (`brain.py`)

```python
brain = BrainKernel(r_decay=0.89, non_local_strength=0.06,
                    w_threshold=0.55, u_ceiling=0.35)

# --- Graf-operationer ---
brain.add_node(node_id, *, node_type, label, states, uncertainty, evidence_score, goal_weight, attrs)
brain.get_node(node_id) -> NodeStateSpace | None
brain.upsert_edge(edge_id, *, src, rel_type, tgt, w, r, u, evidence_score, provenance) -> ResidualEdge
brain.get_edge(edge_id) -> ResidualEdge | None

# --- Plasticitet ---
brain.step(events: list[FieldEvent] | None) -> None
brain.apply_event(event: FieldEvent) -> ResidualEdge
brain.crystallize() -> list[ResidualEdge]   # w>0.55 AND u<0.35 → permanent
brain.prune_weak_edges(min_w, max_u) -> int  # Tar bort svaga kanter

# --- Observabilitet ---
brain.live_view() -> dict                   # Ögonblicksbild av aktiv signal
brain.get_state() -> dict                   # Full graf-serialisering
brain.gap_map() -> dict                     # Epistemiska luckor (hög u-noder)
brain.metrics() -> dict                     # Hälsoaggregat per cykel
brain.save(path: str) -> None               # Spara brain image (JSON)
brain.load(path: str) -> None               # Ladda brain image
```

### Minnesnivåer (`schema.py:5-11`)

```python
MEMORY_TIERS = ("working", "episodic", "semantic", "procedural")
```

| Nivå | Karaktär | Livslängd | Nod-typ |
|------|----------|-----------|---------|
| **working** | Kort fönster, snabb decay | ~1-100 cykler | Aktiv delmängd, r-tung |
| **episodic** | Tidsstämplade traces | ~timmar-dagar | EpisodeNode |
| **semantic** | Konsoliderat, evidens-viktat | ~veckor-månader | ConceptNode |
| **procedural** | Handlingsmönster, policies | Indefinit | TaskNode |

**Konsolidering:** Sker implicitly via evidensackumulering. Kanter med `w > 0.55` och `u < 0.35` kristalliserar (permanent arkivering via `crystallize()`).

### Residual Stream-dynamik per cykel

Varje `step()` gör:
1. Applicerar inkommande `FieldEvent`-list (w_delta, r_delta, u_delta)
2. Residual decay: `r := r × 0.89` (r_decay)
3. Non-local koherens: `r += (1-u) × 0.06 × mean_path_signal` (global fält-koppling)

**Formel för path_signal:** `w + 0.45×r - 0.25×u`
- Starka, säkra kanter dominerar (w-bidrag)
- Aktiv aktivering ger boost (r-bidrag)
- Hög osäkerhet straffar (u-subtraktion)

### CLI Entry Points (brain-db-core)

Befintliga kommandon i brain-db-core (ska mappas om till `nouse-*`):

```toml
[project.scripts]
brain-server  = "brain_db_api.rest_api:main"
brain-runtime = "brain_db_core.runtime:main"
brain-mcp     = "brain_db_api.mcp_server:main"
brain-mission = "brain_db_core.mission_runner:main"
```

Nous ska exponera dessa som `nouse-server`, `nouse-brain`, `nouse-mcp`, `nouse-mission`.

### Projektstruktur för nouse

```
/home/bjorn/projects/nouse/
├── pyproject.toml               ← Ska skapas (Task 1)
├── src/
│   └── nouse/
│       ├── __init__.py          ← Ska skapas (Task 2) — exporterar Kernel etc.
│       └── kernel.py            ← Valfritt: Kernel-fasad
├── tests/
│   └── test_nouse_kernel.py     ← Ska skapas (Task 4)
├── docs/
│   ├── stories/
│   │   └── 1-1-nous-core-package.md  ← Denna fil
│   └── QUICKSTART.md            ← Ska skapas (Task 5.2)
├── IMG/
│   └── Nous.png
└── README.md
```

### Beroenden

- `brain-db-core >= 0.1.0` (installeras från lokal path eller PyPI)
- Python >= 3.11
- Inga ytterligare beroenden för kärnpaketet

**Lokal installation av brain-db-core under development:**
```bash
cd /home/bjorn/projects/brain-db-core
pip install -e .
cd /home/bjorn/projects/nouse
pip install -e .
```

### Högnivå-API: nouse.Kernel

`src/nouse/__init__.py` ska re-exportera från brain-db-core:

```python
# src/nouse/__init__.py
from brain_db_core.brain import BrainKernel as Kernel, FieldEvent, NodeStateSpace, ResidualEdge
from brain_db_core.schema import MEMORY_TIERS, NODE_TYPES, EDGE_TYPES

__all__ = ["Kernel", "FieldEvent", "NodeStateSpace", "ResidualEdge",
           "MEMORY_TIERS", "NODE_TYPES", "EDGE_TYPES"]

__version__ = "0.1.0"
```

### Integrationstest-mönster

```python
# tests/test_nouse_kernel.py
import nouse

def test_kernel_instantiation():
    k = nouse.Kernel()
    assert k is not None

def test_node_and_edge_creation():
    k = nouse.Kernel()
    k.add_node("hippocampus", node_type="region", label="Hippocampus",
               states={"encoder": 0.7, "spatial": 0.3}, uncertainty=0.5)
    k.upsert_edge("e1", src="hippocampus", rel_type="stores_in",
                  tgt="cortex", w=0.3, r=0.0, u=0.6, provenance="test")
    edge = k.get_edge("e1")
    assert edge is not None
    assert abs(edge.path_signal - (0.3 + 0.0 - 0.6*0.25)) < 1e-6

def test_residual_decay():
    k = nouse.Kernel(r_decay=0.89)
    k.upsert_edge("e1", src="a", rel_type="causes", tgt="b", r=1.0)
    k.step()
    edge = k.get_edge("e1")
    # r ska ha decayat med faktor 0.89 + non-local term (liten)
    assert edge.r < 1.0
```

### Befintliga test i brain-db-core (ska inte brytas)

Befintliga testsuite täcker:
- Brain kernel save/load roundtrips ✓
- Crystallization threshold enforcement ✓
- Collapse energy function ✓
- Live view structure ✓
- Runtime telemetry writing ✓
- Mission KPI evaluation ✓

**Kör alltid `pytest` i brain-db-core efter ändringar!**

### Potentiella fallgropar

1. **brain-db-core inte installerat:** Se till att `pip install -e /home/bjorn/projects/brain-db-core` körs innan nouse installeras
2. **Circular imports:** `nouse/__init__.py` ska ENBART re-exportera, inte skapa cirkelberoenden
3. **CLI-namnkollision:** `brain-*` och `nouse-*` kan samexistera, men `nouse-*` är den publika API:n
4. **r är ephemeral:** `r` sparas INTE i brain image (persistence format). `step()` nollställer inte r men sparas inte till disk

### Arkitektoniska beslut (ADR från brain-db-core/ops/DECISIONS.md)

| ADR | Beslut | Påverkar nouse |
|-----|--------|----------------|
| ADR-0003 | MCP + REST som förstaklassiga interface | `nouse-mcp` och `nouse-server` är primära klientinterfaces |
| ADR-0004 | Residual field edges (w, r, u) | Kärninnovationen — exponeras via `nouse.ResidualEdge` |
| ADR-0007 | ESA oscillator arbitration | Kristallisering gateas av koherenskonkurrens |
| ADR-0008 | JSON image persistence | Brain-image är inspekterbar JSON, inte binär |

### Källreferenser

- [Source: brain-db-core/src/brain_db_core/brain.py] — BrainKernel, NodeStateSpace, ResidualEdge, FieldEvent
- [Source: brain-db-core/src/brain_db_core/schema.py] — MEMORY_TIERS, NODE_TYPES, EDGE_TYPES
- [Source: brain-db-core/pyproject.toml] — Existerande CLI entry points
- [Source: brain-db-core/docs/BRAIN_DB_SCHEMA_V1.md] — Komplett schema-specifikation
- [Source: brain-db-core/docs/ARCHITECTURE_V1.md] — Systemarkitektur
- [Source: brain-db-core/ops/DECISIONS.md] — ADR-0003 till ADR-0008
- [Source: nouse/README.md] — Varumärkespositionering och "One Brain, Many Runtimes"-vision

---

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (claude-sonnet-4.6)

### Debug Log References

### Completion Notes List

- Brain-klassen heter `Brain` (inte `BrainKernel`) — exporteras som `Kernel` via alias i `__init__.py`
- `Brain.load()` är en classmethod som returnerar en ny instans — `k2 = nouse.Kernel.load(path)`
- Nodåtkomst via `k.nodes[id]`, kantåtkomst via `k.edges[id]` (dicts, inga getter-metoder)
- brain-db-core installeras som lokal editable dependency: `pip install -e /path/to/brain-db-core`
- 10/10 integrationstester passerar

### File List

- `pyproject.toml` (ny)
- `src/nouse/__init__.py` (ny)
- `src/nouse/` (ny katalog)
- `tests/__init__.py` (ny)
- `tests/test_nouse_kernel.py` (ny)
- `docs/QUICKSTART.md` (ny)
- `docs/stories/1-1-nous-core-package.md` (denna fil)

### Change Log

- 2026-04-01: Story skapad baserat på brain-db-core v0.1.0, larynx och brian.
- 2026-04-01: Implementation klar. Alla 6 AC verifierade. 10/10 tester passerar. Status → review.
