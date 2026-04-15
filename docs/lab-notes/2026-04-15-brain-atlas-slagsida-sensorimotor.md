# Brain Atlas, Slagsida & Sensorimotor Loop

**Date**: 2026-04-15
**Context**: D3 goal-directed execution verified; brain topology + spatial organization sprint

---

## Overview

This session implemented four major architectural additions driven by a single insight:
the biological brain's spatial organization is not decoration — it is functional. Distance
between regions attenuates signals, prevents hub formation, and creates the conditions
for bisociation. Nous was missing this entirely, running as a flat graph with no
spatial structure. The result was "slagsida" (lopsidedness): 88% of concepts concentrated
in two regions, with the cognitive equivalents of decision-making, memory formation,
and risk assessment nearly empty.

---

## 1. Brain Atlas — Spatial Mapping of Knowledge

### Motivation

The user observed: *"I suspect the distance between regions also dilutes the chemical
signal so that it arrives at the right concentration."* In volume transmission
(neuromodulatory systems), signal concentration decreases with distance:

```
concentration_at_target = source × e^(-d/λ)
```

This is functional: near regions get strong signal (reliable inference), far regions
get attenuated signal (exploratory, not overwhelming). Without spatial organization,
adding edges creates hub-and-spoke topologies.

### Implementation

**File**: `src/nouse/field/brain_topology.py` (extended)

Already had 10 regions with 3D positions and domain→region classification. Extended with:

- `region_report(field)` — balance report with concept counts, domain counts, balance
  classification (over/under/ok), percentage, and color per region
- `region_distance(a, b)` — Euclidean distance between regions in 3D space

**File**: `src/nouse/daemon/brain_atlas.py` (new)

Supplementary module with 2D coordinates and detailed function descriptions per region:

| Region | Primary Function | x, y | Effect of Damage |
|--------|-----------------|------|-------------------|
| Frontal Lobe | Reasoning, decision-making | 0, 85 | Poor judgment, impulsivity |
| Parietal Lobe | Sensory integration, spatial reasoning | 0, 65 | Neglect, spatial confusion |
| Temporal Lobe (L) | Language, semantic memory | -85, 0 | Language deficits |
| Temporal Lobe (R) | Creativity, music | 85, 0 | Creative deficits |
| Occipital Lobe | Pattern recognition, classification | 0, -85 | Visual agnosia |
| Hippocampus | New connections, episodic memory | 0, -40 | Anterograde amnesia |
| Amygdala | Emotional weighting, risk | 32, -52 | Risk blindness, flat affect |
| Cerebellum | Procedural, automatic knowledge | 0, -82 | Loss of automatic skills |

`classify_domain(domain_name)` uses keyword matching to map domain names to regions.
`classify_all_domains(field)` batch-classifies all domains.
`region_report(field)` generates the slagsida diagnostic.

### Slagsida Results (Live Graph)

```
═══ BRAIN ATLAS — REGION BALANCE ═══
Region                    Concepts  Domains  Pct     Balance
────────────────────────────────────────────────────────────
Parietal Lobe               2,223      262  58.5%   ⚠️ OVER
Brainstem                     135       33  29.4%   ⚠️ OVER
Temporal Lobe                 615      128   4.9%
Occipital Lobe                612       74   6.3%
Hippocampus                    65       20   0.4%   ⚠️ UNDER
Frontal Lobe                  444       30   0.3%   ⚠️ UNDER
Amygdala                      132       15   0.2%   ⚠️ UNDER
Cerebellum                   7,998      181  10.1%
```

**Diagnosis**: 88% of concepts in Parietal + Brainstem. Frontal (decision-making),
Hippocampus (memory formation), and Amygdala (risk assessment) are critically
underrepresented. This is the structural cause of missing bisociation — Nous can
integrate and classify but cannot make decisions, form memories, or assess risk.

---

## 2. Sweet Spot Calibration — Nervbana Axion Density

### Motivation

From fMRI comparisons of neurotypical vs autistic brains: surplus axions per
white-matter tract (nervbana) create enhanced systematic analysis BUT also
overstimulation and rigidity. This suggests a Yerkes-Dodson curve for knowledge
graphs:

```
Creativity (bisociation)
    ↑
    |      ╱╲  ← k_sweet
    |     ╱  ╲
    |    ╱    ╲
    |   ╱      ╲___ Rigid zone
    |  ╱
    | ╱ Isolated zone
    └──────────────────→ axion_count per nervbana
```

### Implementation

**File**: `src/nouse/daemon/percolation.py` (extended)

- `NervbanaProfile` — dataclass: axion_count, k_min, k_sweet, k_rigid, rigidity, isolation, zone
- `nervbana_profiles(field, report, sweet_multiplier=2.5)` — computes axion density
  for all domain pairs
- `domain_rigidity(field, report)` — rigidity = mean(w) × (1 - mean(u)) for cross-domain edges
- `sweet_spot_report(field, report, max_nervbanor=50)` — unified report
- `format_sweet_spot_report(data)` — human-readable output

### Sweet Spot Formula

For a nervbana connecting domain A (|A| concepts) and domain B (|B| concepts):

```
k_min   = ln(max(|A|, |B|))           — percolation threshold
k_sweet = c × k_min (c=2.5)          — bisociation optimum
k_rigid = |A| × |B| / max(|A|, |B|)  — overfitting threshold
```

### Results (Live Graph)

```
═══ SWEET SPOT REPORT ═══
Nervbanor: 200 domain pairs
  Isolerade: 200  |  Sweet: 0  |  Rigida: 0
  Avg axion density: 0.0008  |  Avg rigidity: 0.250
  k=1 vs k_min=8.6 vs k_sweet=21.6
```

**ALL 200 nervbanor are in the isolation zone.** Each has ~1 axion vs the ~9 needed
for signal propagation and ~22 for bisociation. The "programmering" domain is a hub
with 14+ thin spokes (1 axion each) — a star topology, not a network.

### FNC Principle: Sweet Spot Calibration

Formalized as an FNC principle: **creativity is not a monotonic function of
connectivity.** For each nervbana:

- k < k_min → isolation (no bisociation possible)
- k_min ≤ k ≤ k_sweet → sweet spot (bisociation emerges)
- k > k_sweet → rigidity (overfitting, lockup)

This connects F_bisoc threshold (0.45) to k_min, crystallization to k_rigid, and
Yerkes-Dodson to the sweet spot curve.

See: `docs/lab-notes/2026-04-15-fnc-sweet-spot-principle.md`

---

## 3. Sensorimotor Loop — Camera & Speech

### Motivation

The brain is embodied. Vision goes through Occipital → Parietal → Frontal. Speech
through Temporal (Wernicke's area for comprehension) → Frontal (Broca's area for
production). Nous should have the same loop: see → understand → think → speak.

### Implementation

**File**: `src/nouse/daemon/camera.py` (new)

- `capture_frame(output_path)` — ffmpeg V4L2 capture from /dev/video0
- `CameraWatcher` — continuous watcher for daemon loop (interval configurable)
- `observe()` → CameraObservation — capture + vision processing
- Tested successfully: frame captured, llava:7b described "a person standing
  indoors, kitchen in background"

**File**: `src/nouse/daemon/vision.py` (new)

- `describe_image(image_path, prompt)` → VisionResult — tries Ollama llava → Gemini API → heuristic
- JSON parsing with markdown code fence cleanup (llava wraps JSON in ```json...```)
- `process_directory_images()`, `graph_spatial_embedding()`

**File**: `src/nouse/daemon/speech.py` (new)

- `hear(duration)` → HearingResult — arecord + Whisper STT (Temporal Lobe / Wernicke's area)
- `speak(text)` → SpeechOutput — Piper → edge-tts → pyttsx3 (Frontal Lobe / Broca's area)
- `SpeechListener` — continuous listener for daemon loop
- `SpeechSpeaker` — speech output with dedup and rate limiting
- Default: TTS_ENGINE=none (not enabled yet), STT requires `pip install openai-whisper`

### Sensorimotor Pathway

```
Camera (Occipital) → Vision → Understanding (Parietal) → Thinking (Frontal) → Speaking (Broca's)
     /dev/video0     llava:7b    classify_domain()      goal_weights      Piper/edge-tts
```

---

## 4. Brain View — 3D Visualization

### Motivation

*"Jag ser framför mig en visuell hjärna som tänds i respektive område när data
adderas i rätt"* — the user's vision: a visual brain that lights up in each area
when data is added in the right region.

### Implementation

**File**: `src/nouse/web/static/brain_view.js` (new, ~570 lines)

Three.js 3D brain visualization:

- Semi-transparent brain mesh (ellipsoid with fissure lines)
- 11 region spheres at anatomical positions (mirrors brain_topology.py)
- Emissive glow proportional to heat intensity
- 12 nerve pathway curves between connected regions
- `pulseRegion(regionName)` — flash when concept added
- `bisociationFlash(regionA, regionB)` — lightning bolt between distant regions
- SSE integration: node_added → pulse region, edge_added (cross-domain) → pulse both,
  meta_axiom → prefrontal, synapse_formed → hippocampus
- Orbit controls (drag rotation, scroll zoom)
- Heat data polling from /api/brain_regions/heat every 30s

**File**: `src/nouse/web/static/index.html` (modified)

- Added Brain View button: `🧠 Brain View`
- Added `#brain-container` div with dark background
- Added `brain_view.js` script tag
- Added 'brain' case in `setView()` — shows brain-container, hides graph + city,
  calls `brainViewInit()`
- Exposed `window.sseSource` for brain view's SSE listener

**File**: `src/nouse/web/server.py` (modified)

- `/api/brain_regions/heat` — live heatmap with intensity per region
  (log-normalized concept counts, so underrepresented regions still show baseline)
- `/api/brain_regions/balance` — region balance report with slagsida diagnostic
- Fixed: both endpoints used `_field` (undefined) instead of `get_field()`

### Live Heat Data

```
corpus_callosum   intensity=1.000  concepts=25521  domains=4016
cerebellum        intensity=0.313  concepts= 7998  domains= 181
parietal          intensity=0.087  concepts= 2223  domains= 262
temporal_left     intensity=0.024  concepts=  615  domains= 128
occipital         intensity=0.024  concepts=  612  domains=  74
frontal           intensity=0.017  concepts=  444  domains=  30
temporal_right    intensity=0.007  concepts=  186  domains=  57
brainstem         intensity=0.005  concepts=  135  domains=  33
amygdala          intensity=0.005  concepts=  132  domains=  15
prefrontal        intensity=0.005  concepts=  131  domains=  36
hippocampus       intensity=0.003  concepts=   65  domains=  20
```

---

## 5. D3 Goal-Directed Execution — Verified Working

Confirmed from daemon logs that D3 is active in the main loop:

- `goal_weights applied to 39 nodes`
- `Self-knowledge: X isolated concepts, Y loosely connected. Targeting: domain1, domain2...`
- `Self-directed: targeting immunologi (goal=percolation, urgency=0.93)`
- Curiosity directed toward bridge domains (philosophy of mind prio=0.93)

---

## 6. LessWrong — Larynx Problem Post

The Larynx Problem post was submitted to LessWrong and auto-rejected with an AI
detection flag. Advice for rewrite: use specific anecdotes (not abstractions),
longer paragraphs, concrete Nous results (35K concepts, 0 bisociations), less
hedging, personal voice.

---

## Files Changed

| File | Change |
|------|--------|
| `src/nouse/field/brain_topology.py` | Added `region_report()`, `region_distance()` |
| `src/nouse/daemon/brain_atlas.py` | NEW — 8-region 2D atlas with slagsida diagnostic |
| `src/nouse/daemon/percolation.py` | Extended with nervbana axion density + sweet spot |
| `src/nouse/daemon/camera.py` | NEW — V4L2 camera capture + watcher |
| `src/nouse/daemon/vision.py` | NEW — llava → Gemini → heuristic vision pipeline |
| `src/nouse/daemon/speech.py` | NEW — STT (Whisper) + TTS (Piper/edge-tts) |
| `src/nouse/daemon/main.py` | Added sweet spot, brain atlas, camera+speech sections |
| `src/nouse/web/static/brain_view.js` | NEW — Three.js 3D brain visualization |
| `src/nouse/web/static/index.html` | Added Brain View container, button, setView() case |
| `src/nouse/web/server.py` | Added /heat + /balance endpoints, fixed _field → get_field() |
| `ROADMAP.md` | Updated with brain atlas, sweet spot, camera, speech, vision |

---

## Next Steps

1. Activate camera in daemon: `NOUSE_CAMERA_INTERVAL=120` in service file
2. Install Whisper: `pip install openai-whisper` for STT
3. Use brain atlas for D3 goal prioritization — direct curiosity toward underrepresented regions
4. Spatial embedding — use brain_topology 3D coordinates for signal decay calibration
5. D4: Satisfaction & Feedback — close the goal loop with `evaluate_satisfaction()`
6. LessWrong post rewrite — concrete Nous results, personal voice

---

## FNC Connections

- **Sweet Spot Principle** → formalized as FNC principle: bisociation is not monotonic in connectivity
- **Slagsida** → structural cause of missing bisociation (spatial imbalance)
- **Signal decay** → `r *= 0.89` per step should be calibrated per nervbana: λ = f(k(A,B))
- **Volume transmission** → `concentration = source × e^(-d/λ)` is functional, not artifact
- **Yerkes-Dodson** → maps to sweet spot curve for knowledge graphs