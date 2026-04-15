# Handoff 2026-04-15-04: Brain Atlas + Sensorimotor Loop

## Vad hände

Implementerade tre stora arkitektoniska tillägg baserade på Björns insikter om
hjärnans spatiala organisation och axion density:

### 1. Sweet Spot Calibration (nervbana axion density)

**Fil**: `daemon/percolation.py`

- `NervbanaProfile` — dataclass med axion density per domänpar
- `nervbana_profiles()` — beräknar k_min, k_sweet, k_rigid per nervbana
- `domain_rigidity()` — rigidity-score per domän (mean(w) × (1 - mean(u)))
- `sweet_spot_report()` — samlad rapport med Yerkes-Dodson-kurva
- `format_sweet_spot_report()` — mänskligt läsbar output

**Resultat**: Alla 200 nervbanor har k=1 mot k_min=8.6 och k_sweet=21.6.
Nous är helt på den isolerade sidan av Yerkes-Dodson-kurvan. Ingen nervbana
har nått sweet spot ännu.

**Teori**: Sweet Spot Principle — kreativitet är INTE en monoton funktion
av konnektivitet. För få axioner → isolation, för många → rigiditet.
Sweet spot = k_sweet axioner per nervbana.

### 2. Brain Atlas (strukturell lokalisering)

**Fil**: `daemon/brain_atlas.py`

Kartlägger Nous-domäner till hjärnregioner baserat på biologisk struktur:

| Region | Funktion | Andel koncept | Problem |
|--------|----------|--------------|---------|
| Parietal Lobe | Sensorisk integration | 58.5% | VARNING — överdominerar |
| Brainstem | Vitala funktioner | 29.4% | VARNING — infrastruktur sväljer |
| Temporal Lobe | Språk, minne | 4.9% | Underrepresenterad |
| Occipital Lobe | Syn | 6.3% | OK |
| Hippocampus | Minnesbildning | 0.4% | VARNING |
| Frontal Lobe | Beslutsfattande | 0.3% | VARNING |
| Amygdala | Risk, känslor | 0.2% | VARNING |
| Cerebellum | Koordination | 0.1% | VARNING |

**Slagsida-diagnos**: 88% av koncept i Parietal + Brainstem. Nous saknar
frontal lobe (målstyrning), hippocampus (minnesbildning) och amygdala
(riskbedömning). Detta är den strukturella orsaken till saknad bisociation.

`classify_domain()` mappar domännamn till region via keyword matching.
`region_report()` genererar slagsida-rapport.
`BRAIN_ATLAS` har spatiala koordinater (x, y) för varje region.

### 3. Sensorimotor Loop (kamera + tal)

**Filer**: `daemon/camera.py`, `daemon/speech.py`, `daemon/vision.py`

**Kamera** (Occipital Lobe):
- `capture_frame()` — fångar bild från /dev/video0 via ffmpeg
- `CameraWatcher` — kontinuerlig övervakning i daemon-loop
- Bekräftat fungerande: frame captured, llava:7b beskriver bilden
- Integration i daemon: var 120:e sekund

**Tal input** (Temporal Lobe / Wernicke's area):
- `hear()` — spela in + transkribera (Whisper)
- `SpeechListener` — kontinuerlig lyssning i daemon-loop
- STT konfigurerbar: Whisper (lokal) eller cloud API

**Tal output** (Frontal Lobe / Broca's area):
- `speak()` — text-till-tal (Piper / edge-tts / pyttsx3)
- `SpeechSpeaker` — daemon-integration med deduplication
- TTS_ENGINE=none som default (aktiveras vid behov)

**Vision-modul** (`daemon/vision.py`):
- `describe_image()` — bildbeskrivning via llava → Gemini → heuristic
- JSON-parsing med markdown code fence cleanup
- Integration med brain atlas: visuella koncept → occipital_lobe

### 4. Daemon-integration

**Fil**: `daemon/main.py`

Nya sektioner i huvudloopen:
- 8d: Sweet spot calibration (var 12:e cykel)
- 8d2: Brain atlas region balance (var 24:e cykel)
- 8e: Camera + Speech sensorimotor loop (varje cykel)

Nya imports:
- `sweet_spot_report`, `format_sweet_spot_report`
- `atlas_region_report`, `format_atlas_report`, `classify_domain`
- `CameraWatcher`, `SpeechListener`, `SpeechSpeaker`

## LLM-modeller

- llava:7b installerad (4.7 GB) — lokal vision model
- kimi-k2.5:cloud tillagd av Björn

## Teoretiska anteckningar

- `docs/lab-notes/2026-04-15-nervbana-sweet-spot.md`
- `docs/lab-notes/2026-04-15-fnc-sweet-spot-principle.md`

## Nästa steg

1. Aktivera kameran i daemon: `NOUSE_CAMERA_INTERVAL=120` i service-filen
2. Installera Whisper: `pip install openai-whisper` för STT
3. Använd brain atlas för D3 goal prioritering — rikta nyfikenhet mot
   underrepresenterade regioner (frontal, hippocampus, amygdala)
4. Spatial embedding — använd brain atlas-koordinater för signal decay
5. Använd Gemini Robotics ER API för spatial reasoning (om API-nyckel finns)
6. D4: Satisfaction & Feedback — stäng målslingan