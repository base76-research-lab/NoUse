"""
Extractor — LLM som relationsparsare (Broca's area)
====================================================
Tar text → returnerar typade relationer för grafen.
LLM resonerar INTE. Den parsar strukturerat.
"""
from __future__ import annotations
import json
import logging
import os
import re
import threading
from typing import Any

from nouse.llm.model_router import order_models_for_workload, record_model_result
from nouse.llm.policy import resolve_model_candidates
from nouse.ollama_client.client import AsyncOllama

_log = logging.getLogger("nouse.extractor")

MODEL = (
    os.getenv("NOUSE_EXTRACT_MODEL")
    or os.getenv("NOUSE_OLLAMA_MODEL")
    or "qwen3.5:latest"
).strip()
FALLBACK_MODEL = (os.getenv("NOUSE_EXTRACT_FALLBACK_MODEL") or "").strip()
GLOBAL_CANDIDATES_RAW = (os.getenv("NOUSE_MODEL_CANDIDATES") or "").strip()
EXTRACT_CANDIDATES_RAW = (os.getenv("NOUSE_MODEL_CANDIDATES_EXTRACT") or "").strip()
SYNTH_MODEL = (os.getenv("NOUSE_SYNTH_MODEL") or MODEL).strip()
SYNTH_FALLBACK_MODEL = (os.getenv("NOUSE_SYNTH_FALLBACK_MODEL") or "").strip()
SYNTH_CANDIDATES_RAW = (os.getenv("NOUSE_MODEL_CANDIDATES_SYNTH") or "").strip()

try:
    EXTRACT_MAX_CHARS = max(400, int((os.getenv("NOUSE_EXTRACT_MAX_CHARS") or "2200").strip()))
except ValueError:
    EXTRACT_MAX_CHARS = 2200

try:
    EXTRACT_MAX_RELATIONS = max(1, int((os.getenv("NOUSE_EXTRACT_MAX_RELATIONS") or "15").strip()))
except ValueError:
    EXTRACT_MAX_RELATIONS = 15

try:
    EXTRACT_TIMEOUT_SEC = max(
        1.0,
        float((os.getenv("NOUSE_EXTRACT_TIMEOUT_SEC") or os.getenv("NOUSE_LLM_TIMEOUT_SEC") or "19").strip()),
    )
except ValueError:
    EXTRACT_TIMEOUT_SEC = 19.0

try:
    SYNTH_TIMEOUT_SEC = max(
        1.0,
        float((os.getenv("NOUSE_SYNTH_TIMEOUT_SEC") or os.getenv("NOUSE_LLM_TIMEOUT_SEC") or "12").strip()),
    )
except ValueError:
    SYNTH_TIMEOUT_SEC = 12.0

_BOOL_TRUE = {"1", "true", "yes", "on"}
AUTO_DISCOVER_MODELS = ((os.getenv("NOUSE_MODEL_AUTODISCOVER") or "1").strip().lower() in _BOOL_TRUE)
ENABLE_HEURISTIC_FALLBACK = (
    (os.getenv("NOUSE_EXTRACT_HEURISTIC_FALLBACK", "1") or "").strip().lower() in _BOOL_TRUE
)

RELATION_TYPES = [
    "modulerar","orsakar","konsoliderar","är_del_av",
    "synkroniserar","reglerar","oscillerar","är_analogt_med",
    "stärker","försvagar","producerar","beskriver",
]
_RELATION_TYPE_SET = set(RELATION_TYPES)

_SYSTEM = """Du är en kunskapsextraktor. Extrahera faktiska relationer ur texten.
Returnera BARA giltig JSON-array. Inga förklaringar.

Format:
[
  {"src":"konceptA","domain_src":"domän","type":"relationstyp",
   "tgt":"konceptB","domain_tgt":"domän","why":"kort motivering"}
]

Tillåtna relationstyper: """ + ", ".join(RELATION_TYPES) + """

Regler:
- Bara relationer som faktiskt finns i texten
- Koncept ska vara korta (2-4 ord), svenska eller engelska
- Domän ska vara ett ämnesområde (t.ex. "neurovetenskap", "ekonomi", "fysik")
- why: en mening som förklarar varför kopplingen håller
- Max 15 relationer per text
- Om texten är irrelevant eller för kort: returnera []"""


_RELATION_NORMALIZATION = {
    "leder_till": "orsakar",
    "leder till": "orsakar",
    "causes": "orsakar",
    "cause": "orsakar",
    "causal": "orsakar",
    "modulates": "modulerar",
    "modulate": "modulerar",
    "regulates": "reglerar",
    "regulate": "reglerar",
    "describes": "beskriver",
    "describe": "beskriver",
    "deskriberar": "beskriver",
    "consolidates": "konsoliderar",
    "consolidate": "konsoliderar",
    "is_part_of": "är_del_av",
    "is part of": "är_del_av",
    "part_of": "är_del_av",
    "strengthens": "stärker",
    "strengthen": "stärker",
    "weakens": "försvagar",
    "weaken": "försvagar",
    "produces": "producerar",
    "produce": "producerar",
    "synchronizes": "synkroniserar",
    "synchronize": "synkroniserar",
    "oscillates": "oscillerar",
    "oscillate": "oscillerar",
    "is_analogous_to": "är_analogt_med",
    "is analogous to": "är_analogt_med",
}

_HEURISTIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?P<src>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})\s+"
            r"(?:modulerar|modulates?)\s+"
            r"(?P<tgt>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})",
            re.IGNORECASE,
        ),
        "modulerar",
    ),
    (
        re.compile(
            r"(?P<src>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})\s+"
            r"(?:reglerar|regulates?)\s+"
            r"(?P<tgt>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})",
            re.IGNORECASE,
        ),
        "reglerar",
    ),
    (
        re.compile(
            r"(?P<src>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})\s+"
            r"(?:orsakar|causes?|leder till)\s+"
            r"(?P<tgt>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})",
            re.IGNORECASE,
        ),
        "orsakar",
    ),
    (
        re.compile(
            r"(?P<src>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})\s+"
            r"(?:konsoliderar|consolidates?)\s+"
            r"(?P<tgt>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})",
            re.IGNORECASE,
        ),
        "konsoliderar",
    ),
    (
        re.compile(
            r"(?P<src>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})\s+"
            r"(?:är del av|is part of)\s+"
            r"(?P<tgt>[A-Za-z0-9ÅÄÖåäö_\- ]{2,80})",
            re.IGNORECASE,
        ),
        "är_del_av",
    ),
]

_PHRASE_STOP_TOKENS = {
    "och",
    "and",
    "som",
    "that",
    "under",
    "over",
    "in",
    "i",
    "med",
    "with",
}
_AUTO_DISCOVERY_LOCK = threading.Lock()
_AUTO_DISCOVERED_MODELS: list[str] | None = None


def _autodiscovered_models() -> list[str]:
    if not AUTO_DISCOVER_MODELS:
        return []
    global _AUTO_DISCOVERED_MODELS
    if _AUTO_DISCOVERED_MODELS is not None:
        return list(_AUTO_DISCOVERED_MODELS)

    with _AUTO_DISCOVERY_LOCK:
        if _AUTO_DISCOVERED_MODELS is not None:
            return list(_AUTO_DISCOVERED_MODELS)
        models: list[str] = []
        try:
            import ollama  # type: ignore

            host = os.getenv("NOUSE_OLLAMA_HOST") or os.getenv("OLLAMA_HOST")
            client = ollama.Client(host=host) if host else ollama.Client()
            payload = client.list()
            rows = payload.get("models") if isinstance(payload, dict) else getattr(payload, "models", None)
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("model") or row.get("name") or "").strip()
                    if name:
                        models.append(name)
        except Exception:
            models = []
        _AUTO_DISCOVERED_MODELS = models
        return list(models)


def _split_model_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _extract_model_candidates(override_models: list[str] | None = None) -> list[str]:
    defaults: list[str] = []
    defaults.extend(override_models or [])
    defaults.extend(_split_model_list(EXTRACT_CANDIDATES_RAW))
    defaults.append(MODEL)
    if FALLBACK_MODEL:
        defaults.append(FALLBACK_MODEL)
    defaults.extend(_autodiscovered_models())
    defaults.extend(_split_model_list(GLOBAL_CANDIDATES_RAW))
    return resolve_model_candidates("extract", defaults)


def _synth_model_candidates() -> list[str]:
    defaults: list[str] = []
    defaults.extend(_split_model_list(SYNTH_CANDIDATES_RAW))
    defaults.append(SYNTH_MODEL)
    if SYNTH_FALLBACK_MODEL:
        defaults.append(SYNTH_FALLBACK_MODEL)
    defaults.extend(_autodiscovered_models())
    defaults.extend(_split_model_list(GLOBAL_CANDIDATES_RAW))
    return resolve_model_candidates("synthesize", defaults)


def _normalize_rel_type(raw: Any) -> str | None:
    rel = str(raw or "").strip().lower()
    if not rel:
        return None
    rel = _RELATION_NORMALIZATION.get(rel, rel)
    return rel if rel in _RELATION_TYPE_SET else None


def _clean_phrase(value: str, *, max_words: int = 4) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    cleaned = cleaned.strip(" ,.;:()[]{}\"'")
    if not cleaned:
        return ""
    words = cleaned.split(" ")
    for idx, word in enumerate(words):
        if idx >= 2 and word.lower() in _PHRASE_STOP_TOKENS:
            words = words[:idx]
            break
    if len(words) > max_words:
        words = words[:max_words]
    cleaned = " ".join(words).strip()
    return cleaned


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    saw_empty_array = False

    for idx, ch in enumerate(raw):
        if ch != "[":
            continue
        try:
            parsed, _end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        if not parsed:
            saw_empty_array = True
            continue
        out = [item for item in parsed if isinstance(item, dict)]
        if out:
            return out

    if saw_empty_array:
        return []
    return []


def _normalize_relations(
    rows: list[dict[str, Any]],
    *,
    domain_hint: str,
    max_items: int = EXTRACT_MAX_RELATIONS,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        src = _clean_phrase(str(row.get("src") or ""))
        tgt = _clean_phrase(str(row.get("tgt") or ""))
        rel = _normalize_rel_type(row.get("type") or row.get("rel_type"))
        if not src or not tgt or not rel or src.lower() == tgt.lower():
            continue
        key = (src.lower(), rel, tgt.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "src": src,
                "domain_src": _clean_phrase(str(row.get("domain_src") or domain_hint), max_words=6) or domain_hint,
                "type": rel,
                "tgt": tgt,
                "domain_tgt": _clean_phrase(str(row.get("domain_tgt") or domain_hint), max_words=6) or domain_hint,
                "why": _clean_phrase(str(row.get("why") or ""), max_words=26),
            }
        )
        if len(out) >= max_items:
            break
    return out


def _extraction_quality(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    n = float(len(rows))
    if n <= 0:
        return None
    why_ratio = sum(1 for r in rows if str(r.get("why") or "").strip()) / n
    cross_domain_ratio = (
        sum(
            1
            for r in rows
            if str(r.get("domain_src") or "").strip().lower()
            != str(r.get("domain_tgt") or "").strip().lower()
        )
        / n
    )
    rel_diversity = len({str(r.get("type") or "").strip() for r in rows if str(r.get("type") or "").strip()})
    rel_diversity_ratio = rel_diversity / max(1.0, min(float(len(RELATION_TYPES)), n))
    score = (0.5 * why_ratio) + (0.3 * cross_domain_ratio) + (0.2 * rel_diversity_ratio)
    return max(0.0, min(1.0, float(score)))


def _heuristic_extract_relations(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    domain_hint = str(metadata.get("domain_hint") or "okänd").strip() or "okänd"
    chunks = [c.strip() for c in re.split(r"[.!?\n;]+", text) if len(c.strip()) >= 10]
    if not chunks:
        return []

    rows: list[dict[str, Any]] = []
    for sentence in chunks[:120]:
        clauses = [c.strip() for c in re.split(r"\s*,\s*|\boch\b|\band\b", sentence, flags=re.IGNORECASE) if len(c.strip()) >= 8]
        for clause in clauses:
            for pattern, rel_type in _HEURISTIC_PATTERNS:
                for match in pattern.finditer(clause):
                    src = _clean_phrase(match.group("src"))
                    tgt = _clean_phrase(match.group("tgt"))
                    if not src or not tgt or src.lower() == tgt.lower():
                        continue
                    rows.append(
                        {
                            "src": src,
                            "domain_src": domain_hint,
                            "type": rel_type,
                            "tgt": tgt,
                            "domain_tgt": domain_hint,
                            "why": "Heuristisk extraktion från explicit textmönster.",
                        }
                    )
                    if len(rows) >= EXTRACT_MAX_RELATIONS:
                        break
                if len(rows) >= EXTRACT_MAX_RELATIONS:
                    break
            if len(rows) >= EXTRACT_MAX_RELATIONS:
                break
        if len(rows) >= EXTRACT_MAX_RELATIONS:
            break
    return _normalize_relations(rows, domain_hint=domain_hint, max_items=EXTRACT_MAX_RELATIONS)


async def _extract_with_model(
    *,
    model: str,
    chunk: str,
    domain_hint: str,
    session_id: str = "system",
    run_id: str | None = None,
    timeout_sec: float = EXTRACT_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    client = AsyncOllama(timeout_sec=timeout_sec)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Text:\n{chunk}"},
        ],
        b76_meta={
            "workload": "extract",
            "session_id": session_id,
            "run_id": run_id,
        },
    )
    raw = (resp.message.content or "").strip()
    rows = _extract_json_array(raw)
    return _normalize_relations(rows, domain_hint=domain_hint, max_items=EXTRACT_MAX_RELATIONS)


def _is_timeout_error(exc: Exception) -> bool:
    return "timeout" in str(exc).lower()


def _coerce_positive_float(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if value <= 0:
        return float(default)
    return float(value)


def _coerce_model_override(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


async def extract_relations_with_diagnostics(
    text: str,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Extrahera relationer + returnera diagnosmetadata för autonom styrning.
    """
    diagnostics: dict[str, Any] = {
        "source": str(metadata.get("source") or ""),
        "path": str(metadata.get("path") or ""),
        "chars": len(text or ""),
        "attempted_models": [],
        "model_errors": [],
        "timeouts": 0,
        "used_heuristic_fallback": False,
        "success_model": None,
    }

    if len(text.strip()) < 100:
        diagnostics["skipped_reason"] = "too_short"
        return [], diagnostics

    chunk = text[:EXTRACT_MAX_CHARS]
    domain_hint = str(metadata.get("domain_hint") or "okänd").strip() or "okänd"
    timeout_sec = _coerce_positive_float(
        metadata.get("extract_timeout_sec"),
        EXTRACT_TIMEOUT_SEC,
    )
    override_models = _coerce_model_override(metadata.get("extract_models"))
    diagnostics["extract_timeout_sec"] = timeout_sec
    if override_models:
        diagnostics["extract_models_override"] = list(override_models)

    if override_models:
        raw_candidates = _extract_model_candidates(override_models)
    else:
        raw_candidates = _extract_model_candidates()
    candidates = order_models_for_workload("extract", raw_candidates)

    for model in candidates:
        diagnostics["attempted_models"].append(model)
        try:
            rels = await _extract_with_model(
                model=model,
                chunk=chunk,
                domain_hint=domain_hint,
                session_id=str(metadata.get("session_id") or "system"),
                run_id=(str(metadata.get("run_id") or "").strip() or None),
                timeout_sec=timeout_sec,
            )
            if rels:
                diagnostics["success_model"] = model
                quality = _extraction_quality(rels)
                diagnostics["quality"] = quality
                record_model_result(
                    "extract",
                    model,
                    success=True,
                    timeout=False,
                    quality=quality,
                )
                return rels, diagnostics
            # Tomt svar är inte timeout, men räknas som misslyckat försök.
            record_model_result("extract", model, success=False, timeout=False)
        except Exception as e:
            timed_out = _is_timeout_error(e)
            diagnostics["model_errors"].append({"model": model, "error": str(e), "timeout": timed_out})
            if timed_out:
                diagnostics["timeouts"] = int(diagnostics.get("timeouts", 0) or 0) + 1
            record_model_result("extract", model, success=False, timeout=timed_out)
            _log.warning(f"Extraktion misslyckades (model={model}): {e}")

    if ENABLE_HEURISTIC_FALLBACK:
        heuristic = _heuristic_extract_relations(chunk, metadata)
        if heuristic:
            diagnostics["used_heuristic_fallback"] = True
            _log.info(
                "Extraktion fallback=heuristic gav %d relationer (source=%s)",
                len(heuristic),
                metadata.get("source", "unknown"),
            )
            return heuristic, diagnostics

    return [], diagnostics


async def extract_relations(text: str, metadata: dict) -> list[dict]:
    rels, _diag = await extract_relations_with_diagnostics(text, metadata)
    return rels


_SYNTHESIZE = """Du är ett kreativt kunskapssystem. Du har hittat en nervbana i en kunskapsgraf.
Uppgift: resonera kring denna bana och föreslå 2-4 ATOMÄRA nya relationer som fördjupar kopplingen.

Nervbanan ges som: koncept1 --[reltyp]--> koncept2 --[reltyp]--> ... --> konceptN

Returnera BARA giltig JSON-array i detta format:
[
  {"src":"konceptA","domain_src":"domän","rel_type":"reltyp",
   "tgt":"konceptB","domain_tgt":"domän","why":"varför denna koppling håller"}
]

Tillåtna relationstyper: är_analogt_med, stärker, är_del_av, skiljer_sig_från, möjliggör, beskriver, leder_till, modulerar, orsakar

Regler:
- Lägg BARA till relationer som inte redan finns i banan
- Fokusera på broar mellan OLIKA domäner
- Varje ny relation ska vara vetenskapligt försvarbar"""


async def synthesize_bridges(path: list[tuple], domain_a: str, domain_b: str,
                             lam: float = 0.5) -> list[dict]:
    """
    Be LLM om nya atomära bryggor längs en hittad nervbana.
    lam = λ (kreativitetskoefficient från limbic layer):
      hög λ → temperatur 0.8 (kreativt, tvärdomain)
      låg λ → temperatur 0.2 (konservativt, konsoliderar)
    """
    path_str    = " → ".join(f"{s} --[{r}]--> {t}" for s, r, t in path)
    prompt      = f"Nervbana: {path_str}\nDomän A: {domain_a}, Domän B: {domain_b}"
    temperature = 0.2 + lam * 0.6   # [0.2, 0.8]

    models = order_models_for_workload("synthesize", _synth_model_candidates())
    for model in models:
        try:
            client = AsyncOllama(timeout_sec=SYNTH_TIMEOUT_SEC)
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYNTHESIZE},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": temperature},
                b76_meta={
                    "workload": "synthesize",
                    "session_id": "autonomous",
                },
            )
            raw = (resp.message.content or "").strip()
            rows = _extract_json_array(raw)
            if not rows:
                record_model_result("synthesize", model, success=False, timeout=False)
                continue
            record_model_result("synthesize", model, success=True, timeout=False)
            return rows
        except Exception as e:
            timed_out = _is_timeout_error(e)
            record_model_result("synthesize", model, success=False, timeout=timed_out)
            _log.warning(f"Syntes misslyckades (model={model}): {e}")

    return []
