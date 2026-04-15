"""
nouse.daemon.vision — Visual cortex for embodied environment understanding
=============================================================================

Gives Nous "eyes" — the ability to process visual information and develop
spatial understanding of its environment.  Maps to the Occipital Lobe in
the brain atlas.

Capabilities:
1. Image description — extract concepts from images/diagrams
2. Spatial reasoning — understand spatial relationships in visuals
3. Graph visualization — "see" its own knowledge graph topology
4. Document figure extraction — process figures from PDFs/papers
5. Environment awareness — understand file structures, UI, etc.

Backends:
- Local: Ollama llava (7b) — fast, private, no API cost
- Cloud: Gemini API — higher quality, spatial reasoning (Gemini Robotics ER)
- Fallback: heuristic description from metadata (no vision model)

Integration with brain atlas:
- Processed visual concepts → occipital_lobe region
- Spatial relationships → parietal_lobe region
- Vision-driven goals → frontal_lobe region
"""
from __future__ import annotations

import base64
import logging
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("nouse.vision")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISION_MODEL = os.getenv("NOUSE_VISION_MODEL", "llava:7b")
GEMINI_API_KEY = os.getenv("NOUSE_GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("NOUSE_GEMINI_VISION_MODEL", "gemini-2.0-flash")
OLLAMA_BASE = os.getenv("NOUSE_OLLAMA_BASE", "http://127.0.0.1:11434")

#: Timeout for vision model calls (seconds)
VISION_TIMEOUT = int(os.getenv("NOUSE_VISION_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VisionResult:
    """Result from processing an image."""
    description: str           # text description of the image
    concepts: list[str]        # concepts identified in the image
    spatial_info: dict         # spatial relationships detected
    source: str                # "llava" | "gemini" | "heuristic"
    confidence: float          # 0-1, how confident the description is


# ---------------------------------------------------------------------------
# Ollama llava backend
# ---------------------------------------------------------------------------

def _ollama_describe(image_path: str, prompt: str = "") -> dict | None:
    """Describe an image using local Ollama llava model."""
    import urllib.request

    if not prompt:
        prompt = (
            "Describe this image in detail. Identify all concepts, objects, "
            "relationships, and spatial arrangements. If this is a diagram or "
            "graph, describe the structure and connections. "
            "Respond in JSON: {\"description\": \"...\", \"concepts\": [...], "
            "\"spatial_info\": {\"layout\": \"...\", \"relationships\": [...]}}"
        )

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_data],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    url = f"{OLLAMA_BASE}/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result
    except Exception as e:
        log.warning(f"Ollama vision failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Gemini API backend
# ---------------------------------------------------------------------------

def _gemini_describe(image_path: str, prompt: str = "") -> dict | None:
    """Describe an image using Gemini API (Gemini Robotics ER spatial reasoning)."""
    if not GEMINI_API_KEY:
        log.debug("No Gemini API key configured, skipping cloud vision")
        return None

    import urllib.request

    if not prompt:
        prompt = (
            "Describe this image with focus on: 1) All concepts and objects "
            "visible, 2) Spatial relationships between elements, 3) If this "
            "is a diagram, describe the topology and connections. "
            "4) Any structural patterns. Respond in JSON format."
        )

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine MIME type from extension
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_data}},
            ]
        }],
        "generationConfig": {"temperature": 0.3},
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            # Extract text from Gemini response
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        return {"response": part["text"]}
            return None
    except Exception as e:
        log.warning(f"Gemini vision failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback (no vision model)
# ---------------------------------------------------------------------------

def _heuristic_describe(image_path: str) -> dict:
    """Heuristic image description from file metadata (no vision model)."""
    path = Path(image_path)
    stat = path.stat()

    concepts = []
    # Extract concepts from filename
    name = path.stem.replace("-", " ").replace("_", " ")
    concepts.extend(name.split())

    # Extract concepts from parent directory
    parent = path.parent.name.replace("-", " ").replace("_", " ")
    if parent and parent != ".":
        concepts.extend(parent.split())

    # Size hints
    size_kb = stat.st_size / 1024
    if size_kb > 500:
        concepts.append("high-resolution")
    elif size_kb > 100:
        concepts.append("detailed")
    else:
        concepts.append("simple")

    # Extension hints
    ext = path.suffix.lower()
    if ext in (".svg",):
        concepts.append("vector-graphics")
        concepts.append("diagram")
    elif ext in (".png", ".jpg", ".jpeg"):
        concepts.append("raster-graphics")

    return {
        "response": json.dumps({
            "description": f"Image file: {path.name} ({size_kb:.0f}KB)",
            "concepts": concepts,
            "spatial_info": {"layout": "unknown", "relationships": []},
        }),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_image(image_path: str, prompt: str = "") -> VisionResult:
    """Describe an image using available vision backends.

    Tries: Ollama llava → Gemini API → heuristic fallback.
    """
    # Try local llava first
    result = _ollama_describe(image_path, prompt)
    source = "llava"

    # Fall back to Gemini
    if not result or not result.get("response"):
        result = _gemini_describe(image_path, prompt)
        source = "gemini"

    # Fall back to heuristic
    if not result or not result.get("response"):
        result = _heuristic_describe(image_path)
        source = "heuristic"

    response_text = result.get("response", "") if result else ""

    # Clean markdown code fences that vision models sometimes add
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json, ```, etc.)
        first_newline = cleaned.find("\n")
        if first_newline > 0:
            cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    # Parse JSON from response
    concepts: list[str] = []
    spatial_info: dict = {}
    confidence = 0.5

    try:
        # Try to parse as JSON
        parsed = json.loads(cleaned)
        description = parsed.get("description", response_text)
        concepts = parsed.get("concepts", [])
        spatial_info = parsed.get("spatial_info", {})
        confidence = 0.8 if source != "heuristic" else 0.3
    except (json.JSONDecodeError, TypeError):
        # Not JSON — extract concepts from text
        description = response_text
        # Simple concept extraction: split on commas and "and"
        words = response_text.replace(".", ",").replace(";", ",").split(",")
        for w in words:
            w = w.strip()
            if 3 < len(w) < 40:
                concepts.append(w.lower())
        concepts = concepts[:10]  # limit
        confidence = 0.6 if source != "heuristic" else 0.2

    return VisionResult(
        description=description[:500],
        concepts=concepts[:20],
        spatial_info=spatial_info,
        source=source,
        confidence=confidence,
    )


def process_directory_images(
    directory: str,
    max_images: int = 10,
) -> list[VisionResult]:
    """Process all images in a directory.

    Finds image files and describes them, extracting concepts
    and spatial information.  Useful for processing research
    papers with figures.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    image_files = []

    dir_path = Path(directory)
    if dir_path.is_dir():
        for f in sorted(dir_path.rglob("*")):
            if f.suffix.lower() in image_extensions:
                image_files.append(str(f))

    # Sort by modification time (newest first)
    image_files.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)

    results: list[VisionResult] = []
    for img_path in image_files[:max_images]:
        log.info(f"Processing image: {img_path}")
        try:
            result = describe_image(img_path)
            results.append(result)
            log.info(f"  → {result.source}: {len(result.concepts)} concepts, "
                     f"confidence={result.confidence:.2f}")
        except Exception as e:
            log.warning(f"  Failed: {e}")

    return results


def graph_spatial_embedding(field: Any, prompt: str = "") -> dict[str, tuple[float, float]]:
    """Generate 2D spatial positions for domains using vision model.

    Creates a visual representation of the knowledge graph topology
    and uses a vision model to suggest spatial placement.  This is
    Nous "seeing" its own brain structure.

    Returns: dict of domain_name → (x, y) positions.
    """
    import tempfile

    # Generate a simple text representation of domain relationships
    domains = field.domains()
    if not domains:
        return {}

    # Use domain sizes and cross-domain connections for placement
    positions: dict[str, tuple[float, float]] = {}

    # Simple force-directed placement as fallback
    # (vision model integration would improve this)
    n = len(domains)
    sorted_domains = sorted(domains)

    # Place domains in a circle initially
    for i, domain in enumerate(sorted_domains):
        angle = 2 * math.pi * i / max(1, n)
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        positions[domain] = (round(x, 2), round(y, 2))

    # If a vision model is available, we could:
    # 1. Render the graph to an image
    # 2. Ask the vision model to suggest better placement
    # 3. Parse the response for coordinates
    # This is a TODO — force-directed layout works for now

    log.info(f"Spatial embedding: {len(positions)} domains positioned")
    return positions


# Need math import for spatial embedding
import math