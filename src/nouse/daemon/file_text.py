"""
Filtext-extraktion för b76-källor.
Stödjer vanlig text samt PDF (via pypdf).
"""
from __future__ import annotations

import logging
from pathlib import Path


def extract_text(path: Path) -> str:
    """
    Extrahera text från fil.
    Returnerar tom sträng vid fel.
    """
    try:
        if path.suffix.lower() == ".pdf":
            if not _looks_like_pdf(path):
                # Filer kan vara felmärkta som .pdf men innehålla ren text.
                return _read_text(path)
            return _extract_pdf_text(path)
        return _read_text(path)
    except Exception:
        return ""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except Exception:
        return False
    return head.startswith(b"%PDF-")


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        # Utan pypdf: returnera tom sträng istället för brus från binärdata.
        return ""

    try:
        # Felaktiga PDF-filer kan trigga verbose parser-varningar.
        pypdf_log = logging.getLogger("pypdf")
        previous_level = pypdf_log.level
        if previous_level < logging.ERROR:
            pypdf_log.setLevel(logging.ERROR)
        try:
            reader = PdfReader(str(path), strict=False)
        finally:
            if previous_level < logging.ERROR:
                pypdf_log.setLevel(previous_level)

        parts: list[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt:
                parts.append(txt)
        return "\n\n".join(parts)
    except Exception:
        return ""
