from __future__ import annotations

"""
Canonical field exports for NoUse.

`FieldSurface` is the active SQLite WAL + NetworkX backend.
Kuzu-based field code remains legacy and should not be used for new NoUseAI work.
"""

from nouse.field.surface import FieldSurface

__all__ = ["FieldSurface"]
