"""
b76.tda.bridge — Python-brygga till Rust TDA-motorn
=====================================================
Beräknar Betti-nummer (H0, H1) och topologisk similaritet
via den kompilerade Rust-modulen tda_engine.

Fabrikslogik:
  1. Försöker importera tda_engine (Rust, snabb)
  2. Fallback: Python-implementation (scipy, långsam men fungerar)

Bygga Rust-modulen:
  cd /home/bjorn/projects/nouse/crates/tda_engine
  maturin develop --release -m /home/bjorn/projects/nouse/crates/tda_engine/Cargo.toml

Eller via install.sh (görs automatiskt).
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

log = logging.getLogger("nouse.tda")

# Försök Rust-modulen
_RUST_AVAILABLE = False
try:
    import tda_engine as _tda  # type: ignore[import]
    _RUST_AVAILABLE = True
    log.info("TDA: Rust-motor aktiv (tda_engine)")
except ImportError:
    log.warning("TDA: Rust-motor saknas — faller tillbaka till Python. "
                "Kör: cd crates/tda_engine && maturin develop --release")


# ── Publikt API ──────────────────────────────────────────────────────────────

def compute_distance_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    """Beräkna euklidisk distansmatris från embeddings."""
    if _RUST_AVAILABLE:
        return _tda.compute_distance_matrix(embeddings)
    return _py_distance_matrix(embeddings)


def compute_betti(
    dist_matrix: list[list[float]],
    max_epsilon: float = 2.0,
    steps: int = 50,
) -> tuple[int, int]:
    """
    Beräkna Betti-nummer H0 och H1.
    
    H0 = antal sammanhängande komponenter
    H1 = antal oberoende cykler
    
    Returerar (h0, h1).
    """
    if _RUST_AVAILABLE:
        return _tda.compute_betti(dist_matrix, max_epsilon, steps)
    return _py_betti(dist_matrix, max_epsilon)


def topological_similarity(
    h0_a: int, h1_a: int,
    h0_b: int, h1_b: int,
) -> float:
    """
    Topologisk similaritet [0.0, 1.0] baserat på Betti-profiler.
    Hög τ + låg semantisk likhet → potentiell bisociation!
    """
    if _RUST_AVAILABLE:
        return _tda.topological_similarity(h0_a, h1_a, h0_b, h1_b)
    return _py_topological_similarity(h0_a, h1_a, h0_b, h1_b)


def is_rust_active() -> bool:
    """Returnerar True om Rust TDA-motorn är laddad."""
    return _RUST_AVAILABLE


def identify_knowledge_gaps(
    field: Any,
    domain: str,
    max_gaps: int = 10,
    min_degree: int = 2,
) -> list[tuple[str, str, float]]:
    """
    Identifierar kunskapsluckor i en domän.

    H1-cykler i persistenshomologi representerar konceptpar som är
    indirekt kopplade (via grannar) men saknar direkt relation — dvs.
    topologiska hål i kunskapsstrukturen.

    Approximation (utan fullständig persistent homologi):
      1. Bygg grannskapsgraf från domänkoncepten
      2. Hitta par (A, B) som delar EN gemensam granne men har INGEN direkt kant
      3. Ranka efter gap_score = (delat_grannskap / max_grannskap) × (1 - direkt_evidens)

    Args:
        field:      FieldSurface-instans
        domain:     Domän att analysera
        max_gaps:   Max antal gap-par att returnera
        min_degree: Minimalt antal relationer en nod behöver för att räknas

    Returns:
        Lista av (src, tgt, gap_score) sorterad efter gap_score descending.
        gap_score ∈ [0, 1]: hög = starkt strukturellt motiverat att undersöka
    """
    try:
        concepts = field.concepts(domain=domain)
    except Exception:
        return []

    if len(concepts) < 3:
        return []

    # Bygg adjacency: {concept_name: set(grannar)}
    adjacency: dict[str, set[str]] = {}
    for c in concepts:
        name = c["name"] if isinstance(c, dict) else str(c)
        adjacency[name] = set()

    # Fyll grannar från befintliga relationer
    try:
        for c in concepts:
            name = c["name"] if isinstance(c, dict) else str(c)
            rels = field.relations(src=name, domain=domain)
            for r in rels:
                tgt = r.get("target") or r.get("tgt") or ""
                if tgt and tgt in adjacency:
                    adjacency[name].add(tgt)
                    adjacency[tgt].add(name)
    except Exception:
        return []

    # Filtrera noder med för få relationer
    active = {n for n, nbrs in adjacency.items() if len(nbrs) >= min_degree}
    if len(active) < 2:
        return []

    gaps: list[tuple[str, str, float]] = []
    active_list = sorted(active)

    for i, a in enumerate(active_list):
        for b in active_list[i + 1:]:
            if b in adjacency[a]:
                continue  # direkt koppling finns redan

            # Delat grannskap
            shared = adjacency[a] & adjacency[b]
            if not shared:
                continue  # inte ens indirekt kopplade

            max_degree = max(len(adjacency[a]), len(adjacency[b]))
            gap_score = len(shared) / max(1, max_degree)
            gaps.append((a, b, round(gap_score, 3)))

    gaps.sort(key=lambda x: x[2], reverse=True)
    return gaps[:max_gaps]


# ── Python-fallback (numpy/scipy) ────────────────────────────────────────────

def _py_distance_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    try:
        import numpy as np
        from scipy.spatial.distance import cdist
        # Filtrera inhomogena embeddings — behåll bara dominant dimension
        if embeddings:
            dims = [len(e) for e in embeddings]
            modal_dim = max(set(dims), key=dims.count)
            embeddings = [e for e in embeddings if len(e) == modal_dim]
        if len(embeddings) < 2:
            return [[0.0]]
        arr = np.array(embeddings, dtype=np.float32)
        return cdist(arr, arr, metric="euclidean").tolist()
    except ImportError:
        pass
    # Ren Python-nödlösning om scipy saknas
    n = len(embeddings)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(embeddings[i], embeddings[j])))
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _py_betti(dist_matrix: list[list[float]], max_epsilon: float) -> tuple[int, int]:
    try:
        import numpy as np
        dm = np.asarray(dist_matrix, dtype=np.float32)
        n = len(dm)
        if n < 2:
            return (n, 0)
        rows, cols = np.triu_indices(n, 1)
        dists = dm[rows, cols]
        order = np.argsort(dists, kind="mergesort")
        sorted_u = rows[order]
        sorted_v = cols[order]
        sorted_d = dists[order]
        cutoff = int(np.searchsorted(sorted_d, max_epsilon, side="right"))
        sorted_u = sorted_u[:cutoff].tolist()
        sorted_v = sorted_v[:cutoff].tolist()
    except ImportError:
        # Ren Python-nödlösning
        n = len(dist_matrix)
        if n < 2:
            return (n, 0)
        edges = sorted(
            [(dist_matrix[i][j], i, j) for i in range(n) for j in range(i + 1, n)],
        )
        sorted_u = [e[1] for e in edges if e[0] <= max_epsilon]
        sorted_v = [e[2] for e in edges if e[0] <= max_epsilon]

    parent = list(range(n))
    rank   = [0] * n
    h0 = n
    h1 = 0

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(sorted_u, sorted_v):
        pu, pv = find(u), find(v)
        if pu == pv:
            h1 += 1
        else:
            if rank[pu] < rank[pv]:
                parent[pu] = pv
            elif rank[pu] > rank[pv]:
                parent[pv] = pu
            else:
                parent[pv] = pu
                rank[pu] += 1
            h0 = max(1, h0 - 1)

    return (h0, h1)


def _py_topological_similarity(h0_a: int, h1_a: int, h0_b: int, h1_b: int) -> float:
    dh0 = abs(h0_a - h0_b)
    dh1 = abs(h1_a - h1_b)
    max_h0 = max(h0_a, h0_b)
    max_h1 = max(h1_a, h1_b)
    norm_h0 = 1.0 - dh0 / max_h0 if max_h0 > 0 else 1.0
    norm_h1 = 1.0 - dh1 / max_h1 if max_h1 > 0 else 1.0
    return max(0.0, min(1.0, 0.35 * norm_h0 + 0.65 * norm_h1))
