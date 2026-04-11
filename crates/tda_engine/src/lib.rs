/// tda_engine — Rust TDA motor för NoUse
///
/// Implementerar exakt samma algoritmer som Python-fallbacken i bridge.py men
/// ~10–50× snabbare beroende på grafstorlek.
///
/// Exponerade funktioner (matchar bridge.py publikt API):
///   compute_distance_matrix(embeddings) -> Vec<Vec<f64>>
///   compute_betti(dist_matrix, max_epsilon, steps) -> (usize, usize)
///   topological_similarity(h0_a, h1_a, h0_b, h1_b) -> f64
use pyo3::prelude::*;
use rayon::prelude::*;

// ── Union-Find (path-komprimering + rank-union) ───────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank:   Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank:   vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path compression
            x = self.parent[x];
        }
        x
    }

    /// Returnerar true om de var i samma komponent (dvs. detta skapar en cykel → H1++).
    fn union(&mut self, a: usize, b: usize) -> bool {
        let pa = self.find(a);
        let pb = self.find(b);
        if pa == pb {
            return true; // cykel
        }
        match self.rank[pa].cmp(&self.rank[pb]) {
            std::cmp::Ordering::Less    => self.parent[pa] = pb,
            std::cmp::Ordering::Greater => self.parent[pb] = pa,
            std::cmp::Ordering::Equal   => {
                self.parent[pb] = pa;
                self.rank[pa] += 1;
            }
        }
        false
    }
}

// ── compute_distance_matrix ───────────────────────────────────────────────────

/// Beräkna euklidisk distansmatris från en lista av embedding-vektorer.
///
/// Args:
///     embeddings: Vec av vektorer med samma dimension
///
/// Returns:
///     n×n distansmatris som Vec<Vec<f64>>
#[pyfunction]
fn compute_distance_matrix(embeddings: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = embeddings.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![0.0]];
    }

    // Parallell rad-beräkning via rayon
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0f64; n];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d: f64 = embeddings[i]
                    .iter()
                    .zip(embeddings[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                row[j] = d;
            }
            row
        })
        .collect()
}

// ── compute_betti ─────────────────────────────────────────────────────────────

/// Beräkna Betti-nummer H0 och H1 via inkrementell union-find (Vietoris-Rips).
///
/// Algoritm:
///   1. Sortera alla kantpar (i,j) efter distans
///   2. Bygg grafen incrementellt upp till max_epsilon
///   3. Union-Find: ny edge → union(i,j)
///      - Pu != Pv → H0-- (sammanfogar komponenter)
///      - Pu == Pv → H1++ (skapar cykel)
///
/// Args:
///     dist_matrix: n×n distansmatris
///     max_epsilon: cutoff-avstånd för Vietoris-Rips-komplexet
///     steps:       ignoreras (Python-API-kompatibilitet, default 50)
///
/// Returns:
///     (h0, h1) som (usize, usize)
#[pyfunction]
#[pyo3(signature = (dist_matrix, max_epsilon = 2.0, steps = 50))]
fn compute_betti(
    dist_matrix: Vec<Vec<f64>>,
    max_epsilon: f64,
    steps: usize,
) -> (usize, usize) {
    let _ = steps; // API-kompatibilitet — används ej i union-find-implementationen
    let n = dist_matrix.len();
    if n == 0 {
        return (0, 0);
    }
    if n == 1 {
        return (1, 0);
    }
    let _ = steps;

    // Samla och sortera kanter
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_matrix[i][j];
            if d <= max_epsilon {
                edges.push((d, i, j));
            }
        }
    }
    // Stabil sort på distans
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(n);
    let mut h0 = n;
    let mut h1: usize = 0;

    for (_, u, v) in &edges {
        if uf.union(*u, *v) {
            h1 += 1;
        } else {
            h0 = h0.saturating_sub(1).max(1);
        }
    }

    (h0, h1)
}

// ── topological_similarity ────────────────────────────────────────────────────

/// Topologisk similaritet [0.0, 1.0] baserat på Betti-profiler.
///
/// Samma formel som Python-fallbacken:
///   τ = 0.35 × norm_H0 + 0.65 × norm_H1
///
/// Hög τ + låg semantisk likhet → potentiell bisociation.
#[pyfunction]
fn topological_similarity(
    h0_a: usize,
    h1_a: usize,
    h0_b: usize,
    h1_b: usize,
) -> f64 {
    let dh0 = (h0_a as f64 - h0_b as f64).abs();
    let dh1 = (h1_a as f64 - h1_b as f64).abs();
    let max_h0 = h0_a.max(h0_b) as f64;
    let max_h1 = h1_a.max(h1_b) as f64;

    let norm_h0 = if max_h0 > 0.0 { 1.0 - dh0 / max_h0 } else { 1.0 };
    let norm_h1 = if max_h1 > 0.0 { 1.0 - dh1 / max_h1 } else { 1.0 };

    (0.35 * norm_h0 + 0.65 * norm_h1).clamp(0.0, 1.0)
}

// ── PyO3-modul ────────────────────────────────────────────────────────────────

#[pymodule]
fn tda_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_betti, m)?)?;
    m.add_function(wrap_pyfunction!(topological_similarity, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
