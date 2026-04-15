# Lab Note: Nervbana Axion Density & Sweet Spot Calibration

**Date**: 2026-04-15
**Context**: D3 goal-directed execution verified working in main loop

## Theoretical Background

The biological brain has spatial organization — different regions for different
data types, with distance between regions determining signal propagation delay
and concentration. fMRI comparisons of neurotypical vs autistic brains show
differences in axion density per white-matter tract (nervbana):

- **Surplus axions** → enhanced systematic analysis BUT also overstimulation + rigidity
- **Insufficient axions** → isolation, no cross-domain signal propagation
- **Sweet spot** → enough axions for bisociation without lockup

This maps to a Yerkes-Dodson curve for knowledge graphs:

```
Creativity (bisociation)
    ↑
    |      ╱╲  ← k_sweet (sweet_multiplier × ln(max(|A|,|B|)))
    |     ╱  ╲
    |    ╱    ╲
    |   ╱      ╲___ Rigid zone (overstimuli, lockup)
    |  ╱
    | ╱ Isolated zone (no signal propagation)
    └──────────────────→ axion_count per nervbana
```

## Implementation

Added to `daemon/percolation.py`:

1. **`NervbanaProfile`** — dataclass for each domain pair (nervbana):
   - axion_count (cross-domain edges between domain A and B)
   - k_min = ln(max(|A|,|B|)) — minimum for signal propagation
   - k_sweet = sweet_multiplier × k_min — optimal for bisociation
   - k_rigid = |A|×|B| / max(|A|,|B|) — rigidity threshold
   - zone: "isolated" | "sweet" | "rigid"

2. **`nervbana_profiles()`** — compute axion density for all domain pairs

3. **`domain_rigidity()`** — compute rigidity score per domain:
   - rigidity = mean(w) × (1 - mean(u)) for cross-domain edges
   - High rigidity = over-crystallized, signals lock up
   - Zone: "fluid" | "balanced" | "rigid"

4. **`sweet_spot_report()`** — unified report combining nervbana density
   and domain rigidity

5. **`format_sweet_spot_report()`** — human-readable output

## Results (Live Graph)

```
═══ SWEET SPOT REPORT ═══
Nervbanor: 200 domain pairs
  Isolerade: 200  |  Sweet: 0  |  Rigida: 0
  Sweet spot fraction: 0.0%
  Avg axion density: 0.0008  |  Avg rigidity: 0.250
```

**Interpretation**: ALL nervbanor are in the isolated zone. Each has ~1 axion
vs. k_min ≈ 8.6 and k_sweet ≈ 21.6. Nous needs ~8× more cross-domain edges
per nervbana just for signal propagation, and ~22× for bisociation.

The "programmering" domain is a hub with 14+ thin spokes (1 axion each) —
a star topology, not a network. This is "slagsida" (lopsidedness): the system
can reason about code but not about cognition.

## Structural Localization Insight

The user's key insight: without spatial organization, adding edges
indiscriminately creates hub-and-spoke topologies. The biological brain
avoids this through wire-length optimization — frequently communicating
regions are placed close together, creating natural clustering.

For Nous, this suggests:
- Domains should have spatial coordinates (2D or 3D)
- Wire-length optimization should guide domain placement
- Signal delay = f(distance) should emerge from geometry, not be fixed
- Bridge domains should be placed BETWEEN the domains they bridge
  (like association cortex between sensory and motor areas)

This prevents "slagsida" — no single domain becomes a hub with thin spokes,
because spatial organization distributes connections according to functional
proximity, not just statistical frequency.

## Next Steps

1. Integrate sweet_spot_report into daemon D3 section (every Nth cycle)
2. Use nervbana isolation scores to prioritize goal-directed curiosity
3. Implement spatial embedding for domains (wire-length optimization)
4. Calibrate signal decay rate (λ) based on axion density per nervbana
5. Write FNC formalization: "Sweet Spot Principle" — bisociation emerges
   at k_sweet axions per nervbana, not from scaling volume

## FNC Connection

The Sweet Spot Principle is a formalization of the autism insight:
the same structural property (axion density) that enables systematic
analysis also creates rigidity. Creativity (bisociation) exists in a
narrow zone between isolation and rigidity — it is not a monotonic
function of connection count.

This connects to:
- F_bisoc = prediction_error + λ × complexity_blend (threshold 0.45)
- Percolation threshold: ln(n) connections per domain
- Yerkes-Dodson: arousal curve for limbic system
- Crystalization: w > 0.55 AND u < 0.35 → permanent

The sweet spot is where F_bisoc > 0.45 AND rigidity < 0.3 AND
isolation < 0.2 — the intersection of percolation, flexibility, and
connectivity.