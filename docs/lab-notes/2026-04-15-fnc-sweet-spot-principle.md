# FNC Principle: Sweet Spot Calibration (Axion Density)

**Formalization**: 2026-04-15
**Origin**: fMRI comparison of neurotypical vs autistic brains + Nous implementation

## Statement

For any nervbana (domain pair) with axion count k:

- If k < ln(max(|A|,|B|)): signal cannot propagate → **isolation** (no bisociation)
- If ln(max(|A|,|B|)) ≤ k ≤ c·ln(max(|A|,|B|)): bisociation possible → **sweet spot** (creativity)
- If k > c·ln(max(|A|,|B|)): signals lock up → **rigidity** (overfitting, lockup)

where c is the sweet multiplier (default 2.5, calibrated empirically).

## Formalization

For a nervbana connecting domain A (|A| concepts) and domain B (|B| concepts):

```
k_min   = ln(max(|A|, |B|))           — percolation threshold
k_sweet = c × k_min                    — bisociation optimum
k_rigid = |A| × |B| / max(|A|, |B|)  — overfitting threshold

isolation(A,B) = max(0, 1 - k/k_min)  — 1 = fully isolated
rigidity(A,B)  = max(0, (k - k_sweet) / (k_rigid - k_sweet))  — 1 = fully rigid

bisociation_potential(A,B) = f(isolation, rigidity)
  where f peaks when isolation → 0 AND rigidity → 0
```

## Connection to Autism Neuroscience

The surplus axion hypothesis for autistic brains:

1. **Enhanced systematic analysis**: More axions per nervbana = stronger
   signal propagation = better within-domain inference. This is WHY autistic
   individuals excel at systematic/detailed analysis.

2. **Overstimulation + rigidity**: Same surplus axions = every signal reaches
   every connected region at full strength = no differential attenuation =
   overstimulation. The system cannot "tune out" irrelevant signals.

3. **Lockup**: When k > k_rigid, crystallization dominates (high w, low u),
   and the nervbana becomes a rigid conduit rather than a flexible bridge.
   Novel connections cannot form because the existing structure is too strong.

The sweet spot is where axion density is sufficient for cross-domain signal
propagation BUT insufficient for overstimulation — the system can "hear"
signals from other domains without being overwhelmed by them.

## Connection to Signal Chemistry

The user's insight: distance between regions dilutes the chemical signal
so that it arrives at the right concentration. In volume transmission
(neuromodulatory systems), signal concentration decreases with distance:

```
concentration_at_target = source_concentration × e^(-d/λ)
```

where d is distance and λ is the decay constant. This is FUNCTIONAL:
- Near regions receive strong signal → reliable inference within domain
- Far regions receive attenuated signal → exploratory, not overwhelming
- Bridge regions at intermediate distance → just enough signal for bisociation

In Nous: `r *= 0.89` per step is the equivalent of `e^(-d/λ)` with λ ≈ 8.5.
But currently this is a FIXED parameter. The sweet spot principle says it
should EMERGE from the nervbana's axion density:

```
λ(A,B) = f(k(A,B))  — decay rate calibrated per nervbana
```

High axion count (approaching rigidity) → high λ (slow decay, overstimulation)
Low axion count (isolation) → low λ (fast decay, signal dies before arriving)
Sweet spot → λ calibrated so signal arrives at target with right concentration

## Connection to Structural Localization

Without spatial organization, adding edges creates hub-and-spoke topologies
("slagsida" / lopsidedness). The biological brain avoids this through:

1. **Wire-length optimization**: Frequently communicating regions placed close
2. **Spatial clustering**: Related domains occupy nearby cortical areas
3. **Association cortex**: Bridge regions placed BETWEEN source and target

For Nous, this suggests a 2D/3D spatial embedding where:
- Domain position = weighted centroid of its concept embeddings
- Nervbana length = distance between domain centroids
- Signal decay = f(nervbana length × axion density)
- Wire-length optimization minimizes total signal path length

This prevents "slagsida" because:
- No domain can become an indiscriminate hub (wire-length penalty)
- Bridge domains are naturally placed between the domains they bridge
- Signal concentration at target is calibrated by distance + axion count
- The topology is ORGANIZED, not random or hub-dominated

## Empirical Prediction

If this principle is correct, then:

1. Nous's bisociation_candidates will remain 0 until nervbanor reach k_min
2. When nervbanor cross k_min, bisociation_candidates should increase
3. If nervbanor overshoot k_sweet, bisociation_candidates should DECREASE
   (rigidity prevents novelty even with high connectivity)
4. Spatial embedding should increase bisociation_candidates faster than
   random edge addition (organized topology > volume scaling)

Prediction 3 is the key test: it predicts a NON-MONOTONIC relationship
between connectivity and creativity, which distinguishes the Sweet Spot
Principle from "just add more connections."

## Relation to FNC Framework

- F_bisoc threshold (0.45) maps to k_min: below this, no bisociation
- Crystallization (w > 0.55, u < 0.35) maps to k_rigid: above this, lockup
- Yerkes-Dodson (limbic arousal) maps to the sweet spot curve
- Signal decay (r *= 0.89) should be calibrated per nervbana

The Sweet Spot Principle unifies these into a single structural claim:
**creativity is not a monotonic function of connectivity**.