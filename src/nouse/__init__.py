"""
nouse — The Cognitive Substrate Framework for Model-Agnostic AI.

The missing link to AGI: a persistent, plastic brain layer that gives any LLM
the cognitive architecture of the human mind.

Memory architecture: working → episodic → semantic → procedural
Core innovation: SQLite WAL + NetworkX knowledge graph + Residual Streams (w, r, u) per edge.
Plasticity: STDP + Hebbian learning, NightRun consolidation, DeepDive axiom-discovery.

Quick start (high-level kernel API):
    import nouse
    k = nouse.Kernel()
    k.upsert_edge("e1", src="a", rel_type="causes", tgt="b", w=0.3, r=0.0, u=0.6)
    k.step()

Quick start (knowledge graph API):
    from nouse.field.surface import FieldSurface
    field = FieldSurface()
    field.add_relation("ocean_current", "influences", "climate", why="heat transport")
"""
from nouse.config.env import load_env_files as _load_env_files

_load_env_files()

from nouse.kernel import (
    Brain as Kernel,
    FieldEvent,
    NeuromodulatorState,
    NodeStateSpace,
    ResidualEdge,
    MEMORY_TIERS,
    NEUROMODULATORS,
    SCHEMA_VERSION,
)

from nouse.inject import attach, NouseBrain, Axiom, ConceptProfile, QueryResult, ContradictionResult
from nouse.llm.wrapper import (
    DEFAULT_SYSTEM_PREAMBLE,
    WrappedLLMResponse,
    build_system_prompt,
    extract_response_text,
    run_with_nouse,
)
from nouse.search.escalator import EscalationResult

__version__ = "0.4.0"

__all__ = [
    # Inject API — one-line entry point
    "attach",
    "NouseBrain",
    "Axiom",
    "ConceptProfile",
    "QueryResult",
    "ContradictionResult",
    "DEFAULT_SYSTEM_PREAMBLE",
    "WrappedLLMResponse",
    "build_system_prompt",
    "extract_response_text",
    "run_with_nouse",
    # Escalation
    "EscalationResult",
    # Residual Stream kernel
    "Kernel",
    "FieldEvent",
    "NeuromodulatorState",
    "NodeStateSpace",
    "ResidualEdge",
    "MEMORY_TIERS",
    "NEUROMODULATORS",
    "SCHEMA_VERSION",
]
