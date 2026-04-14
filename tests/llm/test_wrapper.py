from __future__ import annotations

from types import SimpleNamespace

from nouse.inject import Axiom, ConceptProfile, QueryResult
from nouse.llm.wrapper import (
    DEFAULT_SYSTEM_PREAMBLE,
    build_system_prompt,
    extract_response_text,
    run_with_nouse,
)


def _memory() -> QueryResult:
    return QueryResult(
        query="test query",
        concepts=[
            ConceptProfile(
                name="Nous",
                summary="Structured epistemic memory for LLMs.",
                claims=["Nous is the brain layer."],
                evidence_refs=[],
                related_terms=["memory", "epistemics"],
                uncertainty=0.1,
                revision_count=1,
                axioms=[],
            )
        ],
        axioms=[
            Axiom(
                src="Nous",
                rel="IS",
                tgt="brain layer",
                evidence=0.84,
                flagged=False,
            )
        ],
        confidence=0.84,
        domains=["ai"],
        has_knowledge=True,
    )


class _FakeBrain:
    def __init__(self) -> None:
        self.learn_calls: list[dict] = []

    def query(self, question: str, top_k: int = 6) -> QueryResult:
        return _memory()

    def learn(self, prompt: str, response: str, **kwargs) -> None:
        self.learn_calls.append(
            {"prompt": prompt, "response": response, "kwargs": kwargs}
        )


def test_build_system_prompt_includes_context_and_metadata():
    prompt, memory = build_system_prompt(
        "What is Nous?",
        brain=_FakeBrain(),
    )

    assert DEFAULT_SYSTEM_PREAMBLE.strip() in prompt
    assert "[Nous memory]" in prompt
    assert "[Nous meta]" in prompt
    assert "confidence=0.84" in prompt
    assert memory.has_knowledge is True


def test_run_with_nouse_calls_model_and_learns():
    brain = _FakeBrain()

    def _call_model(*, system_prompt: str, user_prompt: str, memory: QueryResult):
        assert "[Nous memory]" in system_prompt
        assert user_prompt == "Explain Nous"
        assert memory.confidence == 0.84
        return "Nous is the brain layer for grounding."

    result = run_with_nouse(
        "Explain Nous",
        _call_model,
        brain=brain,
        source="test-wrapper",
        model="test-model",
    )

    assert result.answer == "Nous is the brain layer for grounding."
    assert len(brain.learn_calls) == 1
    assert brain.learn_calls[0]["kwargs"]["source"] == "test-wrapper"
    assert brain.learn_calls[0]["kwargs"]["model"] == "test-model"


def test_extract_response_text_supports_openai_like_shape():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Grounded answer")
            )
        ]
    )

    assert extract_response_text(response) == "Grounded answer"
