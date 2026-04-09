"""Reusable NoUse-first wrapper helpers for any model call."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Callable

from nouse.inject import QueryResult


DEFAULT_SYSTEM_PREAMBLE = """You are wrapped by NoUse, a persistent epistemic brain layer.

Always read the NoUse memory block before answering.
Use validated relations as primary grounding.
Call out uncertainty, missing knowledge, and weak evidence explicitly.
Do not present unsupported claims as if NoUse had validated them.
"""


@dataclass
class WrappedLLMResponse:
    """Result from a NoUse-wrapped model call."""

    user_prompt: str
    system_prompt: str
    answer: str
    memory: QueryResult
    raw_response: Any = None


def build_system_prompt(
    user_prompt: str,
    *,
    brain: Any | None = None,
    top_k: int = 6,
    max_axioms: int = 15,
    preamble: str = DEFAULT_SYSTEM_PREAMBLE,
    include_metadata: bool = True,
) -> tuple[str, QueryResult]:
    """Build a NoUse-first system prompt for a user query."""
    if brain is None:
        import nouse

        brain = nouse.attach()

    memory = brain.query(user_prompt, top_k=top_k)
    context_block = memory.context_block(max_axioms=max_axioms).strip()

    if not context_block:
        context_block = (
            "[Nouse memory]\n"
            "No grounded memory was found for this query. "
            "Answer carefully and make uncertainty explicit."
        )

    parts = [preamble.strip(), context_block]
    if include_metadata:
        parts.append(_format_memory_metadata(memory))

    return "\n\n".join(part for part in parts if part), memory


def run_with_nouse(
    user_prompt: str,
    call_model: Callable[..., Any],
    *,
    brain: Any | None = None,
    top_k: int = 6,
    max_axioms: int = 15,
    preamble: str = DEFAULT_SYSTEM_PREAMBLE,
    include_metadata: bool = True,
    learn: bool = True,
    source: str = "nouse-wrapper",
    model: str | None = None,
) -> WrappedLLMResponse:
    """Run a model call through NoUse grounding, then learn from the answer."""
    system_prompt, memory = build_system_prompt(
        user_prompt,
        brain=brain,
        top_k=top_k,
        max_axioms=max_axioms,
        preamble=preamble,
        include_metadata=include_metadata,
    )

    raw_response = _call_model(
        call_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        memory=memory,
    )
    answer = extract_response_text(raw_response)

    if learn:
        active_brain = brain
        if active_brain is None:
            import nouse

            active_brain = nouse.attach()
        active_brain.learn(
            user_prompt,
            answer,
            source=source,
            model=model,
            context_block=memory.context_block(max_axioms=max_axioms),
            confidence_in=memory.confidence,
            nodes_used=[concept.name for concept in memory.concepts],
        )

    return WrappedLLMResponse(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        answer=answer,
        memory=memory,
        raw_response=raw_response,
    )


def extract_response_text(response: Any) -> str:
    """Extract text from common model response shapes."""
    if isinstance(response, str):
        return response

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                return content
        text = getattr(first, "text", None)
        if isinstance(text, str) and text:
            return text

    content = getattr(response, "content", None)
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    if isinstance(response, dict):
        for key in ("output_text", "content", "text"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            if isinstance(first, dict):
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                text = first.get("text")
                if isinstance(text, str) and text:
                    return text

    raise TypeError("Could not extract text from model response")


def _call_model(
    call_model: Callable[..., Any],
    *,
    system_prompt: str,
    user_prompt: str,
    memory: QueryResult,
) -> Any:
    params = signature(call_model).parameters
    accepts_kwargs = any(
        param.kind == Parameter.VAR_KEYWORD for param in params.values()
    )

    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "system_prompt" in params:
        kwargs["system_prompt"] = system_prompt
    if accepts_kwargs or "user_prompt" in params:
        kwargs["user_prompt"] = user_prompt
    if accepts_kwargs or "memory" in params:
        kwargs["memory"] = memory
    if kwargs:
        return call_model(**kwargs)

    positional_arity = sum(
        param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for param in params.values()
    )
    if positional_arity >= 3:
        return call_model(system_prompt, user_prompt, memory)
    if positional_arity == 2:
        return call_model(system_prompt, user_prompt)
    if positional_arity == 1:
        return call_model(user_prompt)
    return call_model()


def _format_memory_metadata(memory: QueryResult) -> str:
    domains = ", ".join(memory.domains) if memory.domains else "unknown"
    return (
        "[NoUse meta]\n"
        f"confidence={memory.confidence:.2f}\n"
        f"axioms={len(memory.axioms)}\n"
        f"domains={domains}\n"
        f"has_knowledge={str(memory.has_knowledge).lower()}"
    )
