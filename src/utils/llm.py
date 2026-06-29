"""Shared Ollama LLM client."""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, OLLAMA_HOST


@lru_cache(maxsize=4)
def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatOllama:
    """Return a cached ChatOllama client at the requested temperature."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        num_ctx=LLM_NUM_CTX,
    )


def complete(prompt: str, temperature: float = LLM_TEMPERATURE) -> str:
    """Run a single prompt through the LLM and return its text response."""
    llm = get_llm(temperature)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
