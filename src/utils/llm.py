"""Shared Ollama LLM client."""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, OLLAMA_HOST


@lru_cache(maxsize=4)
def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        num_ctx=LLM_NUM_CTX,
    )


@lru_cache(maxsize=4)
def get_llm_json(temperature: float = LLM_TEMPERATURE) -> ChatOllama:
    """Return a ChatOllama client that forces valid JSON output."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        num_ctx=LLM_NUM_CTX,
        format="json",
    )


def complete(prompt: str, temperature: float = LLM_TEMPERATURE) -> str:
    llm = get_llm(temperature)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def complete_json(prompt: str, temperature: float = LLM_TEMPERATURE) -> str:
    """Run a prompt with JSON mode forced at the decoding level."""
    llm = get_llm_json(temperature)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
