"""Concrete LLM adapters for extracting outbreak evidence fields."""

from .argo import ArgoClient, argo_extract, argo_llm_callable

__all__ = [
    "ArgoClient",
    "argo_extract",
    "argo_llm_callable",
]
