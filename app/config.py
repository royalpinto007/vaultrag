"""Settings, read from the environment. No secrets in code, ever."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://vaultrag:vaultrag@localhost:5433/vaultrag"
    )
    embedder: str = os.getenv("EMBEDDER", "local")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm: str = os.getenv("LLM", "groq")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_settings() -> Settings:
    return Settings()
