"""Configuration settings for Context Guard reverse proxy."""

import os

from pydantic import BaseModel, Field, field_validator


class ProxyConfig(BaseModel):
    """Proxy server configuration options."""

    UPSTREAM_BASE_URL: str = Field(
        default_factory=lambda: (
            (
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("UPSTREAM_BASE_URL")
                or "https://api.groq.com/openai/v1"
            )
            .strip()
            .rstrip("/")
        ),
        validate_default=True,
    )
    UPSTREAM_API_KEY: str | None = Field(
        default_factory=lambda: (
            os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("UPSTREAM_API_KEY")
        )
    )
    DEFAULT_MODEL: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")
    )
    OPENROUTER_HTTP_REFERER: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("HTTP_REFERER")
    )
    OPENROUTER_SITE_TITLE: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_TITLE") or os.getenv("X_TITLE")
    )
    CONTEXT_GUARD_HOST: str = Field(
        default_factory=lambda: os.getenv("CONTEXT_GUARD_HOST", "0.0.0.0")
    )
    CONTEXT_GUARD_PORT: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_GUARD_PORT", "8080"))
    )
    PRESERVE_RECENT_TURNS: int = Field(
        default_factory=lambda: int(os.getenv("PRESERVE_RECENT_TURNS", "3"))
    )
    AUTO_COMPRESS_YELLOW: bool = Field(
        default_factory=lambda: (
            os.getenv("AUTO_COMPRESS_YELLOW", "true").lower() in ("true", "1", "yes")
        )
    )
    INTERVENTION_ON_RED: bool = Field(
        default_factory=lambda: (
            os.getenv("INTERVENTION_ON_RED", "true").lower() in ("true", "1", "yes")
        )
    )

    @field_validator("UPSTREAM_BASE_URL")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """Ensure upstream base URL does not have trailing slashes or whitespace."""
        return v.strip().rstrip("/")
