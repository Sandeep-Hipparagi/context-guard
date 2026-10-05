"""Configuration settings for Context Guard reverse proxy."""

import os

from pydantic import BaseModel, Field, field_validator


class ProxyConfig(BaseModel):
    """Proxy server configuration options."""

    UPSTREAM_BASE_URL: str = Field(
        default_factory=lambda: os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    )
    UPSTREAM_API_KEY: str | None = Field(default_factory=lambda: os.getenv("UPSTREAM_API_KEY"))
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
        """Ensure upstream base URL does not have trailing slashes."""
        return v.rstrip("/")
