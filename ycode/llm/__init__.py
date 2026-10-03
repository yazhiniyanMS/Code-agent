"""LLM providers."""

from __future__ import annotations

from ycode.config import Config
from ycode.errors import ConfigError, MissingCredentialsError
from ycode.llm.base import LLMProvider

PROVIDERS = ("anthropic",)


def create_provider(config: Config, *, check_credentials: bool = True) -> LLMProvider:
    """Build the provider named in the config. Add new providers here."""
    if config.provider == "anthropic":
        from ycode.llm.anthropic import AnthropicProvider, has_credentials

        if check_credentials and not has_credentials():
            raise MissingCredentialsError("No Anthropic API key found (ANTHROPIC_API_KEY is not set).")
        return AnthropicProvider(
            config.model,
            max_tokens=config.max_tokens,
            effort=config.effort,
            refusal_fallback=config.refusal_fallback,
        )
    raise ConfigError(f"Unknown provider {config.provider!r}. Available: {', '.join(PROVIDERS)}")
