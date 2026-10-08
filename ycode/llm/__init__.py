"""LLM providers."""

from __future__ import annotations

from ycode.config import PROVIDERS, Config, resolve_local_model
from ycode.errors import ConfigError, MissingCredentialsError
from ycode.llm.base import LLMProvider

__all__ = ["PROVIDERS", "create_provider", "LLMProvider"]


def create_provider(config: Config, *, check_credentials: bool = True) -> LLMProvider:
    """Build the provider named in the config. Add new providers here."""
    if config.provider == "anthropic":
        try:
            from ycode.llm.anthropic import AnthropicProvider, has_credentials
        except ImportError:
            raise ConfigError(
                "Claude support is not installed. YCode's own model needs nothing extra: run `ycode` "
                "without --provider anthropic.",
                hint='To use Claude anyway: pip install -e ".[claude]"',
            ) from None

        if check_credentials and not has_credentials():
            raise MissingCredentialsError(
                "No Anthropic API key found (ANTHROPIC_API_KEY is not set). Run without --provider anthropic "
                "to use YCode's own model, which needs no key.")
        return AnthropicProvider(
            config.model,
            max_tokens=config.max_tokens,
            effort=config.effort,
            refusal_fallback=config.refusal_fallback,
        )
    if config.provider == "local":
        from ycode.llm.local import LocalProvider

        model_dir = resolve_local_model(config.local_model)
        return LocalProvider(model_dir, max_new_tokens=config.local_max_tokens,
                             temperature=config.local_temperature)
    raise ConfigError(f"Unknown provider {config.provider!r}. Available: {', '.join(PROVIDERS)}")
