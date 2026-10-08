"""Exception hierarchy for YCode.

Every error a user can reasonably cause (bad config, missing key, network
trouble, ...) derives from :class:`YCodeError` so the CLI can print a clean
message instead of a traceback.
"""

from __future__ import annotations


class YCodeError(Exception):
    """Base class for all expected, user-facing YCode errors."""

    hint: str | None = None

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint


class ConfigError(YCodeError):
    """Invalid or missing configuration."""


class MissingCredentialsError(ConfigError):
    """No API credentials were found."""

    hint = (
        "Only needed for Claude (--provider anthropic): set ANTHROPIC_API_KEY in your environment, "
        "in a .env file in the project directory, or in ~/.ycode/.env. Without it, run plain `ycode` "
        "to use YCode's own model."
    )


# ---------------------------------------------------------------- LLM errors


class LLMError(YCodeError):
    """A failure talking to the language model provider."""

    retryable: bool = False


class LLMAuthenticationError(LLMError):
    hint = "Check that ANTHROPIC_API_KEY is set to a valid key."


class LLMPermissionError(LLMError):
    hint = "Your API key does not have access to this model or feature."


class LLMNotFoundError(LLMError):
    hint = "Check the model name (YCODE_MODEL / `model` in config.toml) or use /model."


class LLMRateLimitError(LLMError):
    retryable = True
    hint = "You are being rate limited. Wait a moment and try again."


class LLMConnectionError(LLMError):
    retryable = True
    hint = "Check your network connection (and any proxy settings) and try again."


class LLMServerError(LLMError):
    retryable = True
    hint = "The API returned a server error. This is usually temporary; try again."


class LLMBadRequestError(LLMError):
    """The provider rejected the request (400)."""

    hint = (
        "Check the model and settings in your config. If it persists, /clear starts a "
        "fresh conversation."
    )


class LLMResponseError(LLMError):
    """The provider returned something we could not interpret."""

    retryable = True
    hint = "Try the request again."


# --------------------------------------------------------------- tool errors


class ToolError(YCodeError):
    """A tool failed in a way that should be reported back to the model."""


class PathSecurityError(ToolError):
    """A path resolved outside the workspace."""


class PermissionDenied(ToolError):
    """The user (or the policy) declined an action."""
