"""Typed errors raised when a provider response cannot be turned into an LLMMessage.

Messages and metadata only carry fields that are safe to log (provider, model,
response id, stop reason, block types, token counts), never prompt or completion
text.
"""

from enum import Enum
from typing import Any, Iterable, Mapping, Optional

# Stop reasons meaning the completion budget ran out: resending the same request
# is expected to fail the same way, so these responses are not retried.
LENGTH_STOP_REASONS = frozenset({"max_tokens", "length", "MAX_TOKENS"})

# Stop reasons meaning the provider withheld the output (refusal, safety filter,
# guardrail). Retrying the same request is not expected to help.
CONTENT_FILTER_STOP_REASONS = frozenset(
    {
        # Anthropic
        "refusal",
        # OpenAI / Mistral
        "content_filter",
        # Google
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        # Bedrock
        "guardrail_intervened",
        "content_filtered",
    }
)


def _reason_str(reason: Any) -> Optional[str]:
    if reason is None:
        return None
    if isinstance(reason, Enum):
        reason = reason.value
    return str(reason)


def safe_usage(usage: Any, *fields: str) -> Optional[dict]:
    """Keep only the integer token counts of a provider usage object or dict."""
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        values = {f: usage.get(f) for f in fields}
    else:
        values = {f: getattr(usage, f, None) for f in fields}
    out = {k: v for k, v in values.items() if isinstance(v, int)}
    return out or None


class LLMResponseError(ValueError):
    """Base error for a provider response that could not be formatted.

    Subclasses ValueError so that existing ``except ValueError`` handlers keep
    working. ``retryable`` tells the caller's retry loop whether sending the same
    request again may succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        response_id: Optional[str] = None,
        stop_reason: Any = None,
        block_types: Optional[Iterable[str]] = None,
        usage: Optional[dict] = None,
        details: Optional[Mapping[str, Any]] = None,
        retryable: Optional[bool] = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.response_id = response_id
        self.stop_reason = _reason_str(stop_reason)
        self.block_types = list(block_types) if block_types is not None else None
        self.usage = usage
        self.details = (
            {k: _reason_str(v) for k, v in details.items() if v is not None}
            if details
            else None
        )
        self.retryable = self._default_retryable() if retryable is None else retryable

    def _default_retryable(self) -> bool:
        return False

    @property
    def metadata(self) -> dict:
        """Safe-to-log description of the response, without any generated text."""
        fields = dict(
            provider=self.provider,
            model=self.model,
            response_id=self.response_id,
            stop_reason=self.stop_reason,
            block_types=self.block_types,
            usage=self.usage,
            details=self.details,
        )
        out = {k: v for k, v in fields.items() if v is not None}
        out["retryable"] = self.retryable
        return out

    def __str__(self) -> str:
        meta = ", ".join(f"{k}={v!r}" for k, v in self.metadata.items())
        return f"{self.message} ({meta})"


class InvalidProviderResponse(LLMResponseError):
    """The response does not have a shape FDLLM knows how to format."""


class EmptyLLMResponse(LLMResponseError):
    """The response is well formed but contains nothing usable.

    Retryable unless the model stopped because it ran out of tokens.
    """

    def _default_retryable(self) -> bool:
        return self.stop_reason not in LENGTH_STOP_REASONS


class LLMContentFiltered(EmptyLLMResponse):
    """The provider withheld the output (refusal, safety filter, guardrail)."""

    def _default_retryable(self) -> bool:
        return False


def empty_response_error(message: str, **kwargs) -> EmptyLLMResponse:
    """Build an EmptyLLMResponse, or LLMContentFiltered if the stop reason says so."""
    stop_reason = _reason_str(kwargs.get("stop_reason"))
    if stop_reason in CONTENT_FILTER_STOP_REASONS:
        return LLMContentFiltered(message, **kwargs)
    return EmptyLLMResponse(message, **kwargs)


def is_retryable_response_error(exc: BaseException) -> bool:
    return isinstance(exc, LLMResponseError) and exc.retryable
