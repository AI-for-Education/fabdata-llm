"""When LLMCaller.call/acall re-send a failed request, and how long they wait.

fdllm's retry loop is the only retry layer: the OpenAI and Anthropic SDK clients
are created with ``max_retries=0`` unless ``Client_Args`` says otherwise, so a
request is sent at most ``LLM_DEFAULT_MAX_RETRIES`` times. Exception: with aiohttp
installed, google-genai re-sends async requests once on a connection error, which
is not configurable short of swapping its HTTP client.

HTTP errors are judged by status code, the same rule the OpenAI SDK and
LiteLLM use (see ``errors.is_retryable_status``), because the SDK exception
classes do not separate transient from permanent errors (e.g. Google raises
``ClientError`` for both 400 and 429).
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import anthropic
import httpx
import openai
from google.genai import errors as genai_errors

from .errors import LLMResponseError, is_retryable_status

try:  # google-genai sends async requests through aiohttp when it is installed
    import aiohttp

    _AIOHTTP_ERRORS: tuple[type[BaseException], ...] = (aiohttp.ClientConnectionError,)
except ImportError:  # pragma: no cover
    _AIOHTTP_ERRORS = ()

# Longest Retry-After we wait for; longer values fall back to backoff (as in the OpenAI SDK)
MAX_RETRY_AFTER_SECONDS = 60

# No response was received: the connection failed, was reset or timed out.
# Narrow on purpose: httpx's ProxyError / UnsupportedProtocol are configuration errors.
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,  # includes APITimeoutError
    anthropic.APIConnectionError,  # includes APITimeoutError
    httpx.NetworkError,  # ConnectError, ReadError, WriteError, CloseError
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    *_AIOHTTP_ERRORS,
)

# Anthropic error events sent inside an already open (HTTP 200) stream
_RETRYABLE_ANTHROPIC_STREAM_ERRORS = frozenset({"overloaded_error", "api_error"})


def _status_code(exc: BaseException) -> int | None:
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return exc.status_code
    if isinstance(exc, genai_errors.APIError):
        return exc.code
    return None


def _headers(exc: BaseException):
    return getattr(getattr(exc, "response", None), "headers", None)


def should_retry(exc: BaseException) -> bool:
    """Whether re-sending the same request may succeed."""
    # fdllm's own response errors (empty, filtered, provider-reported, ...) decide for themselves
    if isinstance(exc, LLMResponseError):
        return exc.retryable
    if isinstance(exc, CONNECTION_ERRORS):
        return True
    status = _status_code(exc)
    if status is None:
        # anything else (Mistral and Bedrock SDK errors, bugs) is not retried
        return False
    # the server can say explicitly; the SDKs obey this header, so we do too
    headers = _headers(exc)
    should_retry_header = headers.get("x-should-retry") if headers is not None else None
    if should_retry_header == "true":
        return True
    if should_retry_header == "false":
        return False
    if status == 200:
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        error_type = error.get("type") if isinstance(error, dict) else None
        return error_type in _RETRYABLE_ANTHROPIC_STREAM_ERRORS
    if getattr(exc, "code", None) == "insufficient_quota":
        # OpenAI sends "out of credits" as a 429
        return False
    return is_retryable_status(status)


def retry_after_seconds(exc: BaseException) -> float | None:
    """The server's Retry-After (or retry-after-ms) in seconds, if within the cap."""
    headers = _headers(exc)
    if headers is None:
        return None
    try:
        if (ms := headers.get("retry-after-ms")) is not None:
            seconds = float(ms) / 1000
        elif (value := headers.get("retry-after")) is not None:
            try:
                seconds = float(value)
            except ValueError:  # HTTP-date form
                seconds = (
                    parsedate_to_datetime(value) - datetime.now(timezone.utc)
                ).total_seconds()
        else:
            return None
    except (TypeError, ValueError):
        return None
    return seconds if 0 < seconds <= MAX_RETRY_AFTER_SECONDS else None


def wait_retry_after(fallback):
    """Tenacity wait: the server's Retry-After if it gave one, else ``fallback``."""

    def wait(retry_state) -> float:
        seconds = retry_after_seconds(retry_state.outcome.exception())
        return seconds if seconds is not None else fallback(retry_state)

    return wait


def record_attempts(exc: BaseException, attempts: int) -> None:
    """Attach how many requests were sent, and whether the retry budget ran out.

    ``fdllm_retries_exhausted`` is True when the final error was itself retryable,
    i.e. the call gave up only because it ran out of attempts.
    """
    try:
        exc.fdllm_attempts = attempts
        exc.fdllm_retries_exhausted = attempts > 0 and should_retry(exc)
    except AttributeError:  # exception types that forbid new attributes
        pass
