"""Provider batch job support.

``LLMCaller.create_batch()``/``acreate_batch()`` submit a set of requests to a
provider's asynchronous batch endpoint (OpenAI ``/v1/batches``, Anthropic
Message Batches, Gemini Batch API) and return an ``LLMBatch`` handle. Poll with
``status()``/``astatus()``, then collect ordered results with
``results()``/``aresults()``, or block with ``wait()``/``await_completion()``.
``LLMCaller.call_batch()``/``acall_batch()`` wrap the whole flow in one call.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from .constants import LLM_BATCH_POLL_INTERVAL
from .errors import LLMProviderError, provider_error
from .llmtypes import LLMBatchRequest, LLMBatchResult, LLMCaller
from .logging_utils import get_logger


class LLMBatchStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


_TERMINAL_STATUSES = frozenset(
    {
        LLMBatchStatus.COMPLETED,
        LLMBatchStatus.FAILED,
        LLMBatchStatus.CANCELLED,
        LLMBatchStatus.EXPIRED,
    }
)


class LLMBatchError(RuntimeError):
    """Results were requested before completion, or the job ended in a
    non-completed terminal state."""


class LLMBatch(ABC):
    """Handle to a submitted provider batch job.

    ``Requests`` maps ``Custom_Id`` to the submitted ``LLMBatchRequest``, so
    results can be re-ordered and formatted with the request's response schema.
    """

    provider: str | None = None

    def __init__(
        self,
        caller: LLMCaller,
        batch_id: str,
        requests: list[LLMBatchRequest],
        response_schema=None,
    ):
        self.Caller = caller
        self.Id = batch_id
        self.Requests = {req.Custom_Id: req for req in requests}
        self.Response_Schema = response_schema
        self.logger = get_logger(f"batch.{caller.Model.Name or 'unknown'}")

    @abstractmethod
    def status(self) -> LLMBatchStatus:
        """Current job status."""

    @abstractmethod
    async def astatus(self) -> LLMBatchStatus:
        """Async ``status``."""

    @abstractmethod
    def results(self) -> list[LLMBatchResult]:
        """Per-request results in submission order; requires a completed job."""

    @abstractmethod
    async def aresults(self) -> list[LLMBatchResult]:
        """Async ``results``."""

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation of the job."""

    @abstractmethod
    async def acancel(self) -> None:
        """Async ``cancel``."""

    def wait(
        self,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> list[LLMBatchResult]:
        """Poll until a terminal status; returns results on completion."""
        poll_interval = (
            LLM_BATCH_POLL_INTERVAL if poll_interval is None else poll_interval
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.status()
            if status is LLMBatchStatus.COMPLETED:
                return self.results()
            if status in _TERMINAL_STATUSES:
                raise LLMBatchError(
                    f"Batch {self.Id} ended with status '{status.value}'"
                )
            if deadline is not None and time.monotonic() + poll_interval > deadline:
                raise TimeoutError(
                    f"Batch {self.Id} still '{status.value}' after {timeout}s"
                )
            self.logger.debug("batch %s status %s", self.Id, status)
            time.sleep(poll_interval)

    async def await_completion(
        self,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> list[LLMBatchResult]:
        """Async ``wait``."""
        poll_interval = (
            LLM_BATCH_POLL_INTERVAL if poll_interval is None else poll_interval
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = await self.astatus()
            if status is LLMBatchStatus.COMPLETED:
                return await self.aresults()
            if status in _TERMINAL_STATUSES:
                raise LLMBatchError(
                    f"Batch {self.Id} ended with status '{status.value}'"
                )
            if deadline is not None and time.monotonic() + poll_interval > deadline:
                raise TimeoutError(
                    f"Batch {self.Id} still '{status.value}' after {timeout}s"
                )
            self.logger.debug("batch %s status %s", self.Id, status)
            await asyncio.sleep(poll_interval)

    def _raise_unless_completed(self, status: LLMBatchStatus) -> None:
        if status is not LLMBatchStatus.COMPLETED:
            raise LLMBatchError(
                f"Batch {self.Id} is '{status.value}', results are not available yet"
            )

    def _ordered(self, by_id: dict[str, LLMBatchResult]) -> list[LLMBatchResult]:
        out = []
        for custom_id in self.Requests:
            result = by_id.get(custom_id)
            if result is None:
                result = LLMBatchResult(
                    Custom_Id=custom_id,
                    Error=LLMProviderError(
                        "No result returned for this request",
                        provider=self.provider,
                        model=self.Caller.Model.Name,
                    ),
                )
            out.append(result)
        return out

    def _format_result(self, custom_id: str, output: Any) -> LLMBatchResult:
        try:
            message = self.Caller.format_output(
                output, response_schema=self.Response_Schema
            )
            return LLMBatchResult(Custom_Id=custom_id, Message=message)
        except Exception as e:
            return LLMBatchResult(Custom_Id=custom_id, Error=e)

    def save(self, path) -> None:
        """Write everything needed to re-attach to this job as JSON.

        Persist the model name, batch id, requests and response schema so a
        later process can recover the handle with ``load_batch(path)``.
        """
        Path(path).write_text(
            json.dumps(
                {
                    "provider": self.provider,
                    "model": self.Caller.Model.Name,
                    "batch_id": self.Id,
                    "requests": [
                        req.model_dump(mode="json")
                        for req in self.Requests.values()
                    ],
                    "response_schema": _schema_ref(self.Response_Schema),
                }
            )
        )


def _schema_ref(schema) -> str | None:
    if schema is None:
        return None
    return f"{schema.__module__}:{schema.__qualname__}"


def _resolve_schema(ref: str | None):
    if not ref:
        return None
    module_name, _, qualname = ref.partition(":")
    try:
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError) as e:
        get_logger("batch").warning(
            "could not resolve saved response schema %r (%s); "
            "results will be formatted without it",
            ref,
            e,
        )
        return None


def load_batch(path, caller: LLMCaller | None = None) -> LLMBatch:
    """Re-attach to a batch job saved with ``LLMBatch.save``.

    If ``caller`` is omitted a caller is constructed for the saved model
    name.
    """
    data = json.loads(Path(path).read_text())
    if caller is None:
        from .helpers import get_caller

        caller = get_caller(data["model"])
    requests = [LLMBatchRequest(**r) for r in data["requests"]]
    return caller.get_batch(
        data["batch_id"],
        requests,
        response_schema=_resolve_schema(data.get("response_schema")),
    )


_OPENAI_STATUSES = {
    "validating": LLMBatchStatus.PENDING,
    "in_progress": LLMBatchStatus.PENDING,
    "finalizing": LLMBatchStatus.PENDING,
    "cancelling": LLMBatchStatus.PENDING,
    "completed": LLMBatchStatus.COMPLETED,
    "failed": LLMBatchStatus.FAILED,
    "expired": LLMBatchStatus.EXPIRED,
    "cancelled": LLMBatchStatus.CANCELLED,
}


class OpenAIBatch(LLMBatch):
    """OpenAI ``/v1/batches`` job (also used for the completions endpoint)."""

    provider = "openai"

    def __init__(
        self, caller, batch_id, requests, output_parser, response_schema=None
    ):
        super().__init__(caller, batch_id, requests, response_schema)
        # output_parser validates a result line's body into the SDK response
        # type, e.g. openai.types.chat.ChatCompletion.model_validate
        self._output_parser = output_parser

    @staticmethod
    def _map_status(status) -> LLMBatchStatus:
        return _OPENAI_STATUSES.get(status, LLMBatchStatus.PENDING)

    def status(self) -> LLMBatchStatus:
        batch = self.Caller.Client.batches.retrieve(self.Id)
        return self._map_status(batch.status)

    async def astatus(self) -> LLMBatchStatus:
        batch = await self.Caller.AClient.batches.retrieve(self.Id)
        return self._map_status(batch.status)

    def _file_text(self, file_id: str) -> str:
        content = self.Caller.Client.files.content(file_id)
        return content.text

    async def _afile_text(self, file_id: str) -> str:
        content = await self.Caller.AClient.files.content(file_id)
        return content.text

    def _parse_output_lines(self, text: str) -> dict[str, LLMBatchResult]:
        by_id = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id is None:
                continue
            response = record.get("response") or {}
            status_code = response.get("status_code")
            if status_code == 200:
                try:
                    output = self._output_parser(response.get("body") or {})
                except Exception as e:
                    by_id[custom_id] = LLMBatchResult(
                        Custom_Id=custom_id, Error=e
                    )
                else:
                    by_id[custom_id] = self._format_result(custom_id, output)
            else:
                error = (response.get("body") or {}).get("error") or record.get(
                    "error"
                )
                message = (
                    error.get("message")
                    if isinstance(error, dict)
                    else str(error)
                )
                by_id[custom_id] = LLMBatchResult(
                    Custom_Id=custom_id,
                    Error=provider_error(
                        message or "Batch request failed",
                        code=status_code,
                        provider=self.provider,
                        model=self.Caller.Model.Name,
                    ),
                )
        return by_id

    def _collect(self, output_file_id, error_file_id) -> list[LLMBatchResult]:
        by_id = {}
        if output_file_id:
            by_id.update(
                self._parse_output_lines(self._file_text(output_file_id))
            )
        if error_file_id:
            for cid, res in self._parse_output_lines(
                self._file_text(error_file_id)
            ).items():
                by_id.setdefault(cid, res)
        return self._ordered(by_id)

    async def _acollect(self, output_file_id, error_file_id):
        by_id = {}
        if output_file_id:
            by_id.update(
                self._parse_output_lines(await self._afile_text(output_file_id))
            )
        if error_file_id:
            for cid, res in self._parse_output_lines(
                await self._afile_text(error_file_id)
            ).items():
                by_id.setdefault(cid, res)
        return self._ordered(by_id)

    def results(self) -> list[LLMBatchResult]:
        batch = self.Caller.Client.batches.retrieve(self.Id)
        self._raise_unless_completed(self._map_status(batch.status))
        return self._collect(batch.output_file_id, batch.error_file_id)

    async def aresults(self) -> list[LLMBatchResult]:
        batch = await self.Caller.AClient.batches.retrieve(self.Id)
        self._raise_unless_completed(self._map_status(batch.status))
        return await self._acollect(batch.output_file_id, batch.error_file_id)

    def cancel(self) -> None:
        self.Caller.Client.batches.cancel(self.Id)

    async def acancel(self) -> None:
        await self.Caller.AClient.batches.cancel(self.Id)


_ANTHROPIC_STATUSES = {
    "in_progress": LLMBatchStatus.PENDING,
    "canceling": LLMBatchStatus.PENDING,
    "ended": LLMBatchStatus.COMPLETED,
}


class AnthropicBatch(LLMBatch):
    """Anthropic Message Batches job."""

    provider = "anthropic"

    @property
    def _batches(self):
        return self.Caller.Client.beta.messages.batches

    @property
    def _abatches(self):
        return self.Caller.AClient.beta.messages.batches

    @staticmethod
    def _map_status(processing_status) -> LLMBatchStatus:
        return _ANTHROPIC_STATUSES.get(processing_status, LLMBatchStatus.PENDING)

    def status(self) -> LLMBatchStatus:
        batch = self._batches.retrieve(self.Id)
        return self._map_status(batch.processing_status)

    async def astatus(self) -> LLMBatchStatus:
        batch = await self._abatches.retrieve(self.Id)
        return self._map_status(batch.processing_status)

    def _result_entry(self, item) -> LLMBatchResult:
        custom_id, result = item.custom_id, item.result
        result_type = getattr(result, "type", None)
        if result_type == "succeeded":
            return self._format_result(custom_id, result.message)
        if result_type == "errored":
            error = getattr(result, "error", None)
            inner = getattr(error, "error", error)
            return LLMBatchResult(
                Custom_Id=custom_id,
                Error=LLMProviderError(
                    getattr(inner, "message", None) or "Batch request errored",
                    provider=self.provider,
                    model=self.Caller.Model.Name,
                    details={"type": getattr(inner, "type", None)},
                    retryable=False,
                ),
            )
        return LLMBatchResult(
            Custom_Id=custom_id,
            Error=LLMProviderError(
                f"Batch request {result_type}",
                provider=self.provider,
                model=self.Caller.Model.Name,
                retryable=result_type == "expired",
            ),
        )

    def results(self) -> list[LLMBatchResult]:
        self._raise_unless_completed(self.status())
        by_id = {}
        for item in self._batches.results(self.Id):
            by_id[item.custom_id] = self._result_entry(item)
        return self._ordered(by_id)

    async def aresults(self) -> list[LLMBatchResult]:
        self._raise_unless_completed(await self.astatus())
        by_id = {}
        async for item in await self._abatches.results(self.Id):
            by_id[item.custom_id] = self._result_entry(item)
        return self._ordered(by_id)

    def cancel(self) -> None:
        self._batches.cancel(self.Id)

    async def acancel(self) -> None:
        await self._abatches.cancel(self.Id)


def _google_status(state) -> LLMBatchStatus:
    name = getattr(state, "name", None) or str(state)
    if "SUCCEEDED" in name:
        return LLMBatchStatus.COMPLETED
    if "FAILED" in name:
        return LLMBatchStatus.FAILED
    if "CANCELLED" in name:
        return LLMBatchStatus.CANCELLED
    if "EXPIRED" in name:
        return LLMBatchStatus.EXPIRED
    return LLMBatchStatus.PENDING


class GoogleGenAIBatch(LLMBatch):
    """Gemini Batch API job (inline requests)."""

    provider = "google"

    def status(self) -> LLMBatchStatus:
        job = self.Caller.Client.batches.get(name=self.Id)
        return _google_status(job.state)

    async def astatus(self) -> LLMBatchStatus:
        job = await self.Caller.Client.aio.batches.get(name=self.Id)
        return _google_status(job.state)

    def _collect(self, dest) -> list[LLMBatchResult]:
        inlined = getattr(dest, "inlined_responses", None)
        if inlined is None:
            raise LLMBatchError(
                f"Batch {self.Id} wrote results to "
                f"'{getattr(dest, 'file_name', None) or 'external storage'}', "
                "which is not supported; submit with inlined requests"
            )
        by_id = {}
        custom_ids = list(self.Requests)
        for i, item in enumerate(inlined):
            key = (getattr(item, "metadata", None) or {}).get("key")
            if key in self.Requests:
                custom_id = key
            else:
                custom_id = custom_ids[i] if i < len(custom_ids) else f"req-{i}"
            error = getattr(item, "error", None)
            if error is not None:
                by_id[custom_id] = LLMBatchResult(
                    Custom_Id=custom_id,
                    Error=LLMProviderError(
                        getattr(error, "message", None)
                        or "Batch request failed",
                        provider=self.provider,
                        model=self.Caller.Model.Name,
                        details={"code": getattr(error, "code", None)},
                        retryable=False,
                    ),
                )
            else:
                by_id[custom_id] = self._format_result(
                    custom_id, item.response
                )
        return self._ordered(by_id)

    def results(self) -> list[LLMBatchResult]:
        job = self.Caller.Client.batches.get(name=self.Id)
        self._raise_unless_completed(_google_status(job.state))
        return self._collect(getattr(job, "dest", None))

    async def aresults(self) -> list[LLMBatchResult]:
        job = await self.Caller.Client.aio.batches.get(name=self.Id)
        self._raise_unless_completed(_google_status(job.state))
        return self._collect(getattr(job, "dest", None))

    def cancel(self) -> None:
        self.Caller.Client.batches.cancel(name=self.Id)

    async def acancel(self) -> None:
        await self.Caller.Client.aio.batches.cancel(name=self.Id)
