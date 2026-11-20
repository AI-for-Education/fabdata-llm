"""Tests for provider batch mode (create_batch/call_batch/get_batch,
check_batch_support) for OpenAI, Anthropic and Google GenAI callers,
all with mocked SDK clients.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fdllm import (
    BatchSupport,
    ClaudeCaller,
    GoogleGenAICaller,
    LLMBatchError,
    LLMBatchRequest,
    LLMBatchResult,
    LLMBatchStatus,
    OpenAICaller,
    load_batch,
)
from fdllm.errors import LLMProviderError
from fdllm.llmtypes import (
    LLMCallArgNames,
    LLMCaller,
    LLMMessage,
    OpenAIModelType,
)

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_CLAUDE_MODEL = "claude-test"
DEFAULT_GEMINI_MODEL = "gemini-test"


class FakeCaller(LLMCaller):
    """LLMCaller whose Func/AFunc echo the first message's content."""

    def __init__(self):
        def func(**kwargs):
            return kwargs["messages"][0]["content"]

        async def afunc(**kwargs):
            return kwargs["messages"][0]["content"]

        super().__init__(
            Model=OpenAIModelType(Name=DEFAULT_OPENAI_MODEL),
            Func=func,
            AFunc=afunc,
            Arg_Names=LLMCallArgNames(
                Model="model", Messages="messages", Max_Tokens="max_tokens"
            ),
            Token_Window=100000,
        )

    def format_message(self, message):
        return {"role": message.Role, "content": message.Message}

    def format_messagelist(self, messagelist):
        return [self.format_message(m) for m in messagelist]

    def format_output(self, output, response_schema=None, latency=None):
        return LLMMessage(Role="assistant", Message=output)

    def tokenize(self, messagelist):
        return [0] * 10


def _msgs(n, prefix="m"):
    return [LLMMessage(Role="user", Message=f"{prefix}{i}") for i in range(n)]


# ===== LLMBatchRequest coercion =====


def test_coerce_batch_requests_assigns_ids():
    caller = FakeCaller()
    reqs = caller._coerce_batch_requests(
        [
            LLMMessage(Role="user", Message="one"),
            [LLMMessage(Role="user", Message="two")],
            LLMBatchRequest(
                Messages=[LLMMessage(Role="user", Message="three")],
                Custom_Id="my-id",
            ),
        ]
    )
    assert [r.Custom_Id for r in reqs] == ["req-0", "req-1", "my-id"]
    # the caller-supplied request object is not mutated
    assert all(len(r.Messages) == 1 for r in reqs)


def test_llmbatch_request_accepts_single_message():
    req = LLMBatchRequest(Messages=LLMMessage(Role="user", Message="hi"))
    assert req.Messages == [LLMMessage(Role="user", Message="hi")]


# ===== create_batch on an unsupported caller =====


def test_create_batch_not_implemented():
    caller = FakeCaller()
    with pytest.raises(NotImplementedError, match="batch"):
        caller.create_batch(_msgs(1))


# ===== check_batch_support live probe =====


def _status_error(status):
    import openai

    return openai.APIStatusError(
        f"status {status}",
        response=httpx.Response(
            status, request=httpx.Request("GET", "http://probe")
        ),
        body=None,
    )


def test_check_batch_support_probe():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    caller.Client = MagicMock()

    caller.Client.batches.list.return_value = object()
    assert caller.check_batch_support() is BatchSupport.SUPPORTED
    caller.Client.batches.list.assert_called_once_with(limit=1)


def test_check_batch_support_classifies_errors():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    caller.Client = MagicMock()

    # provider has no /batches route
    caller.Client.batches.list.side_effect = _status_error(404)
    assert caller.check_batch_support() is BatchSupport.UNSUPPORTED

    # route exists but key lacks permission: still counts as supported
    caller.Client.batches.list.side_effect = _status_error(403)
    assert caller.check_batch_support() is BatchSupport.SUPPORTED

    # transient/inconclusive
    caller.Client.batches.list.side_effect = _status_error(503)
    assert caller.check_batch_support() is BatchSupport.UNKNOWN

    caller.Client.batches.list.side_effect = ConnectionError("reset")
    assert caller.check_batch_support() is BatchSupport.UNKNOWN


def test_check_batch_support_unsupported_caller():
    caller = FakeCaller()
    assert caller.check_batch_support() is BatchSupport.UNSUPPORTED


def test_acheck_batch_support_probe():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    caller.AClient = MagicMock()
    caller.AClient.batches.list = AsyncMock(return_value=object())
    assert (
        asyncio.run(caller.acheck_batch_support()) is BatchSupport.SUPPORTED
    )


def test_anthropic_check_batch_support_probe():
    caller = ClaudeCaller(DEFAULT_CLAUDE_MODEL)
    caller.Client = MagicMock()
    caller.Client.beta.messages.batches.list.return_value = object()
    assert caller.check_batch_support() is BatchSupport.SUPPORTED
    caller.Client.beta.messages.batches.list.assert_called_once_with(limit=1)


def test_google_check_batch_support_probe():
    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    assert caller.check_batch_support() is BatchSupport.SUPPORTED
    caller.Client.batches.list.assert_called_once()


# ===== OpenAI provider batch =====


def _openai_completion_body(text="hello", model=DEFAULT_OPENAI_MODEL):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }


def _mock_openai_client(caller):
    caller.Client = MagicMock()
    return caller.Client


def test_openai_create_batch_uploads_jsonl_and_creates_batch():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")

    batch = caller.create_batch(
        [
            LLMMessage(Role="user", Message="one"),
            LLMMessage(Role="user", Message="two"),
        ],
        max_tokens=50,
        temperature=0.2,
    )

    # input file upload
    file_args, file_kwargs = client.files.create.call_args
    name, payload, content_type = file_kwargs["file"]
    assert file_kwargs["purpose"] == "batch"
    lines = payload.decode("utf-8").splitlines()
    assert len(lines) == 2
    rec0, rec1 = (json.loads(l) for l in lines)
    assert rec0["custom_id"] == "req-0"
    assert rec0["url"] == "/v1/chat/completions"
    assert rec0["body"]["model"] == caller.Model.Api_Model_Name
    # call-level settings are shared by every request
    assert rec0["body"]["max_completion_tokens"] == 50
    assert rec0["body"]["temperature"] == 0.2
    assert rec1["custom_id"] == "req-1"
    assert rec1["body"]["max_completion_tokens"] == 50

    # batch create
    _, create_kwargs = client.batches.create.call_args
    assert create_kwargs["input_file_id"] == "file-1"
    assert create_kwargs["endpoint"] == "/v1/chat/completions"
    assert create_kwargs["completion_window"] == "24h"

    assert batch.Id == "batch_1"
    assert set(batch.Requests) == {"req-0", "req-1"}


def test_openai_batch_status_and_results():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")

    batch = caller.create_batch(_msgs(2))

    client.batches.retrieve.return_value = SimpleNamespace(
        status="in_progress", output_file_id=None, error_file_id=None
    )
    assert batch.status() is LLMBatchStatus.PENDING

    output_text = "\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": _openai_completion_body(text=f"resp-{cid}"),
                },
            }
        )
        for cid in ("req-0", "req-1")
    )
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id=None
    )
    client.files.content.return_value = SimpleNamespace(text=output_text)

    results = batch.results()
    assert [r.Custom_Id for r in results] == ["req-0", "req-1"]
    assert [r.Message.Message for r in results] == ["resp-req-0", "resp-req-1"]
    assert results[0].Message.TokensUsed == 3


def test_openai_batch_request_errors_are_results_not_raises():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")
    batch = caller.create_batch(_msgs(2))

    output_text = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 200,
                        "body": _openai_completion_body(),
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 400,
                        "body": {"error": {"message": "bad request"}},
                    },
                }
            ),
        ]
    )
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id=None
    )
    client.files.content.return_value = SimpleNamespace(text=output_text)

    results = batch.results()
    assert results[0].Message.Message == "hello"
    assert isinstance(results[1].Error, LLMProviderError)

    # return_exceptions=False raises the first request error
    caller.Client = MagicMock()
    _rerun_openai_completed(caller, output_text)
    with pytest.raises(LLMProviderError):
        caller.call_batch(_msgs(2), poll_interval=0)

    out = caller.call_batch(_msgs(2), poll_interval=0, return_exceptions=True)
    assert isinstance(out[0], LLMMessage)
    assert isinstance(out[1], LLMProviderError)


def _rerun_openai_completed(caller, output_text):
    caller.Client.files.create.return_value = SimpleNamespace(id="file-1")
    caller.Client.batches.create.return_value = SimpleNamespace(id="batch_1")
    caller.Client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id=None
    )
    caller.Client.files.content.return_value = SimpleNamespace(
        text=output_text
    )


def test_openai_call_batch_polls_until_completed():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")
    client.batches.retrieve.side_effect = [
        SimpleNamespace(
            status="in_progress", output_file_id=None, error_file_id=None
        ),
        SimpleNamespace(
            status="completed", output_file_id="out-1", error_file_id=None
        ),
        SimpleNamespace(
            status="completed", output_file_id="out-1", error_file_id=None
        ),
    ]
    client.files.content.return_value = SimpleNamespace(
        text=json.dumps(
            {
                "custom_id": "req-0",
                "response": {
                    "status_code": 200,
                    "body": _openai_completion_body(),
                },
            }
        )
    )

    out = caller.call_batch(_msgs(1), poll_interval=0)
    assert [m.Message for m in out] == ["hello"]


def test_batch_wait_terminal_failure_and_timeout():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")
    batch = caller.create_batch(_msgs(1))

    client.batches.retrieve.return_value = SimpleNamespace(
        status="failed", output_file_id=None, error_file_id=None
    )
    with pytest.raises(LLMBatchError, match="failed"):
        batch.wait(poll_interval=0)

    client.batches.retrieve.return_value = SimpleNamespace(
        status="in_progress", output_file_id=None, error_file_id=None
    )
    with pytest.raises(TimeoutError):
        batch.wait(poll_interval=0.001, timeout=0.01)


def test_batch_save_and_load_roundtrip(tmp_path):
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")
    batch = caller.create_batch(_msgs(2), response_schema=LLMMessage)

    path = tmp_path / "job.json"
    batch.save(path)
    data = json.loads(path.read_text())
    assert data["model"] == DEFAULT_OPENAI_MODEL
    assert data["batch_id"] == "batch_1"
    assert data["response_schema"] == "fdllm.llmtypes:LLMMessage"

    # re-attach through the saved file; get_caller resolves the model
    loaded = load_batch(path)
    assert loaded.Id == "batch_1"
    assert list(loaded.Requests) == ["req-0", "req-1"]
    assert loaded.Response_Schema is LLMMessage

    # explicit caller works too, and results() functions on the handle
    loaded = load_batch(path, caller)
    loaded.Response_Schema = None  # mocked bodies aren't schema-shaped
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id=None
    )
    client.files.content.return_value = SimpleNamespace(
        text="\n".join(
            json.dumps(
                {
                    "custom_id": cid,
                    "response": {
                        "status_code": 200,
                        "body": _openai_completion_body(),
                    },
                }
            )
            for cid in ("req-0", "req-1")
        )
    )
    assert [r.Message.Message for r in loaded.results()] == [
        "hello",
        "hello",
    ]


def test_get_batch_reattaches_handle():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id=None
    )
    client.files.content.return_value = SimpleNamespace(
        text=json.dumps(
            {
                "custom_id": "req-0",
                "response": {
                    "status_code": 200,
                    "body": _openai_completion_body(),
                },
            }
        )
    )
    batch = caller.get_batch(
        "batch_stored", [LLMMessage(Role="user", Message="hi")]
    )
    assert batch.Id == "batch_stored"
    assert [r.Message.Message for r in batch.results()] == ["hello"]


def test_openai_batch_cancel():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")
    batch = caller.create_batch(_msgs(1))
    batch.cancel()
    client.batches.cancel.assert_called_once_with("batch_1")


# ===== Anthropic provider batch =====


def _anthropic_message(text="hello"):
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

    return BetaMessage(
        id="msg_1",
        type="message",
        role="assistant",
        content=[BetaTextBlock(type="text", text=text)],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=BetaUsage(input_tokens=3, output_tokens=4),
    )


def test_anthropic_create_batch_params():
    caller = ClaudeCaller(DEFAULT_CLAUDE_MODEL)
    caller.Client = MagicMock()
    caller.Client.beta.messages.batches.create.return_value = SimpleNamespace(
        id="msgbatch_1"
    )

    batch = caller.create_batch(
        [
            LLMBatchRequest(
                Messages=[
                    LLMMessage(Role="system", Message="sys"),
                    LLMMessage(Role="user", Message="hi"),
                ],
            )
        ],
        max_tokens=123,
    )

    _, kwargs = caller.Client.beta.messages.batches.create.call_args
    requests = kwargs["requests"]
    assert len(requests) == 1
    assert requests[0]["custom_id"] == "req-0"
    params = requests[0]["params"]
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["max_tokens"] == 123
    assert params["system"] == "sys"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert batch.Id == "msgbatch_1"


def test_anthropic_batch_results_succeeded_and_errored():
    caller = ClaudeCaller(DEFAULT_CLAUDE_MODEL)
    caller.Client = MagicMock()
    batches = caller.Client.beta.messages.batches
    batches.create.return_value = SimpleNamespace(id="msgbatch_1")
    batch = caller.create_batch(_msgs(2))

    batches.retrieve.return_value = SimpleNamespace(
        processing_status="in_progress"
    )
    assert batch.status() is LLMBatchStatus.PENDING

    batches.retrieve.return_value = SimpleNamespace(
        processing_status="ended"
    )
    batches.results.return_value = iter(
        [
            SimpleNamespace(
                custom_id="req-1",  # out of order on purpose
                result=SimpleNamespace(
                    type="succeeded", message=_anthropic_message("two")
                ),
            ),
            SimpleNamespace(
                custom_id="req-0",
                result=SimpleNamespace(
                    type="succeeded", message=_anthropic_message("one")
                ),
            ),
            SimpleNamespace(
                custom_id="req-x",
                result=SimpleNamespace(
                    type="errored",
                    error=SimpleNamespace(
                        error=SimpleNamespace(
                            type="invalid_request_error", message="bad"
                        )
                    ),
                ),
            ),
        ]
    )

    results = batch.results()
    # results re-ordered to submission order; extra custom_id ignored
    assert [r.Custom_Id for r in results] == ["req-0", "req-1"]
    assert [r.Message.Message for r in results] == ["one", "two"]
    assert results[0].Message.TokensUsed == 7

    # now include a real errored result for a submitted id
    batches.results.return_value = iter(
        [
            SimpleNamespace(
                custom_id="req-0",
                result=SimpleNamespace(
                    type="errored",
                    error=SimpleNamespace(
                        error=SimpleNamespace(
                            type="invalid_request_error", message="bad"
                        )
                    ),
                ),
            ),
        ]
    )
    results = batch.results()
    assert isinstance(results[0].Error, LLMProviderError)
    # req-1 has no result at all
    assert isinstance(results[1].Error, LLMProviderError)


def test_anthropic_acreate_batch():
    caller = ClaudeCaller(DEFAULT_CLAUDE_MODEL)
    caller.AClient = MagicMock()
    caller.AClient.beta.messages.batches.create = AsyncMock(
        return_value=SimpleNamespace(id="msgbatch_2")
    )

    async def main():
        return await caller.acreate_batch(_msgs(1))

    batch = asyncio.run(main())
    assert batch.Id == "msgbatch_2"


# ===== Google GenAI provider batch =====


def _google_response(text="hello"):
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part(text=text)]
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1,
            candidates_token_count=2,
            total_token_count=3,
        ),
        model_version="models/gemini-2.5-flash",
    )


def test_google_create_batch_inlined_requests():
    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    caller.Client.batches.create.return_value = SimpleNamespace(
        name="batches/123"
    )

    batch = caller.create_batch(
        [
            LLMBatchRequest(
                Messages=[LLMMessage(Role="user", Message="hi")],
            )
        ],
        max_tokens=77,
    )

    _, kwargs = caller.Client.batches.create.call_args
    assert kwargs["model"] == caller.Model.Api_Model_Name
    src = kwargs["src"]
    assert len(src) == 1
    req = src[0]
    assert req.metadata == {"key": "req-0"}
    assert req.config.max_output_tokens == 77
    assert req.contents[0].parts[0].text == "hi"
    assert batch.Id == "batches/123"


def test_google_batch_results():
    from google.genai import types

    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    caller.Client.batches.create.return_value = SimpleNamespace(
        name="batches/123"
    )
    batch = caller.create_batch(_msgs(2))

    caller.Client.batches.get.return_value = SimpleNamespace(
        state=types.JobState.JOB_STATE_RUNNING, dest=None
    )
    assert batch.status() is LLMBatchStatus.PENDING

    caller.Client.batches.get.return_value = SimpleNamespace(
        state=types.JobState.JOB_STATE_SUCCEEDED,
        dest=SimpleNamespace(
            inlined_responses=[
                types.InlinedResponse(
                    response=_google_response("one"),
                    metadata={"key": "req-0"},
                ),
                types.InlinedResponse(
                    response=None,
                    metadata={"key": "req-1"},
                    error=SimpleNamespace(code=3, message="invalid"),
                ),
            ]
        ),
    )

    results = batch.results()
    assert [r.Custom_Id for r in results] == ["req-0", "req-1"]
    assert results[0].Message.Message == "one"
    assert results[0].Message.TokensUsed == 3
    assert isinstance(results[1].Error, LLMProviderError)


def test_google_batch_results_raise_when_not_completed():
    from google.genai import types

    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    caller.Client.batches.create.return_value = SimpleNamespace(
        name="batches/123"
    )
    batch = caller.create_batch(_msgs(1))
    caller.Client.batches.get.return_value = SimpleNamespace(
        state=types.JobState.JOB_STATE_RUNNING, dest=None
    )
    with pytest.raises(LLMBatchError):
        batch.results()


# ===== Duplicate Custom_Id =====


def test_coerce_batch_requests_duplicate_ids_raise():
    caller = FakeCaller()

    with pytest.raises(ValueError, match="Duplicate Custom_Id"):
        caller._coerce_batch_requests(
            [
                LLMBatchRequest(
                    Messages=[LLMMessage(Role="user", Message="a")],
                    Custom_Id="dup",
                ),
                LLMBatchRequest(
                    Messages=[LLMMessage(Role="user", Message="b")],
                    Custom_Id="dup",
                ),
            ]
        )

    # a supplied id can also collide with a generated req-{i}
    with pytest.raises(ValueError, match="Duplicate Custom_Id"):
        caller._coerce_batch_requests(
            [
                LLMBatchRequest(
                    Messages=[LLMMessage(Role="user", Message="a")],
                    Custom_Id="req-2",
                ),
                LLMMessage(Role="user", Message="b"),
                LLMMessage(Role="user", Message="c"),
            ]
        )


# ===== Async batch methods =====


def _async_pager(items):
    async def gen():
        for item in items:
            yield item

    return gen()


def _openai_acaller():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    caller.AClient = MagicMock()
    return caller


def test_openai_abatch_status_results_cancel():
    caller = _openai_acaller()
    batch = caller.get_batch("batch_1", _msgs(2))

    caller.AClient.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(
            status="in_progress", output_file_id=None, error_file_id=None
        )
    )
    assert asyncio.run(batch.astatus()) is LLMBatchStatus.PENDING

    caller.AClient.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(
            status="completed", output_file_id="out-1", error_file_id="err-1"
        )
    )
    out_text = "\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": _openai_completion_body(text=f"resp-{cid}"),
                },
            }
        )
        for cid in ("req-0", "req-1")
    )
    err_text = "\n".join(
        [
            # mentions req-0 too, but the output-file result must win
            json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 500,
                        "body": {"error": {"message": "should lose"}},
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "req-2",
                    "response": {
                        "status_code": 400,
                        "body": {"error": {"message": "req-2 bad"}},
                    },
                }
            ),
        ]
    )
    caller.AClient.files.content = AsyncMock(
        side_effect=lambda fid: SimpleNamespace(
            text=out_text if fid == "out-1" else err_text
        )
    )
    results = asyncio.run(batch.aresults())
    assert [r.Custom_Id for r in results] == ["req-0", "req-1"]
    assert [r.Message.Message for r in results] == [
        "resp-req-0",
        "resp-req-1",
    ]

    caller.AClient.batches.cancel = AsyncMock()
    asyncio.run(batch.acancel())
    caller.AClient.batches.cancel.assert_awaited_once_with("batch_1")


def test_openai_acreate_and_acall_batch():
    caller = _openai_acaller()
    caller.AClient.files.create = AsyncMock(
        return_value=SimpleNamespace(id="file-1")
    )
    caller.AClient.batches.create = AsyncMock(
        return_value=SimpleNamespace(id="batch_1")
    )

    batch = asyncio.run(caller.acreate_batch(_msgs(1)))
    assert batch.Id == "batch_1"

    caller.AClient.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(
            status="completed", output_file_id="out-1", error_file_id=None
        )
    )
    caller.AClient.files.content = AsyncMock(
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 200,
                        "body": _openai_completion_body(),
                    },
                }
            )
        )
    )
    out = asyncio.run(caller.acall_batch(_msgs(1), poll_interval=0))
    assert [m.Message for m in out] == ["hello"]


def test_batch_await_completion():
    caller = _openai_acaller()
    batch = caller.get_batch("batch_1", _msgs(1))

    caller.AClient.batches.retrieve = AsyncMock(
        side_effect=[
            SimpleNamespace(
                status="in_progress",
                output_file_id=None,
                error_file_id=None,
            ),
            SimpleNamespace(
                status="completed",
                output_file_id="out-1",
                error_file_id=None,
            ),
            SimpleNamespace(
                status="completed",
                output_file_id="out-1",
                error_file_id=None,
            ),
        ]
    )
    caller.AClient.files.content = AsyncMock(
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 200,
                        "body": _openai_completion_body(),
                    },
                }
            )
        )
    )
    results = asyncio.run(batch.await_completion(poll_interval=0))
    assert [r.Message.Message for r in results] == ["hello"]


def test_batch_await_completion_failure_and_timeout():
    caller = _openai_acaller()
    batch = caller.get_batch("batch_1", _msgs(1))

    caller.AClient.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(
            status="expired", output_file_id=None, error_file_id=None
        )
    )
    with pytest.raises(LLMBatchError, match="expired"):
        asyncio.run(batch.await_completion(poll_interval=0))

    caller.AClient.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(
            status="in_progress", output_file_id=None, error_file_id=None
        )
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            batch.await_completion(poll_interval=0.001, timeout=0.01)
        )


def test_anthropic_abatch_status_results_cancel():
    caller = ClaudeCaller(DEFAULT_CLAUDE_MODEL)
    caller.Client = MagicMock()
    caller.AClient = MagicMock()
    batch = caller.get_batch("msgbatch_1", _msgs(2))

    caller.AClient.beta.messages.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(processing_status="ended")
    )
    assert asyncio.run(batch.astatus()) is LLMBatchStatus.COMPLETED

    caller.AClient.beta.messages.batches.results = AsyncMock(
        return_value=_async_pager(
            [
                SimpleNamespace(
                    custom_id="req-1",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=_anthropic_message("two"),
                    ),
                ),
                SimpleNamespace(
                    custom_id="req-0",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=_anthropic_message("one"),
                    ),
                ),
            ]
        )
    )
    results = asyncio.run(batch.aresults())
    assert [r.Custom_Id for r in results] == ["req-0", "req-1"]
    assert [r.Message.Message for r in results] == ["one", "two"]

    caller.Client.beta.messages.batches.cancel = MagicMock()
    batch.cancel()
    caller.Client.beta.messages.batches.cancel.assert_called_once_with(
        "msgbatch_1"
    )
    caller.AClient.beta.messages.batches.cancel = AsyncMock()
    asyncio.run(batch.acancel())
    caller.AClient.beta.messages.batches.cancel.assert_awaited_once_with(
        "msgbatch_1"
    )


def test_google_abatch_status_results_cancel():
    from google.genai import types

    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("batches/123", _msgs(1))

    caller.Client.aio.batches.get = AsyncMock(
        return_value=SimpleNamespace(
            state=types.JobState.JOB_STATE_SUCCEEDED,
            dest=SimpleNamespace(
                inlined_responses=[
                    types.InlinedResponse(
                        response=_google_response("hello"),
                        metadata={"key": "req-0"},
                    )
                ]
            ),
        )
    )
    assert asyncio.run(batch.astatus()) is LLMBatchStatus.COMPLETED
    results = asyncio.run(batch.aresults())
    assert [r.Message.Message for r in results] == ["hello"]

    caller.Client.batches.cancel = MagicMock()
    batch.cancel()
    caller.Client.batches.cancel.assert_called_once_with(name="batches/123")
    caller.Client.aio.batches.cancel = AsyncMock()
    asyncio.run(batch.acancel())
    caller.Client.aio.batches.cancel.assert_awaited_once_with(
        name="batches/123"
    )


def test_google_acreate_batch():
    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    caller.Client.aio.batches.create = AsyncMock(
        return_value=SimpleNamespace(name="batches/9")
    )
    batch = asyncio.run(caller.acreate_batch(_msgs(1)))
    assert batch.Id == "batches/9"


# ===== OpenAI output-line parsing edge cases =====


def test_openai_parse_output_lines_skips_bad_records():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("batch_1", _msgs(2))

    text = "\n".join(
        [
            "",
            "   ",
            # record without custom_id is ignored
            json.dumps({"response": {"status_code": 200, "body": {}}}),
            json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 200,
                        "body": _openai_completion_body(),
                    },
                }
            ),
        ]
    )
    by_id = batch._parse_output_lines(text)
    assert list(by_id) == ["req-0"]
    assert by_id["req-0"].Message.Message == "hello"


def test_openai_parse_output_lines_error_shapes():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("batch_1", _msgs(4))

    text = "\n".join(
        [
            # missing status_code, error nested in body
            json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "body": {"error": {"message": "nested bad"}}
                    },
                }
            ),
            # error stored at the record level as a plain string
            json.dumps({"custom_id": "req-1", "error": "record bad"}),
            # status 200 but the body fails output parsing
            json.dumps(
                {
                    "custom_id": "req-2",
                    "response": {"status_code": 200, "body": {}},
                }
            ),
            # response key entirely absent
            json.dumps({"custom_id": "req-3"}),
        ]
    )
    by_id = batch._parse_output_lines(text)

    assert isinstance(by_id["req-0"].Error, LLMProviderError)
    assert "nested bad" in str(by_id["req-0"].Error)

    assert isinstance(by_id["req-1"].Error, LLMProviderError)
    assert "record bad" in str(by_id["req-1"].Error)

    assert by_id["req-2"].Message is None
    assert by_id["req-2"].Error is not None

    assert isinstance(by_id["req-3"].Error, LLMProviderError)


def test_format_output_failure_becomes_error_result():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    batch = caller.get_batch("batch_1", _msgs(1))

    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id=None
    )
    client.files.content.return_value = SimpleNamespace(
        text=json.dumps(
            {
                "custom_id": "req-0",
                "response": {
                    "status_code": 200,
                    "body": _openai_completion_body(),
                },
            }
        )
    )
    with patch.object(
        OpenAICaller, "format_output", side_effect=RuntimeError("fmt broke")
    ):
        results = batch.results()
    assert results[0].Message is None
    assert isinstance(results[0].Error, RuntimeError)


def test_openai_batch_collect_merges_error_file():
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    batch = caller.get_batch("batch_1", _msgs(2))

    out_text = json.dumps(
        {
            "custom_id": "req-0",
            "response": {
                "status_code": 200,
                "body": _openai_completion_body(),
            },
        }
    )
    # the error file also mentions req-0; the output result must win
    err_text = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 500,
                        "body": {"error": {"message": "should lose"}},
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 400,
                        "body": {"error": {"message": "req-1 bad"}},
                    },
                }
            ),
        ]
    )
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="out-1", error_file_id="err-1"
    )
    client.files.content.side_effect = lambda fid: SimpleNamespace(
        text=out_text if fid == "out-1" else err_text
    )

    results = batch.results()
    assert results[0].Message.Message == "hello"
    assert isinstance(results[1].Error, LLMProviderError)
    assert "req-1 bad" in str(results[1].Error)


# ===== Anthropic expired/canceled results =====


def test_anthropic_batch_results_expired_and_canceled():
    caller = ClaudeCaller(DEFAULT_CLAUDE_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("msgbatch_1", _msgs(2))

    caller.Client.beta.messages.batches.retrieve.return_value = (
        SimpleNamespace(processing_status="ended")
    )
    caller.Client.beta.messages.batches.results.return_value = iter(
        [
            SimpleNamespace(
                custom_id="req-0",
                result=SimpleNamespace(type="expired"),
            ),
            SimpleNamespace(
                custom_id="req-1",
                result=SimpleNamespace(type="canceled"),
            ),
        ]
    )
    results = batch.results()
    assert isinstance(results[0].Error, LLMProviderError)
    # expired requests may be worth resubmitting
    assert results[0].Error.retryable is True
    assert isinstance(results[1].Error, LLMProviderError)
    assert results[1].Error.retryable is False


# ===== Google status mapping and result edge cases =====


def test_google_status_mapping():
    from google.genai import types

    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("batches/123", _msgs(1))

    cases = [
        (types.JobState.JOB_STATE_QUEUED, LLMBatchStatus.PENDING),
        (types.JobState.JOB_STATE_FAILED, LLMBatchStatus.FAILED),
        (types.JobState.JOB_STATE_CANCELLED, LLMBatchStatus.CANCELLED),
        (types.JobState.JOB_STATE_EXPIRED, LLMBatchStatus.EXPIRED),
    ]
    for state, expected in cases:
        caller.Client.batches.get.return_value = SimpleNamespace(
            state=state, dest=None
        )
        assert batch.status() is expected


def test_google_batch_results_raise_for_file_dest():
    from google.genai import types

    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("batches/123", _msgs(1))
    caller.Client.batches.get.return_value = SimpleNamespace(
        state=types.JobState.JOB_STATE_SUCCEEDED,
        dest=SimpleNamespace(
            inlined_responses=None, file_name="files/results-1"
        ),
    )
    with pytest.raises(LLMBatchError, match="not supported"):
        batch.results()


def test_google_batch_results_index_fallback():
    from google.genai import types

    caller = GoogleGenAICaller(DEFAULT_GEMINI_MODEL)
    caller.Client = MagicMock()
    batch = caller.get_batch("batches/123", _msgs(2))
    caller.Client.batches.get.return_value = SimpleNamespace(
        state=types.JobState.JOB_STATE_SUCCEEDED,
        dest=SimpleNamespace(
            inlined_responses=[
                # metadata key doesn't match any submitted Custom_Id:
                # the result is matched back by position instead
                types.InlinedResponse(
                    response=_google_response("one"),
                    metadata={"key": "provider-key"},
                ),
                types.InlinedResponse(
                    response=_google_response("two"),
                    metadata=None,
                ),
            ]
        ),
    )
    results = batch.results()
    assert [r.Custom_Id for r in results] == ["req-0", "req-1"]
    assert [r.Message.Message for r in results] == ["one", "two"]


# ===== save/load edge cases =====


def test_save_and_load_batch_without_schema(tmp_path):
    caller = OpenAICaller(DEFAULT_OPENAI_MODEL)
    client = _mock_openai_client(caller)
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = SimpleNamespace(id="batch_1")
    batch = caller.create_batch(_msgs(1))

    path = tmp_path / "job.json"
    batch.save(path)
    assert json.loads(path.read_text())["response_schema"] is None

    loaded = load_batch(path, caller)
    assert loaded.Id == "batch_1"
    assert loaded.Response_Schema is None


def test_load_batch_unresolvable_schema(tmp_path, caplog):
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": DEFAULT_OPENAI_MODEL,
                "batch_id": "batch_9",
                "requests": [
                    {
                        "Messages": [
                            {"Role": "user", "Message": "hi"}
                        ],
                        "Custom_Id": "req-0",
                    }
                ],
                "response_schema": "no_such_module:Missing",
            }
        )
    )
    loaded = load_batch(path, OpenAICaller(DEFAULT_OPENAI_MODEL))
    assert loaded.Response_Schema is None
    assert "could not resolve saved response schema" in caplog.text
