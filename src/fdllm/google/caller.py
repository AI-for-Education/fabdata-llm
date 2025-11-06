from types import GeneratorType
from typing import Any, List, Optional

from google import genai
from pydantic import ConfigDict, BaseModel
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import (
    TopLogprob,
    ChatCompletionTokenLogprob,
)

from ..llmtypes import LLMCallArgNames, LLMCaller, LLMMessage, LLMModelType, LLMToolCall
from ..tooluse import Tool


class GoogleGenAICaller(LLMCaller):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Client: genai.client.Client

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        Modtype = LLMModelType.get_type(model)
        model: LLMModelType = Modtype(Name=model)

        # Gemini Models
        client = genai.Client(**model.Client_Args)
        # TODO: support vertex models
        # client = genai.Client(
        #     vertexai=True, project="your-project-id", location="us-central1"
        # )

        call_arg_names = LLMCallArgNames(
            Model="model",
            Messages="contents",
            Max_Tokens=model.Max_Token_Arg_Name,
            Response_Schema="response_schema",
        )

        # drop CallArgs for google?
        super().__init__(
            Model=model,
            Func=client.models.generate_content,
            AFunc=client.aio.models.generate_content,
            Arg_Names=call_arg_names,
            Defaults={},
            Token_Window=model.Token_Window,
            Token_Limit_Completion=model.Token_Limit_Completion,
            Client=client,
        )

    def format_message(self, message: LLMMessage):
        ### Handle tool results
        if message.Role == "tool":
            return {
                "role": "tool",
                "parts": [
                    {
                        "function_response": {
                            "id": tc.ID,
                            "name": tc.Name,
                            "response": {"result": tc.Response},
                        },
                    }
                    for tc in message.ToolCalls
                ],
            }

        ### Handle assistant tool calls messages
        elif message.Role == "assistant" and message.ToolCalls is not None:
            return {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": tc.ID,
                            "name": tc.Name,
                            "args": tc.Args,
                        }
                    }
                    for tc in message.ToolCalls
                ],
            }
        ### Handle user messages which contain images
        elif message.Role == "user" and message.Images is not None:
            if not self.Model.Vision:
                raise NotImplementedError(
                    f"Tried to pass images but {self.Model.Name} doesn't support images"
                )
            content = [
                *[
                    {
                        "inline_data": {
                            "data": im.get_bytes(),
                            "mime_type": "image/png",
                        },
                    }
                    for im in message.Images
                ],
                {"text": message.Message},
            ]
            return {"role": message.Role, "parts": content}
        else:
            role = message.Role
            if role == "assistant":
                role = "model"
            return {"role": role, "parts": [{"text": message.Message}]}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        sysmsgs = []
        for message in messagelist:
            if message.Role == "system":
                sysmsgs.append(message.Message)
            else:
                outmsg = self.format_message(message)
                if isinstance(outmsg, list):
                    out.extend(outmsg)
                else:
                    out.append(outmsg)
        if sysmsgs:
            self.Defaults["system"] = sysmsgs[0]
        else:
            self.Defaults.pop("system", None)
        return out

    def format_tool(self, tool: Tool):
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
        }
        if tool.params:
            tool_dict["parameters"] = {
                "type": "OBJECT",
                "properties": {
                    key: val.dict(type_upper=True) for key, val in tool.params.items()
                },
                "required": [key for key, val in tool.params.items() if val.required],
            }
        return {"function_declarations": [tool_dict]}

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        kwargs = super()._proc_call_args(
            messages, max_tokens, response_schema, **kwargs
        )
        # move parameters into config argument for genai client
        config = {"system_instruction": kwargs.pop("system", None)}
        if "logprobs" in kwargs:
            config["response_logprobs"] = kwargs.pop("logprobs")
        if "top_logprobs" in kwargs:
            config["logprobs"] = kwargs.pop("top_logprobs")
        for arg in [
            self.Arg_Names.Max_Tokens,
            "tools",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "seed",
            "response_mime_type",
            "response_schema",
        ]:
            config[arg] = kwargs.pop(arg, None)
        if config.get("response_schema"):
            config["response_mime_type"] = "application/json"
        else:
            config.pop("response_schema", None)
        kwargs["config"] = config
        return kwargs

    def format_output(
        self,
        output: Any,
        response_schema: Optional[BaseModel] = None,
        latency: Optional[float] = None,
    ):
        if isinstance(output, GeneratorType):
            return output
        else:
            ## token counts
            usage_metadata = getattr(output, "usage_metadata", None)
            if usage_metadata is not None:
                token_count_kwargs = dict(
                    TokensUsed=usage_metadata.total_token_count,
                    TokensUsedCompletion=usage_metadata.candidates_token_count,
                    TokensUsedReasoning=usage_metadata.thoughts_token_count,
                )
            else:
                token_count_kwargs = {}
            #####
            parts = getattr(output.candidates[0].content, "parts", None)
            logprobs_result = getattr(output.candidates[0], "logprobs_result", None)
            if logprobs_result is not None:
                logprobs = ChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=chosen.token,
                            logprob=chosen.log_probability,
                            top_logprobs=[
                                TopLogprob(
                                    token=lpr_.token, logprob=lpr_.log_probability
                                )
                                for lpr_ in lpr.candidates
                            ],
                        )
                        for chosen, lpr in zip(
                            logprobs_result.chosen_candidates,
                            logprobs_result.top_candidates,
                        )
                    ]
                )
            else:
                logprobs = None
            if parts is not None and getattr(parts[0], "text", None) is not None:
                return LLMMessage(
                    Role="assistant",
                    Message=parts[0].text,
                    LogProbs=logprobs,
                    Latency=latency,
                    **token_count_kwargs,
                )
            elif (
                parts is not None
                and getattr(parts[0], "function_call", None) is not None
            ):
                tcs = [
                    LLMToolCall(
                        ID=p.function_call.id,
                        Name=p.function_call.name,
                        Args=p.function_call.args,
                    )
                    for p in parts
                    if getattr(p, "function_call", None) is not None
                ]
                return LLMMessage(
                    Role="assistant",
                    ToolCalls=tcs,
                    LogProbs=logprobs,
                    Latency=latency,
                    **token_count_kwargs,
                )
            else:
                raise ValueError("Output must be either content or tool call")

    # this is counting tokens with an API call
    # could count tokens locally with vertex sdk
    # https://medium.com/google-cloud/counting-gemini-text-tokens-locally-with-the-vertex-ai-sdk-78979fea6244
    def count_tokens(self, messagelist: List[LLMMessage]):
        return self.Client.models.count_tokens(
            model=self.Model.Api_Model_Name,
            contents=self.format_messagelist(messagelist),
        ).total_tokens
