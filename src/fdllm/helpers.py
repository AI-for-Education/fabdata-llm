from .anthropic import ClaudeCaller, ClaudeStreamingCaller
from .llmtypes import LLMCaller
from .mistralai import MistralCaller
from .openai import OpenAICaller, OpenAICompletionsCaller
from .google import GoogleGenAICaller
from .bedrock import BedrockCaller
from .sysutils import load_models

API_CALLERS = {
    "OpenAI": OpenAICaller,
    "OpenAICompletions": OpenAICompletionsCaller,
    "AzureOpenAI": OpenAICaller,
    "AzureMistralAI": MistralCaller,
    "Anthropic": ClaudeCaller,
    "AnthropicStreaming": ClaudeStreamingCaller,
    "AnthropicVision": ClaudeCaller,
    "VertexAI": OpenAICaller,
    "GoogleGenAI": GoogleGenAICaller,
    "Bedrock": BedrockCaller
}


def get_caller(model: str) -> LLMCaller:
    models = load_models()
    if model not in models:
        raise NotImplementedError(
            f"{model} is not a recognised model name, check models.yaml"
        )
    model_params = models[model]
    if "Api_Interface" not in model_params:
        raise ValueError(f"{model} does not have an API interface defined")
    if model_params["Api_Interface"] not in API_CALLERS:
        raise ValueError(
            f"{model_params['Api_Interface']} is not a recognised API interface"
        )
    else:
        return API_CALLERS[model_params["Api_Interface"]](model)

