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
    api_interface = model_params.get("api_interface") or model_params.get("Api_Interface")
    if api_interface is None:
        raise ValueError(f"{model} does not have an API interface defined")
    if api_interface not in API_CALLERS:
        raise ValueError(f"{api_interface} is not a recognised API interface")
    else:
        return API_CALLERS[api_interface](model)
