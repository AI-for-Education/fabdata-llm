from .llmtypes import (
    LLMModelType,
    LLMCaller,
    OpenAIModelType,
    AnthropicModelType,
    AnthropicVisionModelType,
    AzureOpenAIModelType,
    AzureMistralAIModelType,
    VertexAIModelType,
    OpenRouterModelType,
    GroqModelType,
    FireworksModelType
)
from .openai import GPTCaller, OpenAICaller
from .anthropic import ClaudeCaller
from .mistralai import MistralCaller
from .sysutils import load_models

API_CALLERS = {
    "OpenAI": OpenAICaller,
    "AzureOpenAI": OpenAICaller,
    "AzureMistralAI": MistralCaller,
    "Anthropic": ClaudeCaller,
    "AnthropicVision": ClaudeCaller,
    "VertexAI": OpenAICaller,
}

def get_caller(model: str) -> LLMCaller:
    models = load_models()
    if model not in models:
        raise NotImplementedError(f"{model} is not a recognised model name, check models.yaml")
    model_params = models[model]
    if model_params.api_interface not in API_CALLERS:
        raise ValueError(f"{model_params.api_inferface} is not a recognised API interface")
    else:
        return API_CALLERS[model_params.api_interface](model)