from .llmtypes import (
    LLMModelType,
    LLMCaller,
    OpenAIModelType,
    AnthropicModelType,
    AnthropicVisionModelType,
    AzureOpenAIModelType,
    AzureMistralAIModelType,
)
from .openai import GPTCaller
from .anthropic import ClaudeCaller
from .mistralai import MistralCaller

def get_caller(model: str) -> LLMCaller:
    modeltype = LLMModelType.get_type(model)
    if modeltype in [OpenAIModelType, AzureOpenAIModelType]:
        return GPTCaller(model)
    elif modeltype in [AnthropicModelType]:
        return ClaudeCaller(model)
    elif modeltype in [AnthropicVisionModelType]:
        return ClaudeCaller(model)
    elif modeltype in [AnthropicVisionModelType]:
        return ClaudeCaller(model)
    elif modeltype in [AzureMistralAIModelType]:
        return MistralCaller(model)
    elif isinstance(modeltype, tuple):
        raise NotImplementedError(
            f"{model} is not a unique name in model config."
            " Currently all model names must be unique."
        )
    else:
        raise NotImplementedError(f"{model} is not a valid model name")