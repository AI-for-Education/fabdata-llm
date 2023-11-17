from .llmtypes import (
    LLMModelType,
    LLMCaller,
    OpenAIModelType,
    OpenAIVisionModelType,
    AnthropicModelType,
    AzureOpenAIModelType,
)
from .openai import GPTCaller, GPTVisionCaller
from .anthropic import ClaudeCaller

def get_caller(model: str) -> LLMCaller:
    modeltype = LLMModelType.get_type(model)
    if modeltype in [OpenAIModelType, AzureOpenAIModelType]:
        return GPTCaller(model)
    elif modeltype in [OpenAIVisionModelType]:
        return GPTVisionCaller(model)
    elif modeltype in [AnthropicModelType]:
        return ClaudeCaller(model)
    elif isinstance(modeltype, tuple):
        raise NotImplementedError(
            f"{model} is not a unique name in model config."
            " Currently all model names must be unique."
        )
    else:
        raise NotImplementedError(f"{model} is not a valid model name")