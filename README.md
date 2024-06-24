## Introduction

FabData-LLM is a set of high-level abstractions around various LLM API providers. It is written in Python and currently covers OpenAI, Azure OpenAI, Anthropic, and Azure MistralAI APIs.

### Why you might consider using this

- You want to create a chatbot with stored history and automatic history token management in 3 lines of code:

    ```python
    # GPT 3.5 Turbo
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("gpt-3.5-turbo"))
    
    inmsg, outmsg = chatter.chat("Hello there")
    print(outmsg)
    ```

    ```python
    # Claude 3
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("claude-3-opus-20240229"))

    inmsg, outmsg = chatter.chat("Hello there")
    print(outmsg)
    ```

    ```python
    # GPT 4 Vision Preview
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("gpt-4-vision-preview"))

    ### load images here into a list of PIL Images
    # images : List[PIL.Image.Image]

    inmsg, outmsg = chatter.chat(
        "Hello there, can you compare these images for me",
        images=images,
        detail="high"
    )
    print(outmsg)
    ```

    - Customize system message placement (multiple system messages can lead to improved robustness against jailbreaks, for example)

        ```python
        from fdllm import get_caller
        from fdllm.chat import ChatController

        chatter = ChatController(
            Caller=get_caller("gpt-3.5-turbo"),
            Sys_Msg={
                0: "This will appear at the start of the conversation"
                -1: "This will appear at the end of the conversation, after the user chat input"
                -2: "This will appear at the end of the conversation, before the user chat input"
            }
        )
        ```
        NOTE: This feature is not supported for Anthropic models, as they only accept a single system message. Setting any `Sys_Msg` key other than `0` will cause a `ValueError` at chat time with Anthropic callers
- You want to easily write new tools with code and definition schema all encapsulated in a single object, and call logic (including parallel calls and chaining of sequential calls) and parameter validation all handled automatically during chat. 

    One of the major inconveniences with writing tools is that the tool definition schemas and the actual functions that they call aren't connected to each other in any way. The `Tool` class in FabData-LLM brings the two things together in a single object. Tool definition schemas are automatically generated from the tool's parameters according to different formats (currently only supports OpenAI's and Anthropic's formats), and tool execution is wrapped in parameter validation steps. This makes debugging easier as you can more easily identify formatting errors in tool call instructions returned by LLMs.

    The `ToolUsePlugin` class lets you connect a tool or set of tools to a `ChatController` object. It automatically handles all of the logic of passing tool definition schemas, intercepting tool call instructions, executing tool calls, and communicating the results back to the LLM. Parallel tool calls and sequential tool calls are also handled automatically. It currently supports all tool-enambled models from OpenAI and Anthropic (`gpt-3.5-turbo-1106`, `gpt-3.5-turbo-0125`, `gpt-4-1106-preview`, `gpt-4-0125-preview`, `gpt-4-turbo-2024-04-09`, `gpt-4o-2024-05-13`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`). Any custom models from the OpenAI or Anthropic families that are known to support tool use can be used if set with the parameter `Tool_Use: True` in the model config file.

    ```python
    from fdllm import get_caller
    from fdllm.chat import ChatController
    from fdllm.tooluse import ToolUsePlugin, Tool, ToolParam

    class TestTool1(Tool):
        name = "mul"
        description = "Multiply 2 numbers"
        params = {
            "x": ToolParam(type="number", required=True),
            "y": ToolParam(type="number", required=True),
        }

        def execute(self, **params):
            res = params["x"] * params["y"]
            return f"{res :.4f}"

        async def aexecute(self, **params):
            return self.execute()
    

    class TestTool2(Tool):
        name = "add"
        description = "Add 2 numbers"
        params = {
            "x": ToolParam(type="number", required=True),
            "y": ToolParam(type="number", required=True),
        }

        def execute(self, **params):
            res = params["x"] + params["y"]
            return f"{res :.4f}"

        async def aexecute(self, **params):
            return self.execute()
    

    chatter = ChatController(Caller=get_caller("gpt-4-1106-preview"),)
    chatter.register_plugin(
        ToolUsePlugin(Tools=[TestTool1(), TestTool2()])
    )

    inmsg, outmsg = chatter.chat("(pi * 5.4) + (6 * e)")
    print(outmsg)

    # view the chat history, including the chain of tools calls
    print(chatter.History)

    # view the history of only the last conversational event (i.e. from user input to final response)
    print(chatter.recent_history)

    # view only the tool calls from the last conversational event (i.e. from user input to final response)
    print(chatter.recent_tool_calls)

    # view only the tool call responses from the last conversational event (i.e. from user input to final response)
    print(chatter.recent_tool_responses)
    ```

    - See [**FabData-LLM-retrieval**](https://github.com/AI-for-Education/fabdata-llm-retrieval) for a more complex example of `ToolUsePlugin` for implementing a Retrieval-Augmented Generation system

    - As well as the provided `ToolUsePlugin`, create other types of plugin with the ```ChatPlugin``` abstract base class. Registered plugins have the ability to intercept and modify both user inputs and Caller responses during chat sessions, make their own LLM API calls, and mutate the state of the ChatController object
- You want to switch between OpenAI API and multiple different Azure OpenAI endpoints without having to change global environment variable configurations and without having to deal with variations between the two APIs
    - Fabdata-LLM allows you to register custom model configuration yaml files with invidual endpoints, api keys, and other client arguments for each model

        ```yaml
        OpenAI:
            gpt-3.5-turbo:
                Client_Args:
                    api_key: my_openai_api_key1
            gpt-3.5-turbo-0301:
                Client_Args:
                    api_key: my_openai_api_key2
            gpt-3.5-turbo-0613:
                Client_Args:
                    api_key: my_openai_api_key3

        AzureOpenAI:
            my_azure_openai_deployment_1:
                Token_Window: 8192
                Client_Args:
                    azure_endpoint: https://my-endpoint-1.openai.azure.com
                    api_version: 2023-09-15-preview
                    api_key: my_azure_openai_api_key1
            my_azure_openai_deployment_2:
                Token_Window: 32768
                Client_Args:
                    azure_endpoint: https://my-endpoint-2.openai.azure.com
                    api_version: 2023-09-15-preview
                    api_key: my_azure_openai_api_key2
        
        AzureMistralAI:
            my_azure_ai_deployment_1:
            Token_Window: 32000
            Client_Args:
                endpoint: https://my-endpoint-1.my-region.inference.ai.azure.com
                api_key: my_azure_ai_api_key1
        ```
    - See `model_config_template.yaml ` for an overview of supported settings
    - WARNING: As with any plain text file containing API keys (such as a .env file), you should definitely not commit these files into any code repositories or otherwise share in an unsecured manner

    - After registering a custom model file, you can simply refer to the custom models by name when creating the LLMCaller object

        ```python
        from fdllm import get_caller
        from fdllm.sysutils import register_models
        
        register_models("/path/to/my/custom/models.yaml")

        caller = get_caller("my_azure_deployment_1")
        ```

    - Custom model configurations are deep-merged with the base model configuration, allowing you to set only a subset of custom values for an existing model (e.g. setting the individual api keys for gpt-3.5-turbo in the above example)
    - You can still use global environment variables if you prefer. The following environment variables will be recognised:

        ```env
        # will apply globally to all models that use the OpenAI API
            OPENAI_API_KEY
        # will apply globally to all models that use the Azure OpenAI API
            AZURE_OPENAI_API_KEY
            AZURE_OPENAI_ENDPOINT
            OPENAI_API_VERSION
        # will apply globally to all models that use the Anthropic API
            ANTHROPIC_API_KEY
        # will apply globally to all models that use the Mistral API
            MISTRAL_API_KEY
        ``````
    
- You want to use the latest models, such as ```gpt-4-turbo-2024-04-09``` and ```gpt-4o``` from OpenAI, ```claude-3-opus-20240229``` from Anthropic, and `Mistral-large` (hosted on Azure AI)
    - FabData-LLM supports the latest versions of OpenAI's and Anthropics's APIs (1.30.1 and 0.26.0 respectively at the time of writing) and the latest models, including multi-modal models
    - FabData-LLM supports Mistral AI models hosted on Azure AI endpoints
- You want all of this functionality in both sync and async applications
    - All LLMCaller objects have two call methods: ```call``` and ```acall```
    - ChatController object has two chat methods: ```chat``` and ```achat```

- You want to use well-tested code
    - FabData-LLM currently has over 80% branch coverage in pytest

## How to use

### Installation

The easiest way to install is to use the dedicated conda environment provided in ```environment.yml```:
```
conda env create -f environment.yml
```
This creates a conda environment called ```fabdata-llm``` and installs the package along with dependencies inside it. See [Miniconda install](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) for instructions on how to install conda if you don't already have it installed and [condas user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) for a more general guide to using conda.

You can also install into any existing python (3.10, or 3.11) environment by running:
```
pip install .
```
from inside the repository directory. If you encounter any dependency clashes then please feel free to report in the discussion and we will see if we can loosen any of the dependency restrictions.

### Configuration
The package comes with a base model configuration which can be extended by user-provided custom configurations. You can get the base model configuration dictionary by:
```python
from fdllm.sysutils import list_models

models = list_models(full_info=True, base_only=True)
```
There are 5 categories of model:
- OpenAI
- AzureOpenAI
- Anthropic
- AnthropicVision
- AzureMistralAI

A full template for a custom model configuration file can be found here: [model template](model_config_template.yaml). All fields except for those marked required are optional and can be omitted. Once created, models in the custom configuration can be added to the set of useable models by:
```python
from fdllm.sysutils import register_models

register_models("/path/to/my/custom/models.yaml")
```

The configuration file is particularly useful for configuring multiple different Azure OpenAI deployments, as they will likely point to different endpoints and use different api keys. If you are only interested in setting global api keys for different providers, then the base model configuration is likely enough for your needs, and you can instead set the following environment variables:
```env
# will apply globally to all models that use the OpenAI API
    OPENAI_API_KEY
# will apply globally to all models that use the Azure OpenAI API
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    OPENAI_API_VERSION
# will apply globally to all models that use the Anthropic API
    ANTHROPIC_KEY
# will apply globally to all models that use the Mistral API
    MISTRAL_API_KEY
``````

### Usage

The [sandbox](./sandbox) folder contains examples of different use cases.