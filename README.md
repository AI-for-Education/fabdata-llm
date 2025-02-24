## Introduction

FabData-LLM is a Python package that provides set of high-level abstractions around various LLM API providers. It currently covers OpenAI, Azure OpenAI, Anthropic, Azure Mistral AI, Google Vertex, Google GenAI, and Amazon Bedrock. It can also use any models exposed via an OpenAI compatible API (e.g. OpenRouter, FireWorks, etc.).  

### Why you might consider using this

- You want to create a chatbot with stored history and automatic history token management in 3 lines of code:

    ```python
    # GPT-4o Mini
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("gpt-4o-mini-2024-07-18"))
    
    inmsg, outmsg = chatter.chat("Hello there")
    print(outmsg)
    ```

    ```python
    # Claude 3.5 Sonnet
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("claude-3-5-sonnet-20241022"))

    inmsg, outmsg = chatter.chat("Hello there")
    print(outmsg)
    ```

    ```python
    # GPT-4o with images
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("gpt-4o-2024-08-06"))

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
            Caller=get_caller("gpt-4o-2024-08-06"),
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

    The `ToolUsePlugin` class lets you connect a tool or set of tools to a `ChatController` object. It automatically handles all of the logic of passing tool definition schemas, intercepting tool call instructions, executing tool calls, and communicating the results back to the LLM. Parallel tool calls and sequential tool calls are also handled automatically. It currently supports all tool-enambled models from OpenAI and Anthropic. Any custom models from the OpenAI or Anthropic families that are known to support tool use can be used if set with the parameter `Tool_Use: True` in the model config file.

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
    

    chatter = ChatController(Caller=get_caller("gpt-4o-2024-08-06"),)
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

    - Fabdata-LLM allows you to register custom model configurations in a yaml file, with invidual endpoints, api keys, and other client arguments for each model. Models must have a unique name and can be entered either in a top level `models:` key, or in other arbitrary top-level keys for which child `models:` inherit the settings from the block. 

        ```yaml
        models:
            # this list defines individual models
            # every model must have Api_Interface and Token_Window defined
            gpt-4o:
                Api_Interface: OpenAI
                Token_Window: 128000
                Token_Limit_Completion: 16384
                Tool_Use: True
                Vision: True
        
        Fireworks:
            # these parameters will apply to models listed in models: of this block
            # individual models can overwrite these settings
            Api_Interface: OpenAI
            Max_Tokens_Arg_Name: max_tokens
            Token_Window: 128000
            Api_Key_Env_Var: FIREWORKS_API_KEY
            Client_Args:
                base_url: https://api.fireworks.ai/inference/v1
            models:
                fw-llama-v3p2-1b-instruct:
                Api_Model_Name: accounts/fireworks/models/llama-v3p2-1b-instruct
        
        AzureDeployments:
            Client_Args:
                api_version: 2023-09-15-preview
            models:
                my_azure_openai_deployment_1:
                    Token_Window: 8192
                    Api_Key_Env_Var: AZURE_DEPLOYMENT1_API_KEY
                    Client_Args:
                        azure_endpoint: https://my-endpoint-1.openai.azure.com
                my_azure_openai_deployment_1:
                    Token_Window: 8192
                    Client_Args:
                        azure_endpoint: https://my-endpoint-1.openai.azure.com
                        api_key: my_azure_openai_api_key2
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
    - You can still use global environment variables if you prefer. The following environment variables will be recognised, by default, but custom environment variables can be specified for individual models with the `Api_Key_Env_Var` setting:

        ```env
        # will apply globally to all models that use the OpenAI API
            OPENAI_API_KEY
        # Azure OpenAI API
            AZURE_OPENAI_API_KEY
            AZURE_OPENAI_ENDPOINT
            OPENAI_API_VERSION
        # Anthropic API
            ANTHROPIC_API_KEY
        # Google Gen AI API
            GEMINI_API_KEY
        # Amazon Bedrock API
            AWS_API_KEYS
            # this variable should contain both "aws_access_key_id" and 
            # "aws_secret_access_key" separated by a space
        # Mistral API
            MISTRAL_API_KEY
        ``````
    
- You want all of this functionality in both sync and async applications
    - All LLMCaller objects have two call methods: ```call``` and ```acall```
    - ChatController object has two chat methods: ```chat``` and ```achat```

- You want to use well-tested code
    - FabData-LLM currently has over 80% branch coverage in pytest

## How to use

### Installation

The easiest way to install is to use the dedicated conda environment provided in ```environment.yml```:

```shell
conda env create -f environment.yml
```

This creates a conda environment called ```fabdata-llm``` and installs the package along with dependencies inside it. See [Miniconda install](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) for instructions on how to install conda if you don't already have it installed and [condas user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) for a more general guide to using conda.

You can also install into any existing python (3.10, or 3.11) environment by running:

```shell
pip install .
```

from inside the repository directory. If you encounter any dependency clashes then please feel free to report in the discussion and we will see if we can loosen any of the dependency restrictions.

### Configuration

The package comes with a base model configuration which can be extended by user-provided custom configurations. You can get the base model configuration dictionary by:

```python
from fdllm.sysutils import list_models

models = list_models(full_info=True, base_only=True)
```

There are 5 categories of model (specified with the `Api_Interface` option):

- OpenAI
- Anthropic
- AzureOpenAI
- AzureMistralAI
- VertexAI

A full template for a custom model configuration file can be found here: [model template](model_config_template.yaml). All fields except for those marked required are optional and can be omitted. Once created, models in the custom configuration can be added to the set of useable models by:

```python
from fdllm.sysutils import register_models

register_models("/path/to/my/custom/models.yaml")
```

The configuration file is particularly useful for configuring other inference providers that provide OpenAI API compatible endpoints. For example, providers like Fireworks.ai or OpenRouter can be easily configured. It is also useful for supporting different Azure OpenAI deployments, with different endpoints and api keys. If you are only interested in setting global api keys for different providers, then the base model configuration is likely enough for your needs, and you can instead set the following environment variables:

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
```

To avoid storing API keys in config files, you can also have multiple keys stored in different environment variables (for example in a `.env` file), and specify the environment variable for each model with the `Api_Key_Env_Var` parameter (i.e. having a different environment variable for each Azure deployment).

### Usage

The [sandbox](./sandbox) folder contains examples of different use cases.