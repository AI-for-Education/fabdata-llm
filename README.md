FabData-LLM is a set of high-level abstractions around various LLM API providers. It is written in python and currently covers OpenAI, Azure OpenAI, and Anthropic APIs.

Why you might consider using this:
- You want to create a chatbot with stored history and automatic history token management in 3 lines of code:

    ```python
    # GPT 3.5 Turbo
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("gpt-3.5-turbo"))

    print(chatter.chat("Hello there"))
    ```

    ```python
    # Claude 2
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("claude-2"))

    print(chatter.chat("Hello there"))
    ```

    ```python
    # GPT 4 Vision Preview
    from fdllm import get_caller
    from fdllm.chat import ChatController

    chatter = ChatController(Caller=get_caller("gpt-4-vision-preview"))

    ### load images here into a list of PIL Images
    # images : List[PIL.Image.Image]

    print(
        chatter.chat(
            "Hello there, can you compare these images for me",
            images=images,
            detail="high"
        )
    )
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

    - Create plugins with the ```ChatPlugin``` abstract base class. Registered plugins have the ability to intercept and modify both user inputs and Caller responses during chat sessions, make their own LLM API calls, and mutate the state of the ChatController object
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
            my_azure_deployment_1:
                Token_Window: 8192
                Client_Args:
                    azure_endpoint: https://my-endpoint-1.openai.azure.com/
                    api_version: 2023-09-15-preview
                    api_key: my_azure_openai_api_key1
            my_azure_deployment_2:
                Token_Window: 32768
                Client_Args:
                    azure_endpoint: https://my-endpoint-2.openai.azure.com/
                    api_version: 2023-09-15-preview
                    api_key: my_azure_openai_api_key2
        ```
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
            ANTHROPIC_KEY
        ``````
    
- You want to use the latest models from OpenAI, such as ```gpt-4-1106-preview``` and ```gpt-4-vision-preview```
    - FabData-LLM supports the latest versions of both OpenAI's and Anthropics's APIs (1.1.1 and 0.7.0 respectively at the time of writing) and the latest models, including multi-modal models
- You want all of this functionality in both sync and async applications
    - All LLMCaller objects have two call methods: ```call``` and ```acall```
    - ChatController object has two chat methods: ```chat``` and ```achat```