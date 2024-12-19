#%%
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('../.env')
import os
#%%
client = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

call = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "mul",
                "description": "Multiply 2 numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                },
            },
        }
    ],
    "model": "models/gemini-2.0-flash-exp",
    "max_completion_tokens": 500,
    "messages": [
        {"role": "system", "content": "Use tools for all calculations where possible."},
        {"role": "user", "content": "what is two times three"},
        # {
        #     "role": "assistant",
        #     "toolCalls": [
        #         {
        #             "function": {"arguments": "{'x': 2, 'y': 3}", "name": "mul"},
        #             "id": "0",
        #             "type": "function",
        #         }
        #     ],
        # },
        # {"role": "tool", "tool_call_id": "0", "name": "mul", "content": "6.0000"},
    ],
}
# from gemini: [{'function': {'arguments': '{"x":2,"y":3}', 'name': 'mul'}, 'id': '0', 'type': 'function'}]
response = client.chat.completions.create(
  model=call["model"],
  messages=call["messages"],
  tools=call["tools"],
  tool_choice="auto"
)

print(response)

#%%
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. Chicago, IL",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]

call = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "mul",
                "description": "Multiply 2 numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                },
            },
        }
    ],
    "model": "models/gemini-2.0-flash-exp",
    "max_completion_tokens": 500,
    "messages": [
        {"role": "system", "content": "Use tools for all calculations where possible."},
        {"role": "user", "content": "what is two times three"},
        {
            "role": "assistant",
            "toolCalls": [
                {
                    "id": "0",
                    "type": "function",
                    "function": {"arguments": "{'x': 2, 'y': 3}", "name": "mul"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "0", "name": "mul", "content": "6.0000"},
    ],
}

messages = [{"role": "user", "content": "What's the weather like in Chicago today in C?"}]
response = client.chat.completions.create(
  model="gemini-1.5-flash",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(response)

# %%
