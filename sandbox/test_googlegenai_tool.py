#%%
from dotenv import load_dotenv
load_dotenv('../.env')
import os
import logging
logging.basicConfig(level=logging.DEBUG)
from google import genai
from google.genai import types
#%%
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

response = client.models.generate_content(
    model='gemini-2.0-flash-exp', contents='What is your name?'
)
print(response.text)

#%%
def mul(x: float, y: float) -> float:
    """Multiplies two numbers
    """
    return x * y

response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents="What is three times twelve",
    config=types.GenerateContentConfig(tools=[mul],
                                       system_instruction="Use tools for all calculations where possible")
)

print(response.text)


#%%
function = dict(
    name="mul",
    description="multiply two numbers",
    parameters={
      "type": "OBJECT",
      "properties": {
          "x": {
              "type": "NUMBER",
              "description": "the first number",
          },
          "y": {
              "type": "NUMBER",
              "description": "the second number",
          },
      },
      "required": ["x","y"],
    }
)

tool = types.Tool(function_declarations=[function])

prompt = "What is twelve times 6.3?"
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=prompt,
    config=types.GenerateContentConfig(tools=[tool],)
)

response.candidates[0].content.parts[0].function_call

#%%
function_call_part = response.candidates[0].content.parts[0]

function_response = mul(**function_call_part.function_call.args)


function_response_part = types.Part.from_function_response(
    name=function_call_part.function_call.name,
    response={'result': function_response}
)

#%%
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=[
        types.Part.from_text(prompt),
        function_call_part,
        function_response_part,
    ])

response
#%%
call = {

    "model": "models/gemini-2.0-flash-exp",
    "max_completion_tokens": 500,
    "messages": [
        {"role": "system", "content": "Use tools for all calculations where possible."},
        {"role": "user", "content": "what is two times three"},
        {
            "role": "assistant",
            "toolCalls": [
                {
                    "function": {"arguments": "{'x': 2, 'y': 3}", "name": "mul"},
                    "id": "0",
                    "type": "function",
                }
            ],
        },
        {"role": "tool", "tool_call_id": "0", "name": "mul", "content": "6.0000"},
    ],
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
def get_current_weather(location: str,) -> int:
  """Returns the current weather.

  Args:
    location: The city and state, e.g. San Francisco, CA
  """
  return 'sunny'

response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(tools=[get_current_weather],)
)

response.text

#%%
function = dict(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
      "type": "OBJECT",
      "properties": {
          "location": {
              "type": "STRING",
              "description": "The city and state, e.g. San Francisco, CA",
          },
      },
      "required": ["location"],
    }
)

tool = types.Tool(function_declarations=[function])


response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(tools=[tool],)
)

response.candidates[0].content.parts[0].function_call

function_call_part = response.candidates[0].content.parts[0]

function_response = get_current_weather(**function_call_part.function_call.args)


function_response_part = types.Part.from_function_response(
    name=function_call_part.function_call.name,
    response={'result': function_response}
)


#%%
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=[
        types.Part.from_text("What is the weather like in Boston?"),
        function_call_part,
        function_response_part,
    ],
    config=types.GenerateContentConfig(tools=[tool],))

response

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
