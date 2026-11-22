
import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.tools import AVAILABLE_TOOLS

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-3.1-flash-lite"

# Convert the existing OpenAI-style tool definitions
# into Gemini function declarations.
function_declarations = []

for tool in AVAILABLE_TOOLS:
    function = tool["function"]

    function_declarations.append(
        types.FunctionDeclaration(
            name=function["name"],
            description=function["description"],
            parameters_json_schema=function["parameters"],
        )
    )

gemini_tools = [
    types.Tool(
        function_declarations=function_declarations
    )
]

prompt = """
You are SmartTraffic.

Select the traffic_priority Python function.

Use exactly these values:

Vehicle counts:
North: 25
South: 18
East: 8
West: 5

Waiting times:
North: 90
South: 65
East: 25
West: 15

Previous phase: null

You must select traffic_priority.
Do not calculate the result yourself.
"""

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=gemini_tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["traffic_priority"],
            )
        ),
        temperature=0.1,
        max_output_tokens=500,
    ),
)

print("Gemini function-calling test")
print("Model:", MODEL)

if response.function_calls:
    for function_call in response.function_calls:
        print("\nFunction selected:", function_call.name)
        print("Arguments:")
        print(json.dumps(function_call.args, indent=4))
else:
    print("\nNo function was selected.")
    print(response.text)
