
import os
import sys
import json

from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import AVAILABLE_TOOLS, execute_tool
from agent.llm_agent import SmartTrafficAgent

load_dotenv()

MODEL = "gemini-3.1-flash-lite"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

agent = SmartTrafficAgent()

# Convert existing SmartTraffic tools to Gemini format.
declarations = []

for tool in AVAILABLE_TOOLS:
    function = tool["function"]

    declarations.append(
        types.FunctionDeclaration(
            name=function["name"],
            description=function["description"],
            parameters_json_schema=function["parameters"],
        )
    )

gemini_tools = [
    types.Tool(function_declarations=declarations)
]

vehicle_counts = {
    "north": 25,
    "south": 18,
    "east": 8,
    "west": 5,
}

waiting_times = {
    "north": 90,
    "south": 65,
    "east": 25,
    "west": 15,
}

previous_phase = None

traffic_data = {
    "vehicle_counts": vehicle_counts,
    "waiting_times": waiting_times,
}

prompt = f"""
You are SmartTraffic.

Select traffic_priority first.

Use exactly:
vehicle_counts = {json.dumps(vehicle_counts)}
waiting_times = {json.dumps(waiting_times)}
previous_phase = null

Do not calculate anything yourself.
You must call traffic_priority.
"""

print("\nSTEP 1: GEMINI SELECTS TRAFFIC PRIORITY")

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

if not response.function_calls:
    raise RuntimeError("Gemini did not select traffic_priority.")

call1 = response.function_calls[0]

if call1.name != "traffic_priority":
    raise RuntimeError("Incorrect first function.")

arguments1 = dict(call1.args)

agent._validate_tool_arguments(
    function_name="traffic_priority",
    arguments=arguments1,
    traffic_data=traffic_data,
    decision_result=None,
    priority_arguments=None,
    previous_phase=previous_phase,
)

priority_result = execute_tool(
    function_name="traffic_priority",
    arguments=arguments1,
)

agent._validate_function_result(
    function_name="traffic_priority",
    result=priority_result,
    decision_result=None,
)

print("Function:", call1.name)
print("Python result:")
print(json.dumps(priority_result, indent=4))

print("\nSTEP 2: RETURN RESULT TO GEMINI")

model_content = response.candidates[0].content

function_response = types.Content(
    role="user",
    parts=[
        types.Part.from_function_response(
            name=call1.name,
            response=priority_result,
        )
    ],
)

second_response = client.models.generate_content(
    model=MODEL,
    contents=[
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
        model_content,
        function_response,
    ],
    config=types.GenerateContentConfig(
        tools=gemini_tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["green_time"],
            )
        ),
        temperature=0.1,
        max_output_tokens=500,
    ),
)

if not second_response.function_calls:
    raise RuntimeError("Gemini did not select green_time.")

call2 = second_response.function_calls[0]

if call2.name != "green_time":
    raise RuntimeError("Incorrect second function.")

arguments2 = dict(call2.args)

print("\nSTEP 3: GEMINI SELECTS GREEN TIME")
print("Function:", call2.name)

priority_arguments = {
    "vehicle_counts": vehicle_counts.copy(),
    "waiting_times": waiting_times.copy(),
}

agent._validate_tool_arguments(
    function_name="green_time",
    arguments=arguments2,
    traffic_data=traffic_data,
    decision_result=priority_result,
    priority_arguments=priority_arguments,
    previous_phase=previous_phase,
)

green_result = execute_tool(
    function_name="green_time",
    arguments=arguments2,
)

agent._validate_function_result(
    function_name="green_time",
    result=green_result,
    decision_result=priority_result,
)

print("Python result:")
print(json.dumps(green_result, indent=4))

print("\nALL GEMINI WORKFLOW TESTS PASSED!")
print("Both Python functions executed successfully.")
