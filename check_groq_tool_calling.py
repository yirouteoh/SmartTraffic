
"""
SmartTraffic - Groq Tool Calling Test

Model: Qwen3.8-27B
Provider: Groq
Purpose: Verify that the LLM can select a Python tool.
"""

import os
import json

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "qwen/qwen3.8-27b"

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )


# ==========================================
# 2. INITIALISE GROQ CLIENT
# ==========================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# 3. DEFINE AVAILABLE PYTHON TOOLS
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "traffic_priority",
            "description": (
                "Evaluate traffic conditions using vehicle "
                "counts and waiting times. Determine whether "
                "North-South or East-West should receive "
                "higher traffic priority."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_counts": {
                        "type": "object",
                        "properties": {
                            "north": {"type": "integer"},
                            "south": {"type": "integer"},
                            "east": {"type": "integer"},
                            "west": {"type": "integer"},
                        },
                        "required": [
                            "north",
                            "south",
                            "east",
                            "west"
                        ],
                        "additionalProperties": False,
                    },
                    "waiting_times": {
                        "type": "object",
                        "properties": {
                            "north": {"type": "number"},
                            "south": {"type": "number"},
                            "east": {"type": "number"},
                            "west": {"type": "number"},
                        },
                        "required": [
                            "north",
                            "south",
                            "east",
                            "west"
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "vehicle_counts",
                    "waiting_times"
                ],
                "additionalProperties": False,
            },
        },
    }
]


# ==========================================
# 4. DEFINE CONVERSATION
# ==========================================

messages = [
    {
        "role": "system",
        "content": (
            "You are SmartTraffic, an intelligent traffic "
            "management assistant. "
            "When the user requests traffic priority analysis "
            "and provides all four vehicle counts and waiting "
            "times, you must call the available tool named "
            "traffic_priority. "
            "Use exactly the registered tool name. "
            "Do not invent missing traffic information."
        ),
    },
    {
        "role": "user",
        "content": (
            "Analyse this four-way intersection. "
            "North has 25 vehicles waiting 90 seconds. "
            "South has 18 vehicles waiting 65 seconds. "
            "East has 8 vehicles waiting 25 seconds. "
            "West has 5 vehicles waiting 15 seconds. "
            "Determine which phase should receive priority."
        ),
    },
]


# ==========================================
# 5. SEND REQUEST TO GROQ
# ==========================================

print("Connecting to Groq...")
print("Model:", MODEL)

try:

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
        max_completion_tokens=500,
    )

    message = response.choices[0].message

    print("\nMODEL RESPONSE:")

    # Check whether the LLM selected a tool.
    if message.tool_calls:

        for call in message.tool_calls:

            print("\nTool name:", call.function.name)

            arguments = json.loads(
                call.function.arguments
            )

            print(
                "Arguments:",
                json.dumps(arguments, indent=4)
            )

            # Verify the returned function name.
            if call.function.name == "traffic_priority":

                print(
                    "\nSUCCESS: Correct tool selected!"
                )

            else:

                print(
                    "\nWARNING: Unexpected tool name."
                )

    else:

        print(message.content)

        print(
            "\nNo tool call was returned in this attempt."
        )

except Exception as error:

    print("\nGROQ REQUEST FAILED:")

    print(type(error).__name__, str(error))
