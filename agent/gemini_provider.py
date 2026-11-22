
"""
SmartTraffic - Gemini Fallback Provider

Provider: Google Gemini
Model: gemini-3.1-flash-lite

Responsibilities:
1. Connect to Gemini API.
2. Select authorised Python functions.
3. Follow the required sequential workflow.
4. Receive actual Python calculation results.
5. Generate the final AI explanation.

Python execution and validation remain in llm_agent.py.
"""

import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.tools import AVAILABLE_TOOLS


# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-3.1-flash-lite"

MAX_OUTPUT_TOKENS = 500


# =====================================================
# 2. INITIALISE GEMINI CLIENT
# =====================================================

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# =====================================================
# 3. CONVERT EXISTING PYTHON TOOLS
# =====================================================

FUNCTION_DECLARATIONS = []

for tool in AVAILABLE_TOOLS:

    function = tool["function"]

    FUNCTION_DECLARATIONS.append(
        types.FunctionDeclaration(
            name=function["name"],
            description=function["description"],
            parameters_json_schema=function["parameters"],
        )
    )


GEMINI_TOOLS = [
    types.Tool(
        function_declarations=FUNCTION_DECLARATIONS
    )
]


# =====================================================
# 4. GEMINI FALLBACK PROVIDER
# =====================================================

class GeminiFallbackProvider:

    def __init__(self):

        """Initialise Gemini fallback provider."""

        if client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = client
        self.model = GEMINI_MODEL

    # =================================================
    # 5. SELECT PYTHON FUNCTION
    # =================================================

    def select_tool(
        self,
        function_name,
        user_input,
        previous_phase,
        decision_result=None,
    ):

        """
        Ask Gemini to select exactly one authorised
        Python function for the current workflow stage.
        """

        if function_name not in {
            "traffic_priority",
            "green_time",
        }:
            raise ValueError(
                "Unauthorised Gemini function request."
            )

        context = {
            "user_request": user_input,
            "previous_phase": previous_phase,
            "completed_priority_result": decision_result,
            "required_function": function_name,
        }

        prompt = f"""
You are SmartTraffic, an intelligent traffic decision agent.

Your task is to select the required Python function.

REQUIRED FUNCTION:
{function_name}

WORKFLOW RULES:

1. Select only the required function.
2. Use the exact traffic information from the user request.
3. Never invent vehicle counts or waiting times.
4. Never invent Python calculation results.
5. Never calculate traffic priority yourself.
6. Never calculate green time yourself.
7. For green_time, use the actual selected_phase
   returned by the Python traffic_priority function.
8. Preserve the supplied previous_phase.
9. Do not select any other function.

CURRENT WORKFLOW CONTEXT:

{json.dumps(context, indent=2)}

Select the required Python function now.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=GEMINI_TOOLS,
                tool_config=types.ToolConfig(
                    function_calling_config=(
                        types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=[
                                function_name
                            ],
                        )
                    )
                ),
                temperature=0.1,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

        if not response.function_calls:

            raise RuntimeError(
                "Gemini did not select a Python function."
            )

        if len(response.function_calls) != 1:

            raise RuntimeError(
                "Gemini returned multiple function calls."
            )

        function_call = response.function_calls[0]

        if function_call.name != function_name:

            raise ValueError(
                "Gemini selected an unexpected function."
            )

        return {
            "function_name": function_call.name,
            "arguments": dict(function_call.args),
            "provider": "Gemini",
            "model": self.model,
        }

    # =================================================
    # 6. GENERATE FINAL EXPLANATION
    # =================================================

    def generate_explanation(
        self,
        user_input,
        previous_phase,
        decision_result,
        timing_result,
    ):

        """
        Generate an explanation using only actual
        Python calculation results.
        """

        actual_results = {
            "traffic_priority": decision_result,
            "green_time": timing_result,
        }

        prompt = f"""
You are SmartTraffic.

Explain the traffic analysis using ONLY the actual
Python calculation results below.

USER REQUEST:
{user_input}

PREVIOUS PHASE:
{previous_phase}

ACTUAL PYTHON RESULTS:
{json.dumps(actual_results, indent=2)}

RULES:

1. Do not invent or modify calculation results.
2. Explain the selected traffic phase.
3. Explain the recommended green duration.
4. Briefly explain the priority results.
5. State that this is a simulation recommendation.
6. Do not claim real traffic lights were activated.
7. Keep the explanation clear and concise.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

        if not response.text:

            raise RuntimeError(
                "Gemini returned an empty explanation."
            )

        return response.text.strip()
