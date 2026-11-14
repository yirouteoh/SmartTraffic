
"""
SmartTraffic - Sequential LLM Agent

Model: Qwen3.8-27B
Provider: Groq

The LLM decides which available tool to call.
Python validates and executes every authorised call.
"""

import os
import json

from dotenv import load_dotenv
from groq import Groq

from agent.tools import (
    AVAILABLE_TOOLS,
    execute_tool,
)


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "qwen/qwen3.8-27b"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")


client = Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """
You are SmartTraffic, an intelligent traffic management agent.

You analyse traffic information and use Python tools
to obtain actual traffic calculations.

Rules:
1. Never invent traffic information.
2. Use the exact registered tool name.
3. Use only the tool currently available to you.
4. Use actual Python results in your explanation.
5. If information is missing, admit that you cannot calculate.
6. Never claim to control real-world traffic lights.
7. All recommendations are for simulation only.

The traffic_priority tool evaluates vehicle demand,
waiting times and fairness.

The green_time tool calculates green-light duration
for a selected NS or EW phase.
"""


class SmartTrafficAgent:

    def __init__(self):
        self.client = client
        self.model = MODEL

    def process_request(self, user_input: str) -> dict:

        if not isinstance(user_input, str):
            raise ValueError("User input must be text.")

        if not user_input.strip():
            raise ValueError("User input cannot be empty.")

        if len(user_input) > 5000:
            raise ValueError("User input is too long.")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        execution_logs = []

        # Tools are exposed sequentially.
        tool_stages = [
            "traffic_priority",
            "green_time",
        ]

        for stage in tool_stages:

            current_tool = next(
                tool for tool in AVAILABLE_TOOLS
                if tool["function"]["name"] == stage
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[current_tool],
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.1,
                max_completion_tokens=1200,
            )

            message = response.choices[0].message

            # Model decides not to call the available tool.
            if not message.tool_calls:

                return {
                    "success": len(execution_logs) > 0,
                    "response": message.content or (
                        "The agent could not complete the request."
                    ),
                    "tool_calls": execution_logs,
                    "model": self.model,
                }

            # Preserve the assistant tool-call message.
            messages.append(
                message.model_dump(exclude_none=True)
            )

            for tool_call in message.tool_calls:

                function_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                arguments = None

                try:

                    if function_name != stage:
                        raise ValueError(
                            "Tool is not permitted in the current stage."
                        )

                    arguments = json.loads(raw_arguments)

                    result = execute_tool(
                        function_name,
                        arguments,
                    )

                    status = "SUCCESS"

                except (ValueError, TypeError, json.JSONDecodeError) as error:

                    result = {"error": str(error)}
                    status = "REJECTED"

                except Exception as error:

                    result = {
                        "error": "Python execution failed: " + str(error)
                    }
                    status = "ERROR"

                execution_logs.append({
                    "function_name": function_name,
                    "arguments": (
                        arguments if arguments is not None
                        else raw_arguments
                    ),
                    "result": result,
                    "status": status,
                })

                # Return actual Python result to the LLM.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

                if status != "SUCCESS":

                    return {
                        "success": False,
                        "response": (
                            "The requested Python function was rejected. "
                            "The agent cannot safely continue."
                        ),
                        "tool_calls": execution_logs,
                        "model": self.model,
                    }

        # Final LLM response without exposing more tools.
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=1200,
        )

        final_message = final_response.choices[0].message

        return {
            "success": True,
            "response": final_message.content or (
                "The calculations were completed, but no explanation was generated."
            ),
            "tool_calls": execution_logs,
            "model": self.model,
        }
