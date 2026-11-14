
"""
SmartTraffic - Python Tool Registry

Responsibilities:
1. Define tools available to the LLM.
2. Maintain an allowlist of authorised functions.
3. Use core.validation for argument validation.
4. Execute authorised Python functions.
"""

import json

from typing import Any

from core.traffic_engine import (
    check_traffic_priority,
    calculate_green_time,
)

from core.validation import (
    validate_traffic_arguments,
    validate_green_time_arguments,
    parse_tool_arguments,
)


# ==========================================
# 1. CONSTANTS
# ==========================================

TRAFFIC_DIRECTIONS = [
    "north",
    "south",
    "east",
    "west",
]


# ==========================================
# 2. TOOL SCHEMAS
# ==========================================

def traffic_data_schema():
    """Define the vehicle-count schema."""

    return {
        "type": "object",
        "properties": {
            direction: {"type": "integer"}
            for direction in TRAFFIC_DIRECTIONS
        },
        "required": TRAFFIC_DIRECTIONS,
        "additionalProperties": False,
    }


def waiting_time_schema():
    """Define the waiting-time schema."""

    return {
        "type": "object",
        "properties": {
            direction: {"type": "number"}
            for direction in TRAFFIC_DIRECTIONS
        },
        "required": TRAFFIC_DIRECTIONS,
        "additionalProperties": False,
    }


# ==========================================
# 3. TOOLS AVAILABLE TO THE LLM
# ==========================================

AVAILABLE_TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "traffic_priority",
            "description": (
                "Evaluate vehicle counts and waiting times "
                "to select the highest-priority NS or EW phase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_counts": traffic_data_schema(),
                    "waiting_times": waiting_time_schema(),
                },
                "required": [
                    "vehicle_counts",
                    "waiting_times",
                ],
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "green_time",
            "description": (
                "Calculate the recommended green-light duration "
                "for the selected NS or EW traffic phase. "
                "Provide exactly one phase field: phase or phaze."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "enum": ["NS", "EW"],
                    },
                    "phaze": {
                        "type": "string",
                        "enum": ["NS", "EW"],
                    },
                    "vehicle_counts": traffic_data_schema(),
                    "waiting_times": waiting_time_schema(),
                },
                "required": [
                    "vehicle_counts",
                    "waiting_times",
                ],
                "additionalProperties": False,
            },
        },
    },
]


# ==========================================
# 4. AUTHORISED FUNCTION ALLOWLIST
# ==========================================

ALLOWED_FUNCTIONS = {
    "traffic_priority",
    "green_time",
}


# ==========================================
# 5. TOOL EXECUTION
# ==========================================

def execute_tool(
    function_name: str,
    arguments: Any,
) -> dict:
    """
    Validate and execute an authorised Python function.

    The LLM cannot execute arbitrary Python code.
    Only functions in the allowlist are permitted.
    """

    # Step 1: Check function name.
    if function_name not in ALLOWED_FUNCTIONS:
        raise ValueError(
            f"Unauthorised function: {function_name}"
        )

    # Step 2: Parse JSON arguments.
    arguments = parse_tool_arguments(arguments)

    # Step 3: Execute traffic priority.
    if function_name == "traffic_priority":

        validate_traffic_arguments(arguments)

        result = check_traffic_priority(
            vehicle_counts=arguments["vehicle_counts"],
            waiting_times=arguments["waiting_times"],
        )

        return result

    # Step 4: Execute green-time calculation.
    if function_name == "green_time":

        validate_green_time_arguments(arguments)

        phase = arguments.get(
            "phase",
            arguments.get("phaze"),
        )

        result = calculate_green_time(
            selected_phase=phase,
            vehicle_counts=arguments["vehicle_counts"],
            waiting_times=arguments["waiting_times"],
        )

        return result

    raise ValueError("Unsupported function.")
