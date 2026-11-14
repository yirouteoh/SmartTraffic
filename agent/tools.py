
"""
SmartTraffic - Python Tool Registry

Responsibilities:
1. Define tools available to the LLM.
2. Maintain an allowlist.
3. Validate function arguments.
4. Execute authorised Python functions.
"""

import json
import math

from typing import Any

from core.traffic_engine import (
    check_traffic_priority,
    calculate_green_time,
)


TRAFFIC_DIRECTIONS = [
    "north",
    "south",
    "east",
    "west",
]


# ==========================================
# 1. TOOL SCHEMAS
# ==========================================

def traffic_data_schema():
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
# 2. AVAILABLE TOOLS
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
                "Calculate green-light duration. "
                "Provide the selected traffic phase using "
                "the field phase or its accepted alias phaze."
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
# 3. TRAFFIC ARGUMENT VALIDATION
# ==========================================

def validate_traffic_arguments(arguments: dict) -> None:

    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a dictionary.")

    required_fields = {
        "vehicle_counts",
        "waiting_times",
    }

    if set(arguments.keys()) != required_fields:
        raise ValueError("Missing or unexpected traffic fields.")

    vehicles = arguments["vehicle_counts"]
    waiting = arguments["waiting_times"]

    if not isinstance(vehicles, dict):
        raise ValueError("Vehicle counts must be a dictionary.")

    if not isinstance(waiting, dict):
        raise ValueError("Waiting times must be a dictionary.")

    if set(vehicles.keys()) != set(TRAFFIC_DIRECTIONS):
        raise ValueError("Invalid vehicle count directions.")

    if set(waiting.keys()) != set(TRAFFIC_DIRECTIONS):
        raise ValueError("Invalid waiting time directions.")

    for direction in TRAFFIC_DIRECTIONS:

        count = vehicles[direction]
        wait = waiting[direction]

        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"Invalid vehicle count for {direction}.")

        if not 0 <= count <= 10000:
            raise ValueError(f"Vehicle count outside permitted range: {direction}.")

        if isinstance(wait, bool) or not isinstance(wait, (int, float)):
            raise ValueError(f"Invalid waiting time for {direction}.")

        if not math.isfinite(wait) or not 0 <= wait <= 86400:
            raise ValueError(f"Waiting time outside permitted range: {direction}.")


# ==========================================
# 4. TOOL ALLOWLIST AND EXECUTION
# ==========================================

def execute_tool(
    function_name: str,
    arguments: Any,
) -> dict:

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON arguments.")

    allowed_functions = {
        "traffic_priority",
        "green_time",
    }

    if function_name not in allowed_functions:
        raise ValueError(f"Unauthorised function: {function_name}")

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")

    if function_name == "traffic_priority":

        validate_traffic_arguments(arguments)

        return check_traffic_priority(
            vehicle_counts=arguments["vehicle_counts"],
            waiting_times=arguments["waiting_times"],
        )

    if function_name == "green_time":

        allowed_fields = {
            "phase",
            "phaze",
            "vehicle_counts",
            "waiting_times",
        }

        required_fields = {
            "vehicle_counts",
            "waiting_times",
        }

        if not required_fields.issubset(arguments):
            raise ValueError("Missing required traffic information.")

        if not set(arguments).issubset(allowed_fields):
            raise ValueError("Unexpected green-time arguments.")

        if "phase" in arguments and "phaze" in arguments:
            raise ValueError("Provide only one phase field.")

        phase = arguments.get(
            "phase",
            arguments.get("phaze"),
        )

        if phase not in ("NS", "EW"):
            raise ValueError("Missing or invalid traffic phase.")

        traffic_arguments = {
            "vehicle_counts": arguments["vehicle_counts"],
            "waiting_times": arguments["waiting_times"],
        }

        validate_traffic_arguments(traffic_arguments)

        return calculate_green_time(
            selected_phase=phase,
            vehicle_counts=traffic_arguments["vehicle_counts"],
            waiting_times=traffic_arguments["waiting_times"],
        )

    raise ValueError("Unsupported function.")
