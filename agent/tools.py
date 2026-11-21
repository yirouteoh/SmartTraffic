
"""
SmartTraffic - Python Tool Registry

Responsibilities:
1. Define tools available to the LLM.
2. Maintain an allowlist of authorised functions.
3. Validate tool arguments.
4. Execute authorised Python functions.
5. Pass previous-phase information for fair tie-breaking.
"""

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

VALID_PHASES = {
    "NS",
    "EW",
}


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
                "to select the highest-priority NS or EW phase. "
                "Use previous_phase to alternate traffic phases "
                "when priority scores are equal. "
                "Use null when no previous phase is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_counts": traffic_data_schema(),

                    "waiting_times": waiting_time_schema(),

                    "previous_phase": {
                        "type": ["string", "null"],
                        "enum": ["NS", "EW", None],
                    },
                },
                "required": [
                    "vehicle_counts",
                    "waiting_times",
                    "previous_phase",
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

        # Validate previous phase.
        previous_phase = arguments.get("previous_phase")

        if previous_phase not in (None, "NS", "EW"):
            raise ValueError(
                "Previous phase must be NS, EW or None."
            )

        # Validate traffic information separately.
        traffic_arguments = {
            "vehicle_counts": arguments.get("vehicle_counts"),
            "waiting_times": arguments.get("waiting_times"),
        }

        validate_traffic_arguments(traffic_arguments)

        # Reject unexpected arguments.
        allowed_arguments = {
            "vehicle_counts",
            "waiting_times",
            "previous_phase",
        }

        if set(arguments.keys()) != allowed_arguments:
            raise ValueError(
                "Missing or unexpected traffic-priority arguments."
            )

        # Execute actual Python traffic priority calculation.
        result = check_traffic_priority(
            vehicle_counts=arguments["vehicle_counts"],
            waiting_times=arguments["waiting_times"],
            previous_phase=previous_phase,
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
