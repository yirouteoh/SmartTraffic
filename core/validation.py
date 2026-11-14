
"""
SmartTraffic - Input and Argument Validation

Responsibilities:
1. Validate user input.
2. Validate traffic vehicle counts.
3. Validate waiting times.
4. Validate traffic phase arguments.
5. Reject malformed or unsupported data.
"""

import math
import json


# ==========================================
# 1. CONSTANTS
# ==========================================

VALID_DIRECTIONS = {
    "north",
    "south",
    "east",
    "west",
}

VALID_PHASES = {
    "NS",
    "EW",
}

MAX_VEHICLES = 10000
MAX_WAITING_TIME = 86400
MAX_INPUT_LENGTH = 5000


# ==========================================
# 2. USER INPUT VALIDATION
# ==========================================

def validate_user_input(user_input: str) -> None:
    """Validate the user's text input."""

    if not isinstance(user_input, str):
        raise ValueError("User input must be text.")

    if not user_input.strip():
        raise ValueError("User input cannot be empty.")

    if len(user_input) > MAX_INPUT_LENGTH:
        raise ValueError("User input exceeds the permitted length.")


# ==========================================
# 3. VEHICLE COUNT VALIDATION
# ==========================================

def validate_vehicle_counts(vehicle_counts: dict) -> None:
    """Validate vehicle counts for all four directions."""

    if not isinstance(vehicle_counts, dict):
        raise ValueError("Vehicle counts must be a dictionary.")

    if set(vehicle_counts.keys()) != VALID_DIRECTIONS:
        raise ValueError(
            "Vehicle counts must contain north, south, east and west."
        )

    for direction, count in vehicle_counts.items():

        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(
                f"Vehicle count for {direction} must be an integer."
            )

        if not 0 <= count <= MAX_VEHICLES:
            raise ValueError(
                f"Vehicle count for {direction} is outside the permitted range."
            )


# ==========================================
# 4. WAITING TIME VALIDATION
# ==========================================

def validate_waiting_times(waiting_times: dict) -> None:
    """Validate waiting times for all four directions."""

    if not isinstance(waiting_times, dict):
        raise ValueError("Waiting times must be a dictionary.")

    if set(waiting_times.keys()) != VALID_DIRECTIONS:
        raise ValueError(
            "Waiting times must contain north, south, east and west."
        )

    for direction, waiting in waiting_times.items():

        if isinstance(waiting, bool) or not isinstance(waiting, (int, float)):
            raise ValueError(
                f"Waiting time for {direction} must be numeric."
            )

        if not math.isfinite(waiting):
            raise ValueError(
                f"Waiting time for {direction} must be finite."
            )

        if not 0 <= waiting <= MAX_WAITING_TIME:
            raise ValueError(
                f"Waiting time for {direction} is outside the permitted range."
            )


# ==========================================
# 5. COMPLETE TRAFFIC ARGUMENT VALIDATION
# ==========================================

def validate_traffic_arguments(arguments: dict) -> None:
    """Validate all traffic information."""

    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a dictionary.")

    required_fields = {
        "vehicle_counts",
        "waiting_times",
    }

    if set(arguments.keys()) != required_fields:
        raise ValueError(
            "Missing or unexpected traffic information."
        )

    validate_vehicle_counts(
        arguments["vehicle_counts"]
    )

    validate_waiting_times(
        arguments["waiting_times"]
    )


# ==========================================
# 6. GREEN TIME ARGUMENT VALIDATION
# ==========================================

def validate_green_time_arguments(arguments: dict) -> None:
    """Validate arguments for green-time calculation."""

    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a dictionary.")

    required_fields = {
        "phase",
        "vehicle_counts",
        "waiting_times",
    }

    allowed_fields = required_fields | {"phaze"}

    if not required_fields.issubset(arguments):
        # Accept the known spelling alias only when phase is absent.
        if "phase" not in arguments and "phaze" in arguments:
            pass
        else:
            raise ValueError("Required green-time information is missing.")

    if not set(arguments.keys()).issubset(allowed_fields):
        raise ValueError("Unexpected green-time arguments.")

    if "phase" in arguments and "phaze" in arguments:
        raise ValueError("Only one phase field is permitted.")

    phase = arguments.get(
        "phase",
        arguments.get("phaze"),
    )

    if not isinstance(phase, str) or phase not in VALID_PHASES:
        raise ValueError("Phase must be NS or EW.")

    traffic_arguments = {
        "vehicle_counts": arguments["vehicle_counts"],
        "waiting_times": arguments["waiting_times"],
    }

    validate_traffic_arguments(traffic_arguments)


# ==========================================
# 7. JSON ARGUMENT PARSING
# ==========================================

def parse_tool_arguments(arguments):
    """Parse JSON arguments returned by the LLM."""

    if isinstance(arguments, str):

        try:
            arguments = json.loads(arguments)

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON arguments.")

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    return arguments
