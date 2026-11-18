
"""
SmartTraffic - Input and Argument Validation

Responsibilities:
1. Validate user input.
2. Check that all four traffic directions are provided.
3. Check that vehicle counts are explicitly provided.
4. Check that waiting times are explicitly provided.
5. Validate traffic vehicle counts.
6. Validate waiting times.
7. Validate traffic phase arguments.
8. Reject malformed or unsupported data.

Important:
Missing traffic information must never be replaced
automatically with zero.
"""

import math
import json
import re


# ==========================================
# 1. CONSTANTS
# ==========================================

VALID_DIRECTIONS = {
    "north",
    "south",
    "east",
    "west",
}

DIRECTION_ORDER = (
    "north",
    "south",
    "east",
    "west",
)

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
    """Validate the user's general text input."""

    if not isinstance(user_input, str):
        raise ValueError("User input must be text.")

    if not user_input.strip():
        raise ValueError("User input cannot be empty.")

    if len(user_input) > MAX_INPUT_LENGTH:
        raise ValueError(
            "User input exceeds the permitted length."
        )


# ==========================================
# 3. TRAFFIC INFORMATION EXTRACTION
# ==========================================

def validate_traffic_request_data(user_input: str) -> dict:
    """
    Check whether the user explicitly supplied:

    - Vehicle count for North, South, East and West.
    - Waiting time for North, South, East and West.

    Accepts examples such as:

    North: 15 vehicles, waiting 40 seconds

    North has 15 vehicles waiting 40 seconds

    Zero is accepted only when explicitly provided.

    Missing information raises ValueError.
    """

    validate_user_input(user_input)

    text = user_input.lower()

    # Find the position of each directional heading.
    direction_positions = []

    for direction in DIRECTION_ORDER:

        match = re.search(
            rf"\b{direction}\b",
            text,
        )

        if match:
            direction_positions.append(
                (match.start(), direction)
            )

    # Sort directions by their position in the input.
    direction_positions.sort()

    extracted_data = {}

    missing_information = []

    for index, direction in enumerate(DIRECTION_ORDER):

        # Locate this direction in the original request.
        current_match = re.search(
            rf"\b{direction}\b",
            text,
        )

        if current_match is None:

            missing_information.append(
                f"{direction.title()}: direction not provided"
            )

            continue

        start = current_match.end()

        # Stop reading when the next direction begins.
        next_positions = [
            position
            for position, name in direction_positions
            if position > current_match.start()
        ]

        end = (
            min(next_positions)
            if next_positions
            else len(text)
        )

        section = text[start:end]

        # Extract explicitly stated vehicle count.
        vehicle_match = re.search(
            r"\b(\d+)\s*(?:vehicles?|cars?)\b",
            section,
        )

        # Extract explicitly stated waiting time.
        waiting_match = re.search(
            r"\b(?:average\s+waiting\s+time|"
            r"waiting\s+time|waiting|waited)\b"
            r"\s*(?:of|is|:|=)?\s*"
            r"(\d+(?:\.\d+)?)\s*"
            r"(?:seconds?|secs?|s)\b",
            section,
        )

        if vehicle_match is None:

            missing_information.append(
                f"{direction.title()}: vehicle count missing"
            )

        if waiting_match is None:

            missing_information.append(
                f"{direction.title()}: waiting time missing"
            )

        if (
            vehicle_match is not None
            and waiting_match is not None
        ):

            extracted_data[direction] = {
                "vehicle_count": int(
                    vehicle_match.group(1)
                ),
                "waiting_time": float(
                    waiting_match.group(1)
                ),
            }

    if missing_information:

        details = "\n".join(
            f"- {item}"
            for item in missing_information
        )

        raise ValueError(
            "INSUFFICIENT_TRAFFIC_INFORMATION\n\n"
            "SmartTraffic requires vehicle counts and "
            "waiting times for all four directions.\n\n"
            "Missing information:\n"
            f"{details}\n\n"
            "Please provide the missing traffic information "
            "before analysis."
        )

    # Validate extracted numerical values.
    vehicle_counts = {
        direction: extracted_data[direction]["vehicle_count"]
        for direction in DIRECTION_ORDER
    }

    waiting_times = {
        direction: extracted_data[direction]["waiting_time"]
        for direction in DIRECTION_ORDER
    }

    validate_traffic_arguments({
        "vehicle_counts": vehicle_counts,
        "waiting_times": waiting_times,
    })

    return {
        "vehicle_counts": vehicle_counts,
        "waiting_times": waiting_times,
    }


# ==========================================
# 4. VEHICLE COUNT VALIDATION
# ==========================================

def validate_vehicle_counts(vehicle_counts: dict) -> None:
    """Validate vehicle counts for all four directions."""

    if not isinstance(vehicle_counts, dict):
        raise ValueError(
            "Vehicle counts must be a dictionary."
        )

    if set(vehicle_counts.keys()) != VALID_DIRECTIONS:
        raise ValueError(
            "Vehicle counts must contain north, south, "
            "east and west."
        )

    for direction, count in vehicle_counts.items():

        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(
                f"Vehicle count for {direction} "
                "must be an integer."
            )

        if not 0 <= count <= MAX_VEHICLES:
            raise ValueError(
                f"Vehicle count for {direction} "
                "is outside the permitted range."
            )


# ==========================================
# 5. WAITING TIME VALIDATION
# ==========================================

def validate_waiting_times(waiting_times: dict) -> None:
    """Validate waiting times for all four directions."""

    if not isinstance(waiting_times, dict):
        raise ValueError(
            "Waiting times must be a dictionary."
        )

    if set(waiting_times.keys()) != VALID_DIRECTIONS:
        raise ValueError(
            "Waiting times must contain north, south, "
            "east and west."
        )

    for direction, waiting in waiting_times.items():

        if (
            isinstance(waiting, bool)
            or not isinstance(waiting, (int, float))
        ):
            raise ValueError(
                f"Waiting time for {direction} must be numeric."
            )

        if not math.isfinite(waiting):
            raise ValueError(
                f"Waiting time for {direction} must be finite."
            )

        if not 0 <= waiting <= MAX_WAITING_TIME:
            raise ValueError(
                f"Waiting time for {direction} "
                "is outside the permitted range."
            )


# ==========================================
# 6. COMPLETE TRAFFIC ARGUMENT VALIDATION
# ==========================================

def validate_traffic_arguments(arguments: dict) -> None:
    """Validate all traffic information."""

    if not isinstance(arguments, dict):
        raise ValueError(
            "Arguments must be a dictionary."
        )

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
# 7. GREEN TIME ARGUMENT VALIDATION
# ==========================================

def validate_green_time_arguments(arguments: dict) -> None:
    """Validate arguments for green-time calculation."""

    if not isinstance(arguments, dict):
        raise ValueError(
            "Arguments must be a dictionary."
        )

    required_fields = {
        "phase",
        "vehicle_counts",
        "waiting_times",
    }

    allowed_fields = required_fields | {"phaze"}

    if not required_fields.issubset(arguments):

        if not (
            "phase" not in arguments
            and "phaze" in arguments
        ):
            raise ValueError(
                "Required green-time information is missing."
            )

    if not set(arguments.keys()).issubset(allowed_fields):
        raise ValueError(
            "Unexpected green-time arguments."
        )

    if "phase" in arguments and "phaze" in arguments:
        raise ValueError(
            "Only one phase field is permitted."
        )

    phase = arguments.get(
        "phase",
        arguments.get("phaze"),
    )

    if not isinstance(phase, str) or phase not in VALID_PHASES:
        raise ValueError(
            "Phase must be NS or EW."
        )

    traffic_arguments = {
        "vehicle_counts": arguments["vehicle_counts"],
        "waiting_times": arguments["waiting_times"],
    }

    validate_traffic_arguments(traffic_arguments)


# ==========================================
# 8. JSON ARGUMENT PARSING
# ==========================================

def parse_tool_arguments(arguments):
    """Parse JSON arguments returned by the LLM."""

    if isinstance(arguments, str):

        try:
            arguments = json.loads(arguments)

        except json.JSONDecodeError:
            raise ValueError(
                "Invalid JSON arguments."
            )

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be a JSON object."
        )

    return arguments
