
"""
SmartTraffic - Traffic Decision Engine

This module:
1. Evaluates traffic priority for North-South and East-West phases.
2. Calculates recommended green-light duration.
3. Uses previous-phase information to alternate phases when scores are tied.
"""

from typing import Dict, Any, Optional


DIRECTIONS = ("north", "south", "east", "west")

PHASES = {
    "NS": ("north", "south"),
    "EW": ("east", "west"),
}

MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 60


def _validate_traffic_data(
    vehicle_counts: Dict[str, int],
    waiting_times: Dict[str, float],
) -> None:
    """Validate traffic information before calculation."""

    if not isinstance(vehicle_counts, dict):
        raise ValueError("Vehicle counts must be a dictionary.")

    if not isinstance(waiting_times, dict):
        raise ValueError("Waiting times must be a dictionary.")

    if set(vehicle_counts.keys()) != set(DIRECTIONS):
        raise ValueError("Vehicle counts must contain all four directions.")

    if set(waiting_times.keys()) != set(DIRECTIONS):
        raise ValueError("Waiting times must contain all four directions.")

    for direction in DIRECTIONS:
        count = vehicle_counts[direction]
        waiting = waiting_times[direction]

        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"Invalid vehicle count for {direction}.")

        if count < 0:
            raise ValueError(
                f"Vehicle count for {direction} cannot be negative."
            )

        if isinstance(waiting, bool) or not isinstance(
            waiting, (int, float)
        ):
            raise ValueError(f"Invalid waiting time for {direction}.")

        if waiting < 0:
            raise ValueError(
                f"Waiting time for {direction} cannot be negative."
            )


def check_traffic_priority(
    vehicle_counts: Dict[str, int],
    waiting_times: Dict[str, float],
    cycles_since_served: Optional[Dict[str, int]] = None,
    previous_phase: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Determine which traffic phase should receive priority.

    Priority formula:
    P = 0.45Q + 0.35W + 0.20F

    Q = Normalised vehicle demand
    W = Normalised waiting-time pressure
    F = Normalised fairness score

    When priority scores are equal, the phase alternates
    according to the previously served phase.
    """

    _validate_traffic_data(vehicle_counts, waiting_times)

    # Validate previous phase.
    if previous_phase not in (None, "NS", "EW"):
        raise ValueError(
            "Previous phase must be NS, EW or None."
        )

    # Initialise fairness information when unavailable.
    if cycles_since_served is None:
        cycles_since_served = {
            "NS": 0,
            "EW": 0,
        }

    if not isinstance(cycles_since_served, dict):
        raise ValueError(
            "Cycles since served must be a dictionary."
        )

    if set(cycles_since_served.keys()) != {"NS", "EW"}:
        raise ValueError(
            "Fairness data must contain NS and EW."
        )

    for phase, cycles in cycles_since_served.items():

        if isinstance(cycles, bool) or not isinstance(cycles, int):
            raise ValueError(
                f"Invalid cycle count for {phase}."
            )

        if cycles < 0:
            raise ValueError(
                f"Cycle count for {phase} cannot be negative."
            )

    # Calculate vehicle demand for both phases.
    phase_demand = {
        phase: sum(
            vehicle_counts[direction]
            for direction in directions
        )
        for phase, directions in PHASES.items()
    }

    total_demand = sum(phase_demand.values())

    # Calculate average waiting time.
    phase_waiting = {
        phase: sum(
            waiting_times[direction]
            for direction in directions
        ) / len(directions)
        for phase, directions in PHASES.items()
    }

    # Normalise vehicle demand.
    demand_scores = {
        phase: (
            phase_demand[phase] / total_demand
            if total_demand > 0 else 0.5
        )
        for phase in PHASES
    }

    # Normalise waiting-time pressure.
    maximum_wait = max(phase_waiting.values())

    waiting_scores = {
        phase: (
            phase_waiting[phase] / maximum_wait
            if maximum_wait > 0 else 0.5
        )
        for phase in PHASES
    }

    # Normalise fairness.
    maximum_cycles = max(cycles_since_served.values())

    fairness_scores = {
        phase: (
            cycles_since_served[phase] / maximum_cycles
            if maximum_cycles > 0 else 0.5
        )
        for phase in PHASES
    }

    # Calculate weighted priority scores.
    priority_scores = {}

    for phase in PHASES:
        priority_scores[phase] = round(
            0.45 * demand_scores[phase]
            + 0.35 * waiting_scores[phase]
            + 0.20 * fairness_scores[phase],
            4,
        )

    # Select the phase with the higher priority score.
    if priority_scores["NS"] > priority_scores["EW"]:

        selected_phase = "NS"
        selection_reason = "NS achieved the higher priority score."

    elif priority_scores["EW"] > priority_scores["NS"]:

        selected_phase = "EW"
        selection_reason = "EW achieved the higher priority score."

    else:
        # Equal scores: alternate from the previous phase.
        selected_phase = (
            "EW" if previous_phase == "NS" else "NS"
        )

        selection_reason = (
            f"Both phases had equal priority scores. "
            f"The previous phase was {previous_phase}. "
            f"SmartTraffic selected {selected_phase} "
            f"to maintain phase alternation."
            if previous_phase is not None
            else
            "Both phases had equal priority scores. "
            "NS was selected as the default initial phase."
        )

    return {
        "selected_phase": selected_phase,
        "priority_scores": priority_scores,
        "vehicle_demand": phase_demand,
        "average_waiting_time": phase_waiting,
        "demand_scores": demand_scores,
        "waiting_scores": waiting_scores,
        "fairness_scores": fairness_scores,
        "previous_phase": previous_phase,
        "selection_reason": selection_reason,
        "explanation": (
            f"Phase {selected_phase} was selected. "
            f"{selection_reason}"
        ),
    }


def calculate_green_time(
    selected_phase: str,
    vehicle_counts: Dict[str, int],
    waiting_times: Dict[str, float],
    min_green: int = MIN_GREEN_TIME,
    max_green: int = MAX_GREEN_TIME,
) -> Dict[str, Any]:
    """
    Calculate recommended green-light duration.

    The result is restricted to the configured safety limits.
    """

    _validate_traffic_data(vehicle_counts, waiting_times)

    if selected_phase not in PHASES:
        raise ValueError(
            "Selected phase must be NS or EW."
        )

    if (
        isinstance(min_green, bool)
        or not isinstance(min_green, int)
        or isinstance(max_green, bool)
        or not isinstance(max_green, int)
    ):
        raise ValueError(
            "Green-time limits must be integers."
        )

    if min_green <= 0 or max_green < min_green:
        raise ValueError(
            "Invalid green-time limits."
        )

    directions = PHASES[selected_phase]

    phase_vehicles = sum(
        vehicle_counts[direction]
        for direction in directions
    )

    maximum_queue = 30
    maximum_wait = 120

    # Calculate normalised queue pressure.
    queue_pressure = min(
        phase_vehicles / maximum_queue,
        1.0,
    )

    # Calculate average waiting time.
    average_wait = sum(
        waiting_times[direction]
        for direction in directions
    ) / len(directions)

    # Calculate normalised waiting-time pressure.
    waiting_pressure = min(
        average_wait / maximum_wait,
        1.0,
    )

    # Calculate combined pressure.
    pressure = (
        0.60 * queue_pressure
        + 0.40 * waiting_pressure
    )

    # Calculate green-light duration.
    raw_green_time = min_green + pressure * (
        max_green - min_green
    )

    green_time = round(raw_green_time)

    # Enforce minimum and maximum limits.
    green_time = max(
        min_green,
        min(green_time, max_green),
    )

    return {
        "selected_phase": selected_phase,
        "green_time_seconds": green_time,
        "queue_pressure": round(queue_pressure, 4),
        "waiting_pressure": round(waiting_pressure, 4),
        "combined_pressure": round(pressure, 4),
        "limits": {
            "minimum": min_green,
            "maximum": max_green,
        },
    }
