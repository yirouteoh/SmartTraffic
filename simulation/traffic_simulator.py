
"""
SmartTraffic - Traffic Intersection Simulator

Features:
- Four incoming traffic approaches
- North-South and East-West traffic phases
- Automatic random vehicle arrivals
- Low, medium and high traffic density
- Vehicle queue and waiting-time updates
- Traffic signal transitions
- Pedestrian crossing restrictions
- Reproducible simulation using random seeds
"""

import random

from typing import Dict, Any

from core.traffic_engine import (
    check_traffic_priority,
    calculate_green_time,
)

from core.safety import validate_signal_action


DIRECTIONS = ("north", "south", "east", "west")

PHASES = {
    "NS": ("north", "south"),
    "EW": ("east", "west"),
}

YELLOW_DURATION = 3
ALL_RED_DURATION = 2

VEHICLES_PER_SECOND = 0.5

TRAFFIC_DENSITY = {
    "low": 0.02,
    "medium": 0.05,
    "high": 0.10,
}


class TrafficSimulator:

    def __init__(
        self,
        vehicle_counts: Dict[str, int],
        waiting_times: Dict[str, float],
        traffic_density: str = "medium",
        random_seed: int = 42,
    ):
        """Initialise the traffic simulation."""

        # Validate initial traffic information.
        check_traffic_priority(
            vehicle_counts,
            waiting_times,
        )

        if traffic_density not in TRAFFIC_DENSITY:
            raise ValueError(
                "Traffic density must be low, medium or high."
            )

        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, int)
        ):
            raise ValueError("Random seed must be an integer.")

        self.vehicle_counts = vehicle_counts.copy()
        self.waiting_times = waiting_times.copy()

        self.traffic_density = traffic_density
        self.arrival_probability = TRAFFIC_DENSITY[
            traffic_density
        ]

        self.random = random.Random(random_seed)

        self.current_phase = "NS"
        self.current_signal_state = "GREEN"

        self.cycles_since_served = {
            "NS": 0,
            "EW": 0,
        }

        self.service_credit = {
            direction: 0.0
            for direction in DIRECTIONS
        }

        self.pedestrian_crossing_active = False
        self.simulation_time = 0

        self.total_arrivals = {
            direction: 0
            for direction in DIRECTIONS
        }

        self.total_departures = {
            direction: 0
            for direction in DIRECTIONS
        }

        self.history = []

    def get_traffic_status(self) -> Dict[str, Any]:
        """Return the current traffic status."""

        return {
            "vehicle_counts": self.vehicle_counts.copy(),
            "waiting_times": self.waiting_times.copy(),
            "current_phase": self.current_phase,
            "current_signal_state": self.current_signal_state,
            "pedestrian_crossing_active": (
                self.pedestrian_crossing_active
            ),
            "traffic_density": self.traffic_density,
            "simulation_time": self.simulation_time,
            "total_arrivals": self.total_arrivals.copy(),
            "total_departures": self.total_departures.copy(),
        }

    def update_pedestrian_request(self, active: bool) -> None:
        """Update pedestrian crossing status."""

        if not isinstance(active, bool):
            raise ValueError(
                "Pedestrian status must be True or False."
            )

        self.pedestrian_crossing_active = active

    def generate_vehicle_arrivals(self) -> Dict[str, int]:
        """
        Generate random vehicle arrivals for each direction.

        Each direction has an independent probability
        of receiving one new vehicle per second.
        """

        arrivals = {}

        for direction in DIRECTIONS:

            if self.random.random() < self.arrival_probability:

                self.vehicle_counts[direction] += 1

                self.total_arrivals[direction] += 1

                arrivals[direction] = 1

            else:
                arrivals[direction] = 0

        return arrivals

    def _advance_time(
        self,
        duration: int,
        green_phase: str = None,
    ) -> None:
        """
        Advance simulation time one second at a time.

        New vehicles may arrive during each second.
        Vehicles are served only during the active green phase.
        """

        if isinstance(duration, bool) or not isinstance(duration, int):
            raise ValueError("Duration must be an integer.")

        if duration < 0:
            raise ValueError("Duration cannot be negative.")

        if green_phase is not None and green_phase not in PHASES:
            raise ValueError("Invalid green phase.")

        for _ in range(duration):

            # Generate new vehicle arrivals.
            self.generate_vehicle_arrivals()

            # Update waiting times and serve vehicles.
            for direction in DIRECTIONS:

                if self.vehicle_counts[direction] > 0:

                    self.waiting_times[direction] += 1

                if (
                    green_phase is not None
                    and direction in PHASES[green_phase]
                ):

                    self.service_credit[direction] += (
                        VEHICLES_PER_SECOND
                    )

                    while (
                        self.service_credit[direction] >= 1
                        and self.vehicle_counts[direction] > 0
                    ):

                        self.vehicle_counts[direction] -= 1

                        self.total_departures[direction] += 1

                        self.service_credit[direction] -= 1

                    if self.vehicle_counts[direction] == 0:
                        self.waiting_times[direction] = 0

            self.simulation_time += 1

    def _apply_signal_transition(
        self,
        next_phase: str,
        next_state: str,
    ) -> None:
        """Validate and apply one signal transition."""

        result = validate_signal_action(
            current_phase=self.current_phase,
            current_signal_state=self.current_signal_state,
            requested_phase=next_phase,
            requested_signal_state=next_state,
            pedestrian_crossing_active=(
                self.pedestrian_crossing_active
            ),
        )

        if not result["allowed"]:
            raise ValueError(result["reason"])

        self.current_phase = next_phase
        self.current_signal_state = next_state

        self.history.append({
            "time": self.simulation_time,
            "phase": self.current_phase,
            "signal_state": self.current_signal_state,
            "reason": result["reason"],
        })

    def run_control_cycle(self) -> Dict[str, Any]:
        """
        Execute one traffic control decision cycle.

        1. Evaluate traffic priority.
        2. Calculate green duration.
        3. Perform signal transitions.
        4. Activate the selected phase.
        5. Generate arrivals and serve vehicles.
        """

        # Step 1: Evaluate traffic priority.
        decision = check_traffic_priority(
            vehicle_counts=self.vehicle_counts,
            waiting_times=self.waiting_times,
            cycles_since_served=self.cycles_since_served,
            previous_phase=self.current_phase,
        )

        selected_phase = decision["selected_phase"]

        # Step 2: Calculate green-light duration.
        timing = calculate_green_time(
            selected_phase=selected_phase,
            vehicle_counts=self.vehicle_counts,
            waiting_times=self.waiting_times,
        )

        green_duration = timing["green_time_seconds"]

        # Step 3: GREEN -> YELLOW.
        if self.current_signal_state == "GREEN":

            self._apply_signal_transition(
                self.current_phase,
                "YELLOW",
            )

            self._advance_time(YELLOW_DURATION)

        # Step 4: YELLOW -> ALL_RED.
        if self.current_signal_state == "YELLOW":

            self._apply_signal_transition(
                self.current_phase,
                "ALL_RED",
            )

            self._advance_time(ALL_RED_DURATION)

        elif self.current_signal_state != "ALL_RED":

            raise ValueError(
                "Invalid simulator signal state."
            )

        # Step 5: Activate selected green phase.
        try:

            self._apply_signal_transition(
                selected_phase,
                "GREEN",
            )

        except ValueError as error:

            return {
                "status": "BLOCKED",
                "reason": str(error),
                "decision": decision,
                "timing": timing,
                "traffic_status": self.get_traffic_status(),
            }

        # Step 6: Simulate green-light traffic movement.
        self._advance_time(
            green_duration,
            green_phase=selected_phase,
        )

        # Update fairness information.
        for phase in PHASES:

            if phase == selected_phase:
                self.cycles_since_served[phase] = 0

            else:
                self.cycles_since_served[phase] += 1

        return {
            "status": "SUCCESS",
            "selected_phase": selected_phase,
            "green_duration": green_duration,
            "decision": decision,
            "timing": timing,
            "traffic_status": self.get_traffic_status(),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate basic simulation performance metrics."""

        total_vehicles = sum(self.vehicle_counts.values())

        total_waiting = sum(self.waiting_times.values())

        total_arrivals = sum(self.total_arrivals.values())

        total_departures = sum(self.total_departures.values())

        return {
            "simulation_time": self.simulation_time,
            "total_vehicles_waiting": total_vehicles,
            "average_waiting_time": round(
                total_waiting / len(DIRECTIONS), 2
            ),
            "total_arrivals": total_arrivals,
            "total_departures": total_departures,
            "traffic_density": self.traffic_density,
        }
