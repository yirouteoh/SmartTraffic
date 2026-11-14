
"""
SmartTraffic - Traffic Intersection Simulator

Simulates:
- Four incoming traffic approaches
- North-South and East-West traffic phases
- Vehicle queues and waiting times
- Traffic signal transitions
- Pedestrian crossing restrictions
"""

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

# Illustrative simulation service rate.
VEHICLES_PER_SECOND = 0.5


class TrafficSimulator:

    def __init__(
        self,
        vehicle_counts: Dict[str, int],
        waiting_times: Dict[str, float],
    ):
        # Validate initial traffic information.
        check_traffic_priority(vehicle_counts, waiting_times)

        self.vehicle_counts = vehicle_counts.copy()
        self.waiting_times = waiting_times.copy()

        self.current_phase = "NS"
        self.current_signal_state = "GREEN"

        self.cycles_since_served = {
            "NS": 0,
            "EW": 0,
        }

        self.pedestrian_crossing_active = False
        self.simulation_time = 0

        self.history = []

    def get_traffic_status(self) -> Dict[str, Any]:
        """Return the current intersection status."""

        return {
            "vehicle_counts": self.vehicle_counts.copy(),
            "waiting_times": self.waiting_times.copy(),
            "current_phase": self.current_phase,
            "current_signal_state": self.current_signal_state,
            "pedestrian_crossing_active": (
                self.pedestrian_crossing_active
            ),
            "simulation_time": self.simulation_time,
        }

    def update_pedestrian_request(self, active: bool) -> None:
        """Update pedestrian crossing status."""

        if not isinstance(active, bool):
            raise ValueError(
                "Pedestrian status must be True or False."
            )

        self.pedestrian_crossing_active = active

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

    def _update_traffic_queues(
        self,
        served_phase: str,
        green_duration: int,
    ) -> None:
        """Update vehicle queues and waiting times."""

        served_directions = PHASES[served_phase]

        for direction in DIRECTIONS:

            if direction in served_directions:

                vehicles_served = min(
                    self.vehicle_counts[direction],
                    int(green_duration * VEHICLES_PER_SECOND),
                )

                self.vehicle_counts[direction] -= (
                    vehicles_served
                )

                # Approximate remaining vehicle waiting time.
                if self.vehicle_counts[direction] == 0:
                    self.waiting_times[direction] = 0

                else:
                    self.waiting_times[direction] = max(
                        0,
                        self.waiting_times[direction]
                        - green_duration,
                    )

            else:
                self.waiting_times[direction] += green_duration

        # Update fairness information.
        for phase in PHASES:

            if phase == served_phase:
                self.cycles_since_served[phase] = 0

            else:
                self.cycles_since_served[phase] += 1

    def run_control_cycle(self) -> Dict[str, Any]:
        """
        Execute one traffic control decision cycle.

        The cycle:
        1. Evaluates traffic priority.
        2. Calculates green duration.
        3. Performs GREEN -> YELLOW -> ALL_RED.
        4. Attempts to activate the selected green phase.
        5. Updates traffic queues.
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

        # Step 3: Complete the required signal transitions.

        # GREEN -> YELLOW
        if self.current_signal_state == "GREEN":

            self._apply_signal_transition(
                self.current_phase,
                "YELLOW",
            )

            self.simulation_time += YELLOW_DURATION

        # YELLOW -> ALL_RED
        if self.current_signal_state == "YELLOW":

            self._apply_signal_transition(
                self.current_phase,
                "ALL_RED",
            )

            self.simulation_time += ALL_RED_DURATION

        elif self.current_signal_state != "ALL_RED":

            raise ValueError(
                "Invalid simulator signal state."
            )

        # Step 4: Attempt to activate the selected phase.
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

        # Step 5: Simulate the green-light period.
        self.simulation_time += green_duration

        self._update_traffic_queues(
            selected_phase,
            green_duration,
        )

        # Return the completed simulation result.
        return {
            "status": "SUCCESS",
            "selected_phase": selected_phase,
            "green_duration": green_duration,
            "decision": decision,
            "timing": timing,
            "traffic_status": self.get_traffic_status(),
        }
