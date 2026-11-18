
"""
SmartTraffic - Individual Vehicle Traffic Simulator

Features:
- Four incoming traffic approaches
- North-South and East-West traffic phases
- Individual car tracking
- Unique vehicle IDs
- Individual vehicle positions
- Automatic random vehicle arrivals
- Vehicle queues and waiting-time updates
- GREEN -> YELLOW -> ALL_RED -> GREEN transitions
- Validated AI traffic decisions
- Pedestrian crossing restrictions
- Reproducible random simulation
- Signal transition history
- Vehicle arrival and departure tracking
- Performance metrics

Integration:
- Accepts validated AI decisions.
- Accepts green duration from Python tools.
- Tracks individual cars in the backend.
- Provides vehicle states to the dashboard.
- Preserves the original run_control_cycle() interface.
"""

import random

from typing import Dict, Any, Optional

from core.traffic_engine import (
    check_traffic_priority,
    calculate_green_time,
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
)

from core.safety import validate_signal_action


# =====================================================
# 1. CONSTANTS
# =====================================================

DIRECTIONS = ("north", "south", "east", "west")

PHASES = {
    "NS": ("north", "south"),
    "EW": ("east", "west"),
}

YELLOW_DURATION = 3
ALL_RED_DURATION = 2

# Each active approach can serve one car every two seconds.
VEHICLES_PER_SECOND = 0.5

TRAFFIC_DENSITY = {
    "low": 0.02,
    "medium": 0.05,
    "high": 0.10,
}

# Individual car movement settings.
CAR_SPACING = 8.0
CAR_SPEED = 4.0
INITIAL_CAR_DISTANCE = 6.0

POSITION_EPSILON = 0.000001

DIRECTION_PREFIX = {
    "north": "N",
    "south": "S",
    "east": "E",
    "west": "W",
}


# =====================================================
# 2. TRAFFIC SIMULATOR
# =====================================================

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

            raise ValueError(
                "Random seed must be an integer."
            )

        self.vehicle_counts = vehicle_counts.copy()
        self.waiting_times = waiting_times.copy()

        self.traffic_density = traffic_density

        self.arrival_probability = TRAFFIC_DENSITY[
            traffic_density
        ]

        self.random = random.Random(random_seed)

        # Simulation clock must exist before car creation.
        self.simulation_time = 0

        # Individual vehicle records.
        self.vehicles = []

        self.vehicle_id_counters = {
            direction: 0
            for direction in DIRECTIONS
        }

        # Initial signal state.
        self.current_phase = "NS"
        self.current_signal_state = "GREEN"

        # Fairness information.
        self.cycles_since_served = {
            "NS": 0,
            "EW": 0,
        }

        # Fractional vehicle service.
        self.service_credit = {
            direction: 0.0
            for direction in DIRECTIONS
        }

        # Pedestrian protection.
        self.pedestrian_crossing_active = False

        # Performance tracking.
        self.total_arrivals = {
            direction: 0
            for direction in DIRECTIONS
        }

        self.total_departures = {
            direction: 0
            for direction in DIRECTIONS
        }

        # Signal and vehicle histories.
        self.history = []
        self.vehicle_history = []

        # Control-cycle state.
        self.cycle_stage = "IDLE"
        self.cycle_status = "IDLE"

        self.selected_phase = None
        self.current_decision = None
        self.current_timing = None

        self.remaining_time = 0
        self.green_duration = 0

        self.last_event = "Simulation initialised."

        # Create initial individual cars.
        for direction in DIRECTIONS:

            for _ in range(vehicle_counts[direction]):

                self._add_vehicle(
                    direction=direction,
                    initial=True,
                )

        self._sync_vehicle_counts()

    # =================================================
    # 3. INDIVIDUAL VEHICLE MANAGEMENT
    # =================================================

    def _add_vehicle(
        self,
        direction: str,
        initial: bool = False,
    ) -> Dict[str, Any]:
        """Create one individual car record."""

        if direction not in DIRECTIONS:

            raise ValueError(
                "Invalid vehicle direction."
            )

        self.vehicle_id_counters[direction] += 1

        vehicle_id = (
            f"{DIRECTION_PREFIX[direction]}"
            f"{self.vehicle_id_counters[direction]:03d}"
        )

        # Find the end of the existing queue.
        queue = [
            vehicle
            for vehicle in self.vehicles
            if vehicle["direction"] == direction
        ]

        if queue:

            position = max(
                vehicle["position"]
                for vehicle in queue
            ) + CAR_SPACING

        else:

            position = INITIAL_CAR_DISTANCE

        vehicle = {
            "vehicle_id": vehicle_id,

            "direction": direction,

            "vehicle_type": "car",

            # Distance from stop line in metres.
            "position": round(position, 2),

            "status": "WAITING",

            "speed": 0.0,

            "arrival_time": self.simulation_time,

            "waiting_time": (
                float(self.waiting_times[direction])
                if initial else 0.0
            ),
        }

        self.vehicles.append(vehicle)

        if not initial:

            self.vehicle_history.append({
                "time": self.simulation_time,
                "vehicle_id": vehicle_id,
                "direction": direction,
                "event": "ARRIVAL",
            })

        return vehicle

    # =================================================
    # 4. GET INDIVIDUAL VEHICLE STATES
    # =================================================

    def get_vehicle_states(self) -> list:
        """Return individual vehicle states."""

        return [
            vehicle.copy()
            for vehicle in self.vehicles
        ]

    def get_vehicles_by_direction(self) -> Dict[str, list]:
        """Return individual cars grouped by direction."""

        return {
            direction: [
                vehicle.copy()
                for vehicle in self.vehicles
                if vehicle["direction"] == direction
            ]
            for direction in DIRECTIONS
        }

    # =================================================
    # 5. SYNCHRONISE VEHICLE COUNTS
    # =================================================

    def _sync_vehicle_counts(self) -> None:
        """Synchronise counts with individual car records."""

        for direction in DIRECTIONS:

            self.vehicle_counts[direction] = sum(
                1
                for vehicle in self.vehicles
                if vehicle["direction"] == direction
            )

    # =================================================
    # 6. GET FRONT VEHICLE
    # =================================================

    def _get_front_vehicle(
        self,
        direction: str,
    ) -> Optional[Dict[str, Any]]:
        """Find the car closest to the stop line."""

        queue = [
            vehicle
            for vehicle in self.vehicles
            if vehicle["direction"] == direction
        ]

        if not queue:
            return None

        return min(
            queue,
            key=lambda vehicle: vehicle["position"],
        )

    # =================================================
    # 7. REMOVE VEHICLE AFTER DEPARTURE
    # =================================================

    def _remove_vehicle(
        self,
        direction: str,
    ) -> Optional[Dict[str, Any]]:
        """Remove a car only after reaching the stop line."""

        vehicle = self._get_front_vehicle(direction)

        if vehicle is None:
            return None

        if vehicle["position"] > POSITION_EPSILON:
            return None

        self.vehicles.remove(vehicle)

        self.total_departures[direction] += 1

        self.vehicle_history.append({
            "time": self.simulation_time,
            "vehicle_id": vehicle["vehicle_id"],
            "direction": direction,
            "event": "DEPARTURE",
        })

        self._sync_vehicle_counts()

        return vehicle

    # =================================================
    # 8. GET CURRENT TRAFFIC STATUS
    # =================================================

    def get_traffic_status(self) -> Dict[str, Any]:
        """Return actual traffic and individual car states."""

        self._sync_vehicle_counts()

        return {
            "vehicle_counts": self.vehicle_counts.copy(),

            "waiting_times": self.waiting_times.copy(),

            "vehicles": self.get_vehicle_states(),

            "vehicles_by_direction": (
                self.get_vehicles_by_direction()
            ),

            "current_phase": self.current_phase,

            "current_signal_state": self.current_signal_state,

            "pedestrian_crossing_active": (
                self.pedestrian_crossing_active
            ),

            "traffic_density": self.traffic_density,

            "simulation_time": self.simulation_time,

            "total_arrivals": self.total_arrivals.copy(),

            "total_departures": self.total_departures.copy(),

            "cycles_since_served": (
                self.cycles_since_served.copy()
            ),

            "cycle_stage": self.cycle_stage,

            "cycle_status": self.cycle_status,

            "selected_phase": self.selected_phase,

            "remaining_time": self.remaining_time,

            "green_duration": self.green_duration,

            "total_individual_cars": len(self.vehicles),
        }

    # =================================================
    # 9. PEDESTRIAN REQUEST
    # =================================================

    def update_pedestrian_request(
        self,
        active: bool,
    ) -> None:
        """Update pedestrian crossing status."""

        if not isinstance(active, bool):

            raise ValueError(
                "Pedestrian status must be True or False."
            )

        self.pedestrian_crossing_active = active

        self.last_event = (
            "Pedestrian crossing activated."
            if active
            else "Pedestrian crossing cleared."
        )

    # =================================================
    # 10. VEHICLE ARRIVALS
    # =================================================

    def generate_vehicle_arrivals(self) -> Dict[str, int]:
        """Generate random arrivals and create car records."""

        arrivals = {}

        for direction in DIRECTIONS:

            if (
                self.random.random()
                < self.arrival_probability
            ):

                self._add_vehicle(
                    direction=direction,
                    initial=False,
                )

                self.total_arrivals[direction] += 1

                arrivals[direction] = 1

            else:

                arrivals[direction] = 0

        self._sync_vehicle_counts()

        return arrivals

    # =================================================
    # 11. UPDATE INDIVIDUAL CAR POSITIONS
    # =================================================

    def _move_vehicle_queue(
        self,
        direction: str,
        moving: bool,
    ) -> None:
        """
        Update individual car positions.

        RED / YELLOW / ALL_RED:
            Cars remain stationary.

        GREEN:
            Cars move towards the stop line.
            Queue spacing is maintained.

        MOVING status is assigned only when the
        actual position changes.
        """

        queue = sorted(
            [
                vehicle
                for vehicle in self.vehicles
                if vehicle["direction"] == direction
            ],
            key=lambda vehicle: vehicle["position"],
        )

        previous_position = None

        for vehicle in queue:

            old_position = float(vehicle["position"])

            if moving:

                # Calculate desired new position.
                desired_position = max(
                    0.0,
                    old_position - CAR_SPEED,
                )

                # Prevent cars from overlapping.
                if previous_position is not None:

                    new_position = max(
                        desired_position,
                        previous_position + CAR_SPACING,
                    )

                else:

                    new_position = desired_position

                new_position = round(
                    new_position,
                    2,
                )

                # Calculate actual movement distance.
                actual_distance = max(
                    0.0,
                    round(
                        old_position - new_position,
                        2,
                    ),
                )

                # Update actual position.
                vehicle["position"] = new_position

                # Speed is zero when position does not change.
                vehicle["speed"] = (
                    actual_distance
                )

                vehicle["status"] = (
                    "MOVING"
                    if actual_distance > POSITION_EPSILON
                    else "WAITING"
                )

            else:

                # No movement during non-green stages.
                vehicle["speed"] = 0.0

                vehicle["status"] = "WAITING"

            previous_position = vehicle["position"]

    # =================================================
    # 12. SERVE FRONT VEHICLES
    # =================================================

    def _serve_vehicles(
        self,
        direction: str,
    ) -> int:
        """
        Serve vehicles according to the configured
        service rate.

        A vehicle must reach the stop line before
        it can be removed from the simulation.
        """

        self.service_credit[direction] += (
            VEHICLES_PER_SECOND
        )

        departures = 0

        while self.service_credit[direction] >= 1.0:

            vehicle = self._remove_vehicle(direction)

            if vehicle is None:
                break

            self.service_credit[direction] -= 1.0

            departures += 1

        return departures

    # =================================================
    # 13. ADVANCE SIMULATION TIME
    # =================================================

    def _advance_time(
        self,
        duration: int,
        green_phase: Optional[str] = None,
    ) -> None:
        """
        Advance simulation time.

        Individual vehicles:
        - Arrive as separate car records.
        - Queue at red signals.
        - Move during their green phase.
        - Depart after reaching the stop line.
        """

        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
        ):

            raise ValueError(
                "Duration must be an integer."
            )

        if duration < 0:

            raise ValueError(
                "Duration cannot be negative."
            )

        if (
            green_phase is not None
            and green_phase not in PHASES
        ):

            raise ValueError(
                "Invalid green phase."
            )

        for _ in range(duration):

            # Generate new vehicles.
            self.generate_vehicle_arrivals()

            for direction in DIRECTIONS:

                # Determine whether this approach is green.
                is_green = (
                    green_phase is not None
                    and direction in PHASES[green_phase]
                )

                # Move or stop individual vehicles.
                self._move_vehicle_queue(
                    direction=direction,
                    moving=is_green,
                )

                # Update individual waiting times.
                for vehicle in self.vehicles:

                    if vehicle["direction"] != direction:
                        continue

                    if vehicle["status"] == "WAITING":

                        vehicle["waiting_time"] += 1

                # Update approach waiting time.
                if self.vehicle_counts[direction] > 0:

                    self.waiting_times[direction] += 1

                # Serve eligible vehicles during green.
                if is_green:

                    self._serve_vehicles(direction)

                # Reset waiting time if the queue is empty.
                if self.vehicle_counts[direction] == 0:

                    self.waiting_times[direction] = 0

            self.simulation_time += 1

            self._sync_vehicle_counts()

    # =================================================
    # 14. SIGNAL TRANSITION VALIDATION
    # =================================================

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

            raise ValueError(
                result["reason"]
            )

        self.current_phase = next_phase

        self.current_signal_state = next_state

        event = {
            "time": self.simulation_time,
            "phase": self.current_phase,
            "signal_state": self.current_signal_state,
            "reason": result["reason"],
        }

        self.history.append(event)

        self.last_event = (
            f"{next_phase} signal changed to {next_state}."
        )

    # =================================================
    # 15. UPDATE FAIRNESS
    # =================================================

    def _update_fairness(self) -> None:
        """Update cycles since each phase was served."""

        for phase in PHASES:

            if phase == self.selected_phase:

                self.cycles_since_served[phase] = 0

            else:

                self.cycles_since_served[phase] += 1

    # =================================================
    # 16. VALIDATE EXTERNAL AI DECISION
    # =================================================

    def _validate_external_decision(
        self,
        selected_phase: str,
        green_duration: int,
        decision: Optional[Dict[str, Any]] = None,
        timing: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate the decision received from the AI workflow."""

        if selected_phase not in PHASES:

            raise ValueError(
                "Selected phase must be NS or EW."
            )

        if (
            isinstance(green_duration, bool)
            or not isinstance(green_duration, int)
        ):

            raise ValueError(
                "Green duration must be an integer."
            )

        if not (
            MIN_GREEN_TIME
            <= green_duration
            <= MAX_GREEN_TIME
        ):

            raise ValueError(
                f"Green duration must be between "
                f"{MIN_GREEN_TIME} and {MAX_GREEN_TIME} seconds."
            )

        if decision is not None:

            if not isinstance(decision, dict):

                raise ValueError(
                    "Decision result must be a dictionary."
                )

            if (
                "selected_phase" in decision
                and decision["selected_phase"] != selected_phase
            ):

                raise ValueError(
                    "Selected phase does not match "
                    "the Python priority result."
                )

        if timing is not None:

            if not isinstance(timing, dict):

                raise ValueError(
                    "Timing result must be a dictionary."
                )

            if (
                "green_time_seconds" in timing
                and timing["green_time_seconds"] != green_duration
            ):

                raise ValueError(
                    "Green duration does not match "
                    "the Python timing result."
                )

    # =================================================
    # 17. START CONTROL CYCLE
    # =================================================

    def start_control_cycle(
        self,
        selected_phase: Optional[str] = None,
        green_duration: Optional[int] = None,
        decision: Optional[Dict[str, Any]] = None,
        timing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start a cycle using external AI results or internal engine."""

        if self.cycle_status == "RUNNING":

            raise ValueError(
                "A control cycle is already running."
            )

        if self.cycle_status == "COMPLETED":

            raise ValueError(
                "The previous cycle is completed."
            )

        # EXTERNAL AI DECISION MODE.
        if (
            selected_phase is not None
            or green_duration is not None
        ):

            if (
                selected_phase is None
                or green_duration is None
            ):

                raise ValueError(
                    "Both selected_phase and green_duration "
                    "must be provided together."
                )

            self._validate_external_decision(
                selected_phase=selected_phase,
                green_duration=green_duration,
                decision=decision,
                timing=timing,
            )

            self.selected_phase = selected_phase

            self.green_duration = green_duration

            self.current_decision = (
                decision.copy()
                if decision is not None
                else {
                    "selected_phase": selected_phase
                }
            )

            self.current_timing = (
                timing.copy()
                if timing is not None
                else {
                    "selected_phase": selected_phase,
                    "green_time_seconds": green_duration,
                }
            )

        # INTERNAL MODE FOR BACKWARD COMPATIBILITY.
        else:

            calculated_decision = check_traffic_priority(
                vehicle_counts=self.vehicle_counts,
                waiting_times=self.waiting_times,
                cycles_since_served=self.cycles_since_served,
                previous_phase=self.current_phase,
            )

            calculated_phase = calculated_decision[
                "selected_phase"
            ]

            calculated_timing = calculate_green_time(
                selected_phase=calculated_phase,
                vehicle_counts=self.vehicle_counts,
                waiting_times=self.waiting_times,
            )

            self.selected_phase = calculated_phase

            self.green_duration = (
                calculated_timing["green_time_seconds"]
            )

            self.current_decision = calculated_decision

            self.current_timing = calculated_timing

        self.cycle_status = "RUNNING"

        # Begin signal transition.
        if self.current_signal_state == "GREEN":

            self._apply_signal_transition(
                self.current_phase,
                "YELLOW",
            )

            self.cycle_stage = "YELLOW"

            self.remaining_time = YELLOW_DURATION

        elif self.current_signal_state == "YELLOW":

            self.cycle_stage = "YELLOW"

            self.remaining_time = YELLOW_DURATION

        elif self.current_signal_state == "ALL_RED":

            self.cycle_stage = "ALL_RED"

            self.remaining_time = 0

        else:

            raise ValueError(
                "Invalid traffic signal state."
            )

        self.last_event = (
            f"Control cycle started. "
            f"Selected phase: {self.selected_phase}. "
            f"Green duration: {self.green_duration} seconds."
        )

        return {
            "status": "RUNNING",

            "selected_phase": self.selected_phase,

            "green_duration": self.green_duration,

            "decision": self.current_decision,

            "timing": self.current_timing,

            "traffic_status": self.get_traffic_status(),
        }

    # =================================================
    # 18. ACTIVATE SELECTED GREEN
    # =================================================

    def _activate_selected_green(self) -> bool:
        """Activate the selected phase after safety validation."""

        try:

            self._apply_signal_transition(
                self.selected_phase,
                "GREEN",
            )

        except ValueError as error:

            self.cycle_stage = "BLOCKED"

            self.cycle_status = "BLOCKED"

            self.remaining_time = 0

            self.last_event = (
                f"Green activation blocked: {error}"
            )

            return False

        self.cycle_stage = "GREEN"

        self.remaining_time = self.green_duration

        return True

    # =================================================
    # 19. COMPLETE CONTROL CYCLE
    # =================================================

    def _complete_control_cycle(self) -> None:
        """Mark the current control cycle as completed."""

        self.cycle_stage = "COMPLETE"

        self.cycle_status = "COMPLETED"

        self.remaining_time = 0

        self._update_fairness()

        self.last_event = (
            f"Control cycle completed. "
            f"Served phase: {self.selected_phase}."
        )

    # =================================================
    # 20. ADVANCE ONE SIMULATION STEP
    # =================================================

    def _step_once(self) -> None:
        """Advance simulation by exactly one second."""

        if self.cycle_status != "RUNNING":

            raise ValueError(
                "No active control cycle."
            )

        # YELLOW STAGE.
        if self.cycle_stage == "YELLOW":

            self._advance_time(1)

            self.remaining_time -= 1

            if self.remaining_time == 0:

                self._apply_signal_transition(
                    self.current_phase,
                    "ALL_RED",
                )

                self.cycle_stage = "ALL_RED"

                self.remaining_time = ALL_RED_DURATION

            return

        # ALL RED STAGE.
        if self.cycle_stage == "ALL_RED":

            if self.remaining_time > 0:

                self._advance_time(1)

                self.remaining_time -= 1

                if self.remaining_time > 0:

                    return

            activated = self._activate_selected_green()

            if not activated:
                return

            return

        # GREEN STAGE.
        if self.cycle_stage == "GREEN":

            self._advance_time(
                1,
                green_phase=self.selected_phase,
            )

            self.remaining_time -= 1

            if self.remaining_time == 0:

                self._complete_control_cycle()

            return

        raise ValueError(
            f"Unknown simulation stage: {self.cycle_stage}"
        )

    # =================================================
    # 21. PUBLIC STEP SIMULATION
    # =================================================

    def step_simulation(
        self,
        seconds: int = 1,
    ) -> Dict[str, Any]:
        """Advance the active simulation by the requested seconds."""

        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int)
            or seconds <= 0
        ):

            raise ValueError(
                "Seconds must be a positive integer."
            )

        if self.cycle_status != "RUNNING":

            raise ValueError(
                "No active simulation cycle."
            )

        events = []

        for _ in range(seconds):

            if self.cycle_status != "RUNNING":
                break

            previous_history_length = len(self.history)

            previous_time = self.simulation_time

            self._step_once()

            if len(self.history) > previous_history_length:

                events.extend(
                    self.history[previous_history_length:]
                )

            elif self.simulation_time != previous_time:

                events.append({
                    "time": self.simulation_time,
                    "phase": self.current_phase,
                    "signal_state": self.current_signal_state,
                    "reason": self.last_event,
                })

        return {
            "status": self.cycle_status,

            "stage": self.cycle_stage,

            "selected_phase": self.selected_phase,

            "green_duration": self.green_duration,

            "remaining_time": self.remaining_time,

            "simulation_time": self.simulation_time,

            "last_event": self.last_event,

            "events": events,

            "traffic_status": self.get_traffic_status(),

            "vehicle_states": self.get_vehicle_states(),

            "performance_metrics": (
                self.get_performance_metrics()
            ),
        }

    # =================================================
    # 22. RUN COMPLETE CONTROL CYCLE
    # =================================================

    def run_control_cycle(self) -> Dict[str, Any]:
        """Execute one complete control cycle."""

        self.start_control_cycle()

        while self.cycle_status == "RUNNING":

            self.step_simulation(seconds=1)

        if self.cycle_status == "BLOCKED":

            return {
                "status": "BLOCKED",

                "reason": self.last_event,

                "decision": self.current_decision,

                "timing": self.current_timing,

                "traffic_status": self.get_traffic_status(),

                "vehicle_states": self.get_vehicle_states(),
            }

        return {
            "status": "SUCCESS",

            "selected_phase": self.selected_phase,

            "green_duration": self.green_duration,

            "decision": self.current_decision,

            "timing": self.current_timing,

            "traffic_status": self.get_traffic_status(),

            "vehicle_states": self.get_vehicle_states(),
        }

    # =================================================
    # 23. PERFORMANCE METRICS
    # =================================================

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate simulation performance metrics."""

        self._sync_vehicle_counts()

        total_vehicles = sum(
            self.vehicle_counts.values()
        )

        total_waiting = sum(
            self.waiting_times.values()
        )

        total_arrivals = sum(
            self.total_arrivals.values()
        )

        total_departures = sum(
            self.total_departures.values()
        )

        return {
            "simulation_time": self.simulation_time,

            "total_vehicles_waiting": total_vehicles,

            "total_individual_cars": len(self.vehicles),

            "average_waiting_time": round(
                total_waiting / len(DIRECTIONS),
                2,
            ),

            "total_arrivals": total_arrivals,

            "total_departures": total_departures,

            "traffic_density": self.traffic_density,

            "current_phase": self.current_phase,

            "current_signal_state": self.current_signal_state,

            "cycle_status": self.cycle_status,

            "cycle_stage": self.cycle_stage,
        }
