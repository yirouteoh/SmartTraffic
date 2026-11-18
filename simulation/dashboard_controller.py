
"""
SmartTraffic - Dashboard Controller

Connects:
1. Qwen AI agent results
2. Python traffic decision functions
3. TrafficSimulator
4. Individual vehicle records
5. Gradio dashboard

Main workflow:
User Input
    -> AI Agent
    -> Python Tool Results
    -> Dashboard Controller
    -> Traffic Simulator
    -> Individual Car States
    -> Dashboard Animation

The controller passes actual validated AI/Python
results to the simulator.

The simulator remains the authority for vehicle
movement, traffic queues and signal states.
"""

from copy import deepcopy
from typing import Dict, Any, Optional

from simulation.traffic_simulator import (
    TrafficSimulator,
)

from core.traffic_engine import (
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
)


# =====================================================
# 1. DASHBOARD CONTROLLER
# =====================================================

class DashboardController:

    def __init__(self):
        """Initialise the dashboard controller."""

        self.simulator: Optional[TrafficSimulator] = None

        self.initial_vehicle_counts = None
        self.initial_waiting_times = None

        self.traffic_density = "medium"
        self.random_seed = 42

        self.last_result = None

        self.last_ai_decision = None
        self.last_ai_timing = None

        self.controller_status = "NOT_INITIALISED"

    # =================================================
    # 2. INITIALISE SIMULATION
    # =================================================

    def initialise(
        self,
        vehicle_counts: Dict[str, int],
        waiting_times: Dict[str, float],
        traffic_density: str = "medium",
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Initialise a new traffic simulation.

        Individual car records are created by
        TrafficSimulator using the supplied counts.
        """

        if not isinstance(vehicle_counts, dict):

            raise ValueError(
                "Vehicle counts must be a dictionary."
            )

        if not isinstance(waiting_times, dict):

            raise ValueError(
                "Waiting times must be a dictionary."
            )

        self.simulator = TrafficSimulator(
            vehicle_counts=vehicle_counts,
            waiting_times=waiting_times,
            traffic_density=traffic_density,
            random_seed=random_seed,
        )

        self.initial_vehicle_counts = (
            vehicle_counts.copy()
        )

        self.initial_waiting_times = (
            waiting_times.copy()
        )

        self.traffic_density = traffic_density
        self.random_seed = random_seed

        self.last_result = None
        self.last_ai_decision = None
        self.last_ai_timing = None

        self.controller_status = "READY"

        return self.get_dashboard_data()

    # =================================================
    # 3. CHECK SIMULATOR
    # =================================================

    def _require_simulator(self) -> None:
        """Ensure the simulator has been initialised."""

        if self.simulator is None:

            raise RuntimeError(
                "Simulation has not been initialised."
            )

    # =================================================
    # 4. VALIDATE AI RESULTS
    # =================================================

    def _validate_ai_results(
        self,
        selected_phase: str,
        green_duration: int,
        decision: Dict[str, Any],
        timing: Dict[str, Any],
    ) -> None:
        """
        Validate actual Python function results before
        passing them to TrafficSimulator.
        """

        if not isinstance(decision, dict):

            raise ValueError(
                "AI traffic decision must be a dictionary."
            )

        if not isinstance(timing, dict):

            raise ValueError(
                "AI green-time result must be a dictionary."
            )

        if selected_phase not in ("NS", "EW"):

            raise ValueError(
                "Invalid selected traffic phase."
            )

        if (
            isinstance(green_duration, bool)
            or not isinstance(green_duration, int)
        ):

            raise ValueError(
                "Green duration must be an integer."
            )

        # Verify priority result.
        if decision.get("selected_phase") != selected_phase:

            raise ValueError(
                "Selected phase does not match "
                "the Python priority result."
            )

        # Verify green-time result.
        if timing.get("selected_phase") != selected_phase:

            raise ValueError(
                "Green-time phase does not match "
                "the selected priority phase."
            )

        if timing.get("green_time_seconds") != green_duration:

            raise ValueError(
                "Green duration does not match "
                "the Python calculation result."
            )

        # Verify configured green-time limits.
        if not (
            MIN_GREEN_TIME
            <= green_duration
            <= MAX_GREEN_TIME
        ):

            raise ValueError(
                f"Green duration must be between "
                f"{MIN_GREEN_TIME} and {MAX_GREEN_TIME} seconds."
            )

    # =================================================
    # 5. START CYCLE USING AI DECISION
    # =================================================

    def start_cycle(
        self,
        selected_phase: str,
        green_duration: int,
        decision: Dict[str, Any],
        timing: Dict[str, Any],
        pedestrian_active: bool = False,
    ) -> Dict[str, Any]:
        """
        Start simulation using actual AI/Python results.

        No new traffic priority is calculated here.

        The validated decision is passed directly to
        TrafficSimulator.
        """

        self._require_simulator()

        if not isinstance(pedestrian_active, bool):

            raise ValueError(
                "Pedestrian status must be True or False."
            )

        # Validate actual Python results.
        self._validate_ai_results(
            selected_phase=selected_phase,
            green_duration=green_duration,
            decision=decision,
            timing=timing,
        )

        # Update pedestrian status.
        self.simulator.update_pedestrian_request(
            pedestrian_active
        )

        # Pass actual AI decision to the simulator.
        self.last_result = (
            self.simulator.start_control_cycle(
                selected_phase=selected_phase,
                green_duration=green_duration,
                decision=decision,
                timing=timing,
            )
        )

        self.last_ai_decision = deepcopy(decision)
        self.last_ai_timing = deepcopy(timing)

        self.controller_status = (
            self.last_result["status"]
        )

        return {
            "action": "START_AI_CONTROL_CYCLE",

            "result": deepcopy(self.last_result),

            **self.get_dashboard_data(),
        }

    # =================================================
    # 6. ADVANCE SIMULATION
    # =================================================

    def step(
        self,
        seconds: int = 1,
    ) -> Dict[str, Any]:
        """
        Advance the active simulation.

        Each step updates:
        - Individual car positions
        - Vehicle queues
        - Waiting times
        - Signal states
        - Performance metrics
        """

        self._require_simulator()

        self.last_result = (
            self.simulator.step_simulation(
                seconds=seconds
            )
        )

        self.controller_status = (
            self.last_result["status"]
        )

        return {
            "action": "SIMULATION_STEP",

            "result": deepcopy(self.last_result),

            **self.get_dashboard_data(),
        }

    # =================================================
    # 7. UPDATE PEDESTRIAN STATUS
    # =================================================

    def set_pedestrian_crossing(
        self,
        active: bool,
    ) -> Dict[str, Any]:
        """Update pedestrian crossing status."""

        self._require_simulator()

        self.simulator.update_pedestrian_request(
            active
        )

        return self.get_dashboard_data()

    # =================================================
    # 8. RUN COMPLETE CONTROL CYCLE
    # =================================================

    def run_complete_cycle(
        self,
        selected_phase: str,
        green_duration: int,
        decision: Dict[str, Any],
        timing: Dict[str, Any],
        pedestrian_active: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a complete cycle using AI results.

        This method is intended for complete-cycle
        execution and integration testing.

        The animated dashboard should use step()
        instead, so it can display each update.
        """

        self._require_simulator()

        self.start_cycle(
            selected_phase=selected_phase,
            green_duration=green_duration,
            decision=decision,
            timing=timing,
            pedestrian_active=pedestrian_active,
        )

        while self.simulator.cycle_status == "RUNNING":

            self.step(seconds=1)

        self.controller_status = (
            self.simulator.cycle_status
        )

        return {
            "action": "COMPLETE_AI_CONTROL_CYCLE",

            **self.get_dashboard_data(),
        }

    # =================================================
    # 9. RESET SIMULATION
    # =================================================

    def reset(self) -> Dict[str, Any]:
        """Restore the original traffic conditions."""

        if self.initial_vehicle_counts is None:

            raise RuntimeError(
                "No previous simulation is available to reset."
            )

        return self.initialise(
            vehicle_counts=self.initial_vehicle_counts,
            waiting_times=self.initial_waiting_times,
            traffic_density=self.traffic_density,
            random_seed=self.random_seed,
        )

    # =================================================
    # 10. GET INDIVIDUAL VEHICLE DATA
    # =================================================

    def get_vehicle_data(self) -> Dict[str, Any]:
        """
        Return actual individual car records.

        This data is used by the Gradio visualisation
        to render cars at their current positions.
        """

        self._require_simulator()

        return {
            "vehicles": (
                self.simulator.get_vehicle_states()
            ),

            "vehicles_by_direction": (
                self.simulator.get_vehicles_by_direction()
            ),

            "total_individual_cars": (
                len(self.simulator.vehicles)
            ),

            "vehicle_history": (
                deepcopy(self.simulator.vehicle_history)
            ),
        }

    # =================================================
    # 11. GET DASHBOARD DATA
    # =================================================

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Return all actual backend information.

        No vehicle positions, traffic counts or
        signal states are hardcoded here.
        """

        self._require_simulator()

        status = (
            self.simulator.get_traffic_status()
        )

        metrics = (
            self.simulator.get_performance_metrics()
        )

        vehicle_data = self.get_vehicle_data()

        return {
            # Actual traffic status.
            "traffic_status": status,

            # Individual vehicle information.
            "vehicle_states": vehicle_data["vehicles"],

            "vehicles_by_direction": (
                vehicle_data["vehicles_by_direction"]
            ),

            "total_individual_cars": (
                vehicle_data["total_individual_cars"]
            ),

            "vehicle_history": (
                vehicle_data["vehicle_history"]
            ),

            # Performance information.
            "performance_metrics": metrics,

            # Signal information.
            "signal_history": (
                deepcopy(self.simulator.history)
            ),

            # AI decision information.
            "last_ai_decision": (
                deepcopy(self.last_ai_decision)
            ),

            "last_ai_timing": (
                deepcopy(self.last_ai_timing)
            ),

            # Execution information.
            "last_result": (
                deepcopy(self.last_result)
            ),

            "last_event": self.simulator.last_event,

            # Simulation status.
            "cycle_status": (
                self.simulator.cycle_status
            ),

            "cycle_stage": (
                self.simulator.cycle_stage
            ),

            "controller_status": (
                self.controller_status
            ),
        }
