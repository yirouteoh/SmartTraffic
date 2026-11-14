
import pytest

from simulation.traffic_simulator import TrafficSimulator


@pytest.fixture
def simulator():
    vehicles = {
        "north": 25,
        "south": 18,
        "east": 8,
        "west": 5,
    }

    waiting = {
        "north": 90,
        "south": 65,
        "east": 25,
        "west": 15,
    }

    return TrafficSimulator(vehicles, waiting)


def test_initial_signal_state(simulator):
    status = simulator.get_traffic_status()

    assert status["current_phase"] == "NS"
    assert status["current_signal_state"] == "GREEN"


def test_initial_vehicle_counts(simulator):
    status = simulator.get_traffic_status()

    assert status["vehicle_counts"]["north"] == 25
    assert status["vehicle_counts"]["south"] == 18


def test_run_control_cycle(simulator):
    result = simulator.run_control_cycle()

    assert result["status"] == "SUCCESS"
    assert result["selected_phase"] == "NS"
    assert result["green_duration"] == 54


def test_traffic_queue_reduces_after_green(simulator):
    original_north = simulator.vehicle_counts["north"]

    simulator.run_control_cycle()

    assert simulator.vehicle_counts["north"] < original_north


def test_waiting_time_increases_for_red_phase(simulator):
    original_east_wait = simulator.waiting_times["east"]

    simulator.run_control_cycle()

    assert simulator.waiting_times["east"] > original_east_wait


def test_signal_history_is_recorded(simulator):
    simulator.run_control_cycle()

    states = [
        item["signal_state"]
        for item in simulator.history
    ]

    assert states == ["YELLOW", "ALL_RED", "GREEN"]


def test_pedestrian_blocks_vehicle_green(simulator):
    simulator.update_pedestrian_request(True)

    result = simulator.run_control_cycle()

    assert result["status"] == "BLOCKED"
    assert simulator.current_signal_state == "ALL_RED"


def test_pedestrian_clearance_allows_next_phase(simulator):
    simulator.update_pedestrian_request(True)

    first_result = simulator.run_control_cycle()

    assert first_result["status"] == "BLOCKED"

    simulator.update_pedestrian_request(False)

    second_result = simulator.run_control_cycle()

    assert second_result["status"] == "SUCCESS"
    assert simulator.current_signal_state == "GREEN"


def test_invalid_pedestrian_status(simulator):
    with pytest.raises(ValueError):
        simulator.update_pedestrian_request("yes")
