
import pytest

from core.traffic_engine import (
    check_traffic_priority,
    calculate_green_time,
)


@pytest.fixture
def traffic_data():
    vehicle_counts = {
        "north": 25,
        "south": 18,
        "east": 8,
        "west": 5,
    }

    waiting_times = {
        "north": 90,
        "south": 65,
        "east": 25,
        "west": 15,
    }

    return vehicle_counts, waiting_times


def test_highest_traffic_phase_selected(traffic_data):
    vehicles, waiting = traffic_data

    result = check_traffic_priority(vehicles, waiting)

    assert result["selected_phase"] == "NS"


def test_priority_scores(traffic_data):
    vehicles, waiting = traffic_data

    result = check_traffic_priority(vehicles, waiting)

    assert result["priority_scores"]["NS"] == 0.7955
    assert result["priority_scores"]["EW"] == 0.2948


def test_green_time_within_limits(traffic_data):
    vehicles, waiting = traffic_data

    result = calculate_green_time("NS", vehicles, waiting)

    assert 15 <= result["green_time_seconds"] <= 60


def test_green_time_high_traffic(traffic_data):
    vehicles, waiting = traffic_data

    result = calculate_green_time("NS", vehicles, waiting)

    assert result["green_time_seconds"] == 54


def test_green_time_zero_traffic():
    vehicles = {d: 0 for d in
                ("north", "south", "east", "west")}

    waiting = {d: 0 for d in
               ("north", "south", "east", "west")}

    result = calculate_green_time("NS", vehicles, waiting)

    assert result["green_time_seconds"] == 15


def test_invalid_negative_vehicle_count(traffic_data):
    vehicles, waiting = traffic_data
    vehicles["north"] = -5

    with pytest.raises(ValueError):
        check_traffic_priority(vehicles, waiting)


def test_missing_direction(traffic_data):
    vehicles, waiting = traffic_data
    del vehicles["north"]

    with pytest.raises(ValueError):
        check_traffic_priority(vehicles, waiting)


def test_invalid_traffic_phase(traffic_data):
    vehicles, waiting = traffic_data

    with pytest.raises(ValueError):
        calculate_green_time("NE", vehicles, waiting)
