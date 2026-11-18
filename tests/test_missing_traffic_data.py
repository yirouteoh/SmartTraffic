
from core.validation import validate_traffic_request_data


def test_missing_all_traffic_information():

    request = (
        "Please analyse the traffic at this intersection "
        "and recommend the best traffic light timing."
    )

    try:
        validate_traffic_request_data(request)

    except ValueError as error:
        assert "INSUFFICIENT_TRAFFIC_INFORMATION" in str(error)

    else:
        raise AssertionError(
            "Missing traffic information was incorrectly accepted."
        )


def test_complete_traffic_information():

    request = """
    Analyse this traffic:

    North: 15 vehicles, waiting 40 seconds
    South: 12 vehicles, waiting 35 seconds
    East: 14 vehicles, waiting 38 seconds
    West: 13 vehicles, waiting 36 seconds
    """

    result = validate_traffic_request_data(request)

    assert result["vehicle_counts"] == {
        "north": 15,
        "south": 12,
        "east": 14,
        "west": 13,
    }

    assert result["waiting_times"] == {
        "north": 40.0,
        "south": 35.0,
        "east": 38.0,
        "west": 36.0,
    }


def test_explicit_zero_is_accepted():

    request = """
    Analyse this traffic:

    North: 0 vehicles, waiting 0 seconds
    South: 0 vehicles, waiting 0 seconds
    East: 0 vehicles, waiting 0 seconds
    West: 0 vehicles, waiting 0 seconds
    """

    result = validate_traffic_request_data(request)

    assert all(
        count == 0
        for count in result["vehicle_counts"].values()
    )


def test_partial_traffic_information_is_rejected():

    request = """
    Analyse this traffic:

    North: 15 vehicles, waiting 40 seconds
    South: 12 vehicles, waiting 35 seconds
    """

    try:
        validate_traffic_request_data(request)

    except ValueError as error:
        assert "INSUFFICIENT_TRAFFIC_INFORMATION" in str(error)

    else:
        raise AssertionError(
            "Incomplete traffic information was accepted."
        )


if __name__ == "__main__":

    test_missing_all_traffic_information()
    test_complete_traffic_information()
    test_explicit_zero_is_accepted()
    test_partial_traffic_information_is_rejected()

    print("All 4 traffic validation tests passed!")
