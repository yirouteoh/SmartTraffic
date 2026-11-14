
import pytest

from core.safety import (
    validate_signal_action,
    check_conflicting_phases,
)


def test_valid_green_to_yellow():
    result = validate_signal_action(
        "NS", "GREEN", "NS", "YELLOW"
    )

    assert result["allowed"] is True


def test_valid_yellow_to_all_red():
    result = validate_signal_action(
        "NS", "YELLOW", "NS", "ALL_RED"
    )

    assert result["allowed"] is True


def test_valid_all_red_to_next_green():
    result = validate_signal_action(
        "NS", "ALL_RED", "EW", "GREEN"
    )

    assert result["allowed"] is True


def test_reject_direct_green_to_green():
    result = validate_signal_action(
        "NS", "GREEN", "EW", "GREEN"
    )

    assert result["allowed"] is False


def test_reject_green_during_pedestrian_crossing():
    result = validate_signal_action(
        "NS", "ALL_RED", "EW", "GREEN",
        pedestrian_crossing_active=True,
    )

    assert result["allowed"] is False


def test_reject_invalid_phase():
    result = validate_signal_action(
        "NS", "ALL_RED", "NE", "GREEN"
    )

    assert result["allowed"] is False


def test_reject_both_phases_active():
    result = check_conflicting_phases(["NS", "EW"])

    assert result["allowed"] is False


def test_allow_one_active_phase():
    result = check_conflicting_phases(["NS"])

    assert result["allowed"] is True
