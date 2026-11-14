
"""
SmartTraffic - Safety Guardrails

Enforces safe traffic signal transitions before
the simulator executes a requested action.
"""

VALID_PHASES = ("NS", "EW")

VALID_SIGNAL_STATES = (
    "GREEN",
    "YELLOW",
    "ALL_RED",
)


def validate_signal_action(
    current_phase: str,
    current_signal_state: str,
    requested_phase: str,
    requested_signal_state: str,
    pedestrian_crossing_active: bool = False,
) -> dict:
    """
    Validate a requested traffic signal transition.

    Permitted transitions:
        GREEN  -> YELLOW
        YELLOW -> ALL_RED
        ALL_RED -> GREEN

    A new vehicle green phase cannot start while
    a pedestrian crossing is active.
    """

    # 1. Validate traffic phases.
    if current_phase not in VALID_PHASES:
        return {
            "allowed": False,
            "reason": "Invalid current traffic phase.",
        }

    if requested_phase not in VALID_PHASES:
        return {
            "allowed": False,
            "reason": "Invalid requested traffic phase.",
        }

    # 2. Validate signal states.
    if current_signal_state not in VALID_SIGNAL_STATES:
        return {
            "allowed": False,
            "reason": "Invalid current signal state.",
        }

    if requested_signal_state not in VALID_SIGNAL_STATES:
        return {
            "allowed": False,
            "reason": "Invalid requested signal state.",
        }

    # 3. Validate pedestrian request data.
    if not isinstance(pedestrian_crossing_active, bool):
        return {
            "allowed": False,
            "reason": "Pedestrian crossing status must be boolean.",
        }

    # 4. Prevent unsafe green activation.
    if (
        requested_signal_state == "GREEN"
        and pedestrian_crossing_active
    ):
        return {
            "allowed": False,
            "reason": (
                "Vehicle green phase blocked because "
                "pedestrian crossing is active."
            ),
        }

    # 5. GREEN must transition to YELLOW.
    if current_signal_state == "GREEN":

        if (
            requested_signal_state == "YELLOW"
            and requested_phase == current_phase
        ):
            return {
                "allowed": True,
                "reason": "Valid GREEN to YELLOW transition.",
            }

        return {
            "allowed": False,
            "reason": (
                "GREEN must transition to YELLOW "
                "before changing traffic phases."
            ),
        }

    # 6. YELLOW must transition to ALL_RED.
    if current_signal_state == "YELLOW":

        if (
            requested_signal_state == "ALL_RED"
            and requested_phase == current_phase
        ):
            return {
                "allowed": True,
                "reason": "Valid YELLOW to ALL_RED transition.",
            }

        return {
            "allowed": False,
            "reason": (
                "YELLOW must transition to ALL_RED "
                "before another green phase begins."
            ),
        }

    # 7. ALL_RED may activate the next green phase.
    if current_signal_state == "ALL_RED":

        if requested_signal_state == "GREEN":
            return {
                "allowed": True,
                "reason": "Valid ALL_RED to GREEN transition.",
            }

        return {
            "allowed": False,
            "reason": (
                "ALL_RED must transition to GREEN "
                "to activate a vehicle phase."
            ),
        }

    return {
        "allowed": False,
        "reason": "Unrecognised signal transition.",
    }


def check_conflicting_phases(active_phases: list) -> dict:
    """
    Prevent North-South and East-West phases
    from being active simultaneously.
    """

    if not isinstance(active_phases, list):
        return {
            "allowed": False,
            "reason": "Active phases must be provided as a list.",
        }

    if any(phase not in VALID_PHASES for phase in active_phases):
        return {
            "allowed": False,
            "reason": "Unknown traffic phase detected.",
        }

    if len(active_phases) != len(set(active_phases)):
        return {
            "allowed": False,
            "reason": "Duplicate active phases are not permitted.",
        }

    if len(active_phases) > 1:
        return {
            "allowed": False,
            "reason": (
                "Conflicting North-South and East-West "
                "phases cannot be active simultaneously."
            ),
        }

    return {
        "allowed": True,
        "reason": "Traffic phase configuration is valid.",
    }
