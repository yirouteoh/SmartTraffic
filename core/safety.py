
"""
SmartTraffic - Safety Guardrails

Responsibilities:
1. Validate traffic signal transitions.
2. Prevent conflicting traffic phases.
3. Detect basic prompt-injection attempts.
4. Block unsupported real-world traffic-control requests.
5. Validate proposed signal configurations.
"""

import re


# ==========================================
# 1. VALID CONFIGURATIONS
# ==========================================

VALID_PHASES = ("NS", "EW")

VALID_SIGNAL_STATES = (
    "GREEN",
    "YELLOW",
    "ALL_RED",
)


# ==========================================
# 2. USER REQUEST SAFETY CHECK
# ==========================================

def check_user_request_safety(user_input: str) -> dict:
    """
    Check whether a user request is permitted.

    This is a baseline rule-based guardrail.
    It does not replace human review or comprehensive
    prompt-injection detection.
    """

    if not isinstance(user_input, str):
        return {
            "allowed": False,
            "category": "INVALID_INPUT",
            "reason": "User input must be text.",
        }

    if not user_input.strip():
        return {
            "allowed": False,
            "category": "INVALID_INPUT",
            "reason": "User input cannot be empty.",
        }

    text = user_input.lower()

    # Basic prompt-injection patterns.
    injection_patterns = [
        r"\bignore\b.{0,40}\b(previous|prior|above|all)\b.{0,30}\b(instructions?|rules?|prompts?)\b",
        r"\b(disregard|override|bypass)\b.{0,40}\b(instructions?|rules?|guardrails?)\b",
        r"\breveal\b.{0,30}\b(system prompt|hidden instructions|api key|secret)\b",
        r"\bact as\b.{0,30}\b(unrestricted|unfiltered|without rules)\b",
    ]

    for pattern in injection_patterns:

        if re.search(pattern, text):

            return {
                "allowed": False,
                "category": "PROMPT_INJECTION",
                "reason": (
                    "The request contains an attempt to "
                    "override SmartTraffic instructions."
                ),
            }

    # Block requests that attempt to control actual
    # external traffic infrastructure.
    real_world_patterns = [
        r"\b(real|actual|live)\b.{0,30}\b(traffic lights?|traffic signals?|intersection)\b.{0,40}\b(control|activate|change|override|operate)\b",
        r"\b(control|activate|override|operate)\b.{0,40}\b(real|actual|live)\b.{0,30}\b(traffic lights?|traffic signals?)\b",
        r"\bconnect\b.{0,30}\b(real traffic controller|traffic signal hardware)\b",
    ]

    for pattern in real_world_patterns:

        if re.search(pattern, text):

            return {
                "allowed": False,
                "category": "UNSUPPORTED_ACTION",
                "reason": (
                    "SmartTraffic is a simulation system "
                    "and cannot control real-world traffic infrastructure."
                ),
            }

    # Block explicit unsafe simultaneous green requests.
    unsafe_patterns = [
        r"\b(all|every)\b.{0,20}\b(traffic lights?|signals?)\b.{0,30}\b(green|go)\b",
        r"\b(make|set|turn)\b.{0,30}\b(all|every)\b.{0,20}\b(traffic lights?|signals?)\b.{0,20}\bgreen\b",
        r"\bactivate\b.{0,30}\b(both|all)\b.{0,20}\bphases\b.{0,30}\bsimultaneously\b",
    ]

    for pattern in unsafe_patterns:

        if re.search(pattern, text):

            return {
                "allowed": False,
                "category": "UNSAFE_ACTION",
                "reason": (
                    "Simultaneously activating conflicting "
                    "traffic phases is not permitted."
                ),
            }

    return {
        "allowed": True,
        "category": "PERMITTED",
        "reason": "Request passed the basic safety checks.",
    }


# ==========================================
# 3. SIGNAL TRANSITION VALIDATION
# ==========================================

def validate_signal_action(
    current_phase: str,
    current_signal_state: str,
    requested_phase: str,
    requested_signal_state: str,
    pedestrian_crossing_active: bool = False,
) -> dict:

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

    if not isinstance(pedestrian_crossing_active, bool):
        return {
            "allowed": False,
            "reason": "Pedestrian crossing status must be boolean.",
        }

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

    # GREEN -> YELLOW
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

    # YELLOW -> ALL_RED
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

    # ALL_RED -> GREEN
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


# ==========================================
# 4. CONFLICTING PHASE VALIDATION
# ==========================================

def check_conflicting_phases(active_phases: list) -> dict:

    if not isinstance(active_phases, list):
        return {
            "allowed": False,
            "reason": "Active phases must be provided as a list.",
        }

    if any(
        phase not in VALID_PHASES
        for phase in active_phases
    ):
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


# ==========================================
# 5. SIGNAL CONFIGURATION VALIDATION
# ==========================================

def validate_signal_configuration(
    active_phases: list,
    signal_state: str,
) -> dict:
    """Validate the proposed intersection signal configuration."""

    if signal_state not in VALID_SIGNAL_STATES:
        return {
            "allowed": False,
            "reason": "Invalid signal state.",
        }

    phase_check = check_conflicting_phases(active_phases)

    if not phase_check["allowed"]:
        return phase_check

    if signal_state == "GREEN" and len(active_phases) != 1:
        return {
            "allowed": False,
            "reason": (
                "Exactly one compatible phase must be active "
                "during a green signal."
            ),
        }

    if signal_state in ("YELLOW", "ALL_RED") and len(active_phases) > 1:
        return {
            "allowed": False,
            "reason": "Conflicting phases are not permitted.",
        }

    return {
        "allowed": True,
        "reason": "Signal configuration is valid.",
    }
