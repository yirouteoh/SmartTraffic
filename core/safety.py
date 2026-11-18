
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

    Detects:
    1. Prompt injection attempts.
    2. Unsupported real-world traffic-control requests.
    3. Unsafe simultaneous traffic-phase activation.

    Safety checks are enforced in Python before the LLM
    receives the user request.
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

    # ==========================================
    # 1. PROMPT-INJECTION DETECTION
    # ==========================================

    injection_patterns = [

        # Ignore or override existing instructions.
        r"\b(ignore|disregard|forget|override)\b.{0,60}\b(previous|prior|above|system|developer|all)\b.{0,40}\b(instructions?|rules?|prompts?|policies?)\b",

        # Bypass or disable safety mechanisms.
        r"\b(bypass|disable|circumvent|evade|remove|turn off)\b.{0,60}\b(safety\s+(checks?|rules?|restrictions?|controls?)|guardrails?|security\s+checks?|instructions?|rules?)\b",

        # Reveal hidden instructions or confidential information.
        r"\b(reveal|show|print|display|expose|repeat|provide)\b.{0,60}\b(hidden\s+(system\s+)?instructions?|system\s+prompt|developer\s+instructions?|api\s+keys?|secrets?)\b",

        # Attempt to change the agent's identity or restrictions.
        r"\bact as\b.{0,40}\b(unrestricted|unfiltered|without rules|without restrictions)\b",

        # Developer-mode and unrestricted-mode attempts.
        r"\b(enable|activate|enter|switch to)\b.{0,40}\b(developer mode|unrestricted mode|jailbreak mode)\b",

    ]

    for pattern in injection_patterns:

        if re.search(pattern, text):

            return {
                "allowed": False,
                "category": "PROMPT_INJECTION",
                "reason": (
                    "The request contains an attempt to bypass "
                    "SmartTraffic instructions or safety restrictions."
                ),
            }

    # ==========================================
    # 2. UNSUPPORTED REAL-WORLD CONTROL
    # ==========================================

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

    # ==========================================
    # 3. UNSAFE SIMULTANEOUS PHASE ACTIVATION
    # ==========================================

    # Recognise North-South and East-West phases.
    ns_pattern = (
        r"\b(ns|north[\s-]*south|north\s+(and|&)\s+south)\b"
    )

    ew_pattern = (
        r"\b(ew|east[\s-]*west|east\s+(and|&)\s+west)\b"
    )

    # Recognise commands that attempt to activate signals.
    activation_pattern = (
        r"\b(activate|enable|turn\s+on|switch\s+on|set|make|"
        r"keep|maintain|run|operate|allow|give)\b"
    )

    # Recognise simultaneous activation requests.
    simultaneous_pattern = (
        r"\b(simultaneously|at\s+the\s+same\s+time|together|both)\b"
    )

    has_ns = bool(re.search(ns_pattern, text))
    has_ew = bool(re.search(ew_pattern, text))

    requests_activation = bool(
        re.search(activation_pattern, text)
    )

    requests_simultaneous = bool(
        re.search(simultaneous_pattern, text)
    )

    requests_green = bool(
        re.search(r"\b(green|go)\b", text)
    )

    # Both conflicting phases must not receive green together.
    if (
        has_ns
        and has_ew
        and requests_activation
        and (requests_simultaneous or requests_green)
    ):

        return {
            "allowed": False,
            "category": "UNSAFE_ACTION",
            "reason": (
                "Simultaneously activating conflicting "
                "North-South and East-West traffic phases "
                "is not permitted."
            ),
        }

    # ==========================================
    # 4. EXISTING GENERAL UNSAFE SIGNAL REQUESTS
    # ==========================================

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

    # ==========================================
    # 5. REQUEST PASSED PYTHON SAFETY CHECKS
    # ==========================================

    return {
        "allowed": True,
        "category": "PERMITTED",
        "reason": "Request passed the Python safety checks.",
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
