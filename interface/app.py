"""
SmartTraffic - Intelligent Traffic Decision Assistant

Integrated architecture:

User Request
    -> Qwen AI Agent
    -> Python Traffic Tools
    -> Dashboard Controller
    -> Individual Vehicle Traffic Simulator
    -> Directional SVG Traffic Visualisation

All displayed car positions and signal states come from the
actual TrafficSimulator backend.

CSS is loaded from styles/dashboard.css.
"""

import time
import html

from pathlib import Path

import spaces
import gradio as gr

from agent.llm_agent import SmartTrafficAgent
from simulation.dashboard_controller import DashboardController


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSS_PATH = PROJECT_ROOT / "styles" / "dashboard.css"


# =====================================================
# 2. LOAD EXTERNAL CSS
# =====================================================

if not CSS_PATH.is_file():
    raise FileNotFoundError(
        f"Dashboard CSS file not found: {CSS_PATH}"
    )

CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8")


# =====================================================
# 3. INITIALISATION
# =====================================================

# =====================================================
# ZEROGPU DIAGNOSTIC
# =====================================================

@spaces.GPU(duration=30)
def check_gpu_status():
    """Check whether ZeroGPU can allocate a CUDA device."""

    import torch

    test_tensor = torch.ones(1, device="cuda")

    return (
        "ZeroGPU is working successfully! "
        f"Device: {test_tensor.device}"
    )
agent = SmartTrafficAgent()

# Retain the most recent successfully completed simulation data.
LAST_VALID_SIMULATION_DATA = None

# Preserve the last successfully served traffic phase.
# None means that no previous phase is available yet.
LAST_SERVED_PHASE = None

DEFAULT_SCENARIO = (25, 90, 18, 65, 8, 25, 5, 15)

PRESETS = {
    "Rush hour (NS heavy)": (25, 90, 18, 65, 8, 25, 5, 15),
    "Balanced": (12, 40, 11, 38, 12, 42, 10, 35),
    "Event exit (EW heavy)": (6, 20, 8, 25, 28, 110, 22, 95),
    "Quiet night": (2, 10, 1, 8, 2, 12, 1, 6),
}

# Animation speed.
ANIMATION_DELAY = 0.12


# =====================================================
# 4. SVG INTERSECTION CONFIGURATION
# =====================================================

CANVAS_SIZE = 1500
CENTER = 750
ROAD_WIDTH = 200
ROAD_START = 650
ROAD_END = 850
POSITION_SCALE = 2.8

# The simulation is drawn on a 1500 x 1500 canvas, but only the
# central part is shown so the roads, cars and lights appear larger.
VIEW_BOX = "150 150 1200 1200"

# Car body dimensions.
CAR_LENGTH = 38
CAR_WIDTH = 20

# Incoming lane coordinates.
#
# North: moves downward.
# South: moves upward.
# East: moves left.
# West: moves right.

LANE_COORDINATES = {
    "north": {"x": 790, "rotation": 90},
    "south": {"x": 710, "rotation": 270},
    "east": {"y": 710, "rotation": 180},
    "west": {"y": 790, "rotation": 0},
}

# Traffic light locations. Each light sits beside the road it controls
# (on the driver's right side), and the four lights do not overlap.

TRAFFIC_LIGHT_POSITIONS = {
    "north": {"x": 915, "y": 430, "label": "NORTH"},
    "south": {"x": 585, "y": 1070, "label": "SOUTH"},
    "east": {"x": 1010, "y": 520, "label": "EAST"},
    "west": {"x": 490, "y": 980, "label": "WEST"},
}

SIGNAL_COLOURS = {
    "RED": "#ef4444",
    "YELLOW": "#eab308",
    "GREEN": "#22c55e",
}

CAR_COLOURS = [
    "#3b82f6",
    "#ef4444",
    "#f59e0b",
    "#22c55e",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#e879f9",
]


# =====================================================
# 5. STATUS DISPLAY
# =====================================================

def status_html(kind, title, lines, loading=False):
    """Create a dashboard status message safely.

    loading=True shows a spinner beside the title so the user can see
    that the app is working and not frozen.
    """

    if kind not in {"ok", "warn", "bad"}:
        kind = "warn"

    body = "<br>".join(html.escape(str(line)) for line in lines)

    spinner = '<span class="st-spinner"></span>' if loading else ""

    return (
        f'<div class="st-status st-{kind}">'
        f'<div class="st-status-title">{spinner}{html.escape(str(title))}</div>'
        f'{body}'
        f'</div>'
    )


IDLE_STATUS = status_html(
    "warn",
    "READY",
    ["Enter traffic conditions to begin analysis."]
)


def section_title(number, text):
    """Numbered section heading."""

    return (
        f'<div class="st-section">'
        f'<span class="st-num">{number}</span>'
        f'{html.escape(text)}'
        f'</div>'
    )


# =====================================================
# 6. TRAFFIC DATA VALIDATION
# =====================================================

def validate_traffic_data(vehicle_counts, waiting_times):
    """Validate traffic information from Python tool logs."""

    if not isinstance(vehicle_counts, dict):
        raise ValueError("Invalid vehicle count data.")

    if not isinstance(waiting_times, dict):
        raise ValueError("Invalid waiting time data.")

    required = {"north", "south", "east", "west"}

    if set(vehicle_counts) != required:
        raise ValueError("All four vehicle directions are required.")

    if set(waiting_times) != required:
        raise ValueError("All four waiting-time directions are required.")

    for direction in required:

        count = vehicle_counts[direction]
        waiting = waiting_times[direction]

        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= 10000
        ):
            raise ValueError(f"Invalid vehicle count for {direction}.")

        if (
            isinstance(waiting, bool)
            or not isinstance(waiting, (int, float))
            or not 0 <= waiting <= 86400
        ):
            raise ValueError(f"Invalid waiting time for {direction}.")


# =====================================================
# 7. RENDER TRAFFIC CARDS
# =====================================================

def render_cards(vehicle_counts=None, waiting_times=None):
    """Render traffic cards only when real data is supplied."""

    if vehicle_counts is None or waiting_times is None:
        return (
            '<div class="st-panel">'
            '<b>NO CURRENT TRAFFIC DATA</b>'
            '<div>Submit a valid traffic request to display current conditions.</div>'
            '</div>'
        )

    validate_traffic_data(vehicle_counts, waiting_times)
    cards = []

    for direction in ("north", "south", "east", "west"):
        count = vehicle_counts[direction]
        waiting = waiting_times[direction]

        if count >= 20 or waiting >= 80:
            level, tag = "high", "HEAVY"
        elif count >= 10 or waiting >= 40:
            level, tag = "mid", "MODERATE"
        else:
            level, tag = "low", "LIGHT"

        percentage = min(count / 30 * 100, 100)
        cards.append(f"""
        <div class="st-dir lvl-{level}">
            <div class="st-dir-name">{direction.upper()} - {tag}</div>
            <div class="st-dir-count">{count} <small>cars</small></div>
            <div class="st-dir-wait">Waiting {waiting:.1f} seconds</div>
            <div class="st-bar"><span style="width:{percentage:.0f}%"></span></div>
        </div>
        """)

    return '<div class="st-grid">' + "".join(cards) + "</div>"


def render_historical_cards(data=None):
    """Render the last successful simulation as clearly labelled history."""

    if not data:
        return (
            '<div class="st-panel">'
            '<b>LAST VALID SIMULATION DATA</b>'
            '<div>No previous successful simulation data is available.</div>'
            '</div>'
        )

    cards = render_cards(
        data["vehicle_counts"],
        data["waiting_times"],
    )
    timestamp = html.escape(str(data.get("label", "Previous successful simulation")))
    return (
        '<div class="st-panel st-history">'
        '<b>LAST VALID SIMULATION DATA — HISTORICAL</b>'
        f'<div>{timestamp}. This is not the result of the current request.</div>'
        '</div>'
        + cards
    )


# =====================================================
# 8. DRAW TRAFFIC SIGNAL
# =====================================================

def draw_signal(x, y, state, label):
    """
    Draw one individual traffic signal.

    state: RED | YELLOW | GREEN
    label: NORTH | SOUTH | EAST | WEST
    """

    lamps = [
        ("RED", SIGNAL_COLOURS["RED"], 0),
        ("YELLOW", SIGNAL_COLOURS["YELLOW"], 1),
        ("GREEN", SIGNAL_COLOURS["GREEN"], 2),
    ]

    safe_label = html.escape(str(label))

    parts = [
        f'<g transform="translate({x},{y}) scale(1.5)">',

        # Signal housing.
        '<rect x="-17" y="-40" width="34" height="94" '
        'rx="9" fill="#0b121b" '
        'stroke="#cbd5e1" stroke-width="2.5"/>'
    ]

    for name, colour, index in lamps:

        opacity = 1.0 if state == name else 0.18
        cy = -25 + index * 28

        parts.append(
            f'<circle cx="0" cy="{cy}" r="10" '
            f'fill="{colour}" opacity="{opacity}"/>'
        )

    # Directional label.
    parts.append(
        f'<text x="0" y="75" fill="#ffffff" font-size="15" '
        f'font-weight="bold" text-anchor="middle" '
        f'style="paint-order:stroke;stroke:#0b1220;stroke-width:3px;">'
        f'{safe_label}</text>'
    )

    parts.append("</g>")

    return "".join(parts)


# =====================================================
# 9. DETERMINE ACTUAL SIGNAL STATES
# =====================================================

def get_directional_signal_states(status=None):
    """
    Translate the backend phase and signal state
    into four individual directional signal states.

    NS: North and South operate together.
    EW: East and West operate together.
    """

    all_red = {
        "north": "RED",
        "south": "RED",
        "east": "RED",
        "west": "RED",
    }

    if status is None:
        return all_red

    phase = status["current_phase"]
    signal = status["current_signal_state"]

    if signal == "ALL_RED":
        return all_red

    if phase == "NS":
        return {
            "north": signal,
            "south": signal,
            "east": "RED",
            "west": "RED",
        }

    if phase == "EW":
        return {
            "north": "RED",
            "south": "RED",
            "east": signal,
            "west": signal,
        }

    raise ValueError("Invalid traffic phase returned by simulator.")


# =====================================================
# 10. DRAW INDIVIDUAL CAR
# =====================================================

def draw_car(vehicle, index):
    """
    Draw one individual car using its actual
    TrafficSimulator backend record.

    Backend fields:
        vehicle_id, direction, position, status, speed

    Position represents distance from the stop line.
    """

    direction = vehicle["direction"]
    position = float(vehicle["position"])
    vehicle_id = str(vehicle["vehicle_id"])
    status = vehicle["status"]

    if direction not in LANE_COORDINATES:
        return ""

    # Convert backend distance into SVG coordinates.
    distance = position * POSITION_SCALE

    lane = LANE_COORDINATES[direction]
    rotation = lane["rotation"]

    # Account for the half-length of the car body so the car's front
    # is aligned with the stop line when the backend position is zero.

    if direction == "north":
        x = lane["x"]
        y = 635 - distance - CAR_LENGTH / 2

    elif direction == "south":
        x = lane["x"]
        y = 865 + distance + CAR_LENGTH / 2

    elif direction == "west":
        x = 635 - distance - CAR_LENGTH / 2
        y = lane["y"]

    else:
        x = 865 + distance + CAR_LENGTH / 2
        y = lane["y"]

    colour = CAR_COLOURS[index % len(CAR_COLOURS)]

    outline = "#86efac" if status == "MOVING" else "#e2e8f0"

    safe_title = html.escape(f"{vehicle_id} | {direction} | {status}")

    return f"""
    <g transform="translate({x:.2f},{y:.2f}) rotate({rotation})">
        <title>{safe_title}</title>
        <rect x="-19" y="-10" width="38" height="20" rx="6"
              fill="{colour}" stroke="{outline}" stroke-width="2"/>
        <rect x="6" y="-7" width="8" height="14" rx="2" fill="#bae6fd" opacity="0.9"/>
        <rect x="-13" y="-7" width="7" height="14" rx="2" fill="#bae6fd" opacity="0.7"/>
        <rect x="-2" y="-8" width="5" height="16" rx="1" fill="#172033" opacity="0.55"/>
        <rect x="17" y="-7" width="3" height="4" rx="1" fill="#fef3c7"/>
        <rect x="17" y="3" width="3" height="4" rx="1" fill="#fef3c7"/>
        <rect x="-12" y="-12" width="9" height="3" rx="1" fill="#111827"/>
        <rect x="8" y="-12" width="9" height="3" rx="1" fill="#111827"/>
        <rect x="-12" y="9" width="9" height="3" rx="1" fill="#111827"/>
        <rect x="8" y="9" width="9" height="3" rx="1" fill="#111827"/>
    </g>
    """


# =====================================================
# 11. RENDER CAR COLLECTION
# =====================================================

def render_cars(vehicles):
    """Render every individual car returned by the simulator."""

    if not vehicles:
        return ""

    return "".join(
        draw_car(vehicle, index)
        for index, vehicle in enumerate(vehicles)
    )


# =====================================================
# 12. RENDER LIVE INTERSECTION
# =====================================================

def render_intersection(status=None):
    """
    Render the complete four-way intersection.

    Traffic lights are controlled by the actual Python simulator state.
    Individual car positions come from TrafficSimulator vehicle records.
    """

    if status is None:
        phase = "NONE"
        signal = "RED"
        vehicles = []
        simulation_time = 0
        remaining = 0
        cycle_status = "IDLE"
        badge_class = "st-badge-idle"
        badge_text = "WAITING"

    else:
        phase = status["current_phase"]
        signal = status["current_signal_state"]
        vehicles = status.get("vehicles", [])
        simulation_time = status["simulation_time"]
        remaining = status["remaining_time"]
        cycle_status = status["cycle_status"]

        badge_class = {
            "GREEN": (
                "st-badge-ns" if phase == "NS"
                else "st-badge-ew"
            ),
            "YELLOW": "st-badge-yellow",
            "ALL_RED": "st-badge-red",
        }.get(signal, "st-badge-idle")

        if cycle_status == "COMPLETED":
            badge_text = f"CYCLE COMPLETED - LAST SIGNAL: {signal}"
        elif cycle_status == "BLOCKED":
            badge_text = "SAFETY BLOCKED"
        else:
            badge_text = f"{phase} - {signal}"

    # Four actual signal states and individual cars.
    signal_states = get_directional_signal_states(status)
    car_svg = render_cars(vehicles)
    car_count = len(vehicles)

    lights_svg = "".join(
        draw_signal(
            position["x"],
            position["y"],
            signal_states[direction],
            position["label"],
        )
        for direction, position in TRAFFIC_LIGHT_POSITIONS.items()
    )

    # Legend: direction, its phase and its current light colour.
    legend_html = "".join(
        f'<div class="st-legend-item">'
        f'<b><span class="st-dot" style="background:'
        f'{SIGNAL_COLOURS.get(signal_states[direction], "#ef4444")};">'
        f'</span>{direction.upper()}</b>'
        f'<span>{"NS" if direction in ("north", "south") else "EW"} phase</span>'
        f'</div>'
        for direction in ("north", "south", "east", "west")
    )

    crossings = "".join(
        [
            "".join(
                f'<rect x="{660 + i * 24}" y="610" width="13" height="17" rx="2"/>'
                for i in range(7)
            ),
            "".join(
                f'<rect x="{660 + i * 24}" y="873" width="13" height="17" rx="2"/>'
                for i in range(7)
            ),
            "".join(
                f'<rect x="610" y="{660 + i * 24}" width="17" height="13" rx="2"/>'
                for i in range(7)
            ),
            "".join(
                f'<rect x="873" y="{660 + i * 24}" width="17" height="13" rx="2"/>'
                for i in range(7)
            ),
        ]
    )

    label_style = (
        "paint-order:stroke;stroke:#0b1220;stroke-width:7px;"
        "stroke-linejoin:round;"
    )

    return f"""
    <div class="st-intersection">

        <div class="st-int-head">
            <span class="st-int-title">LIVE FOUR-WAY INTERSECTION</span>
            <span class="st-badge {badge_class}">{html.escape(badge_text)}</span>
        </div>

        <div class="st-legend">{legend_html}</div>

        <div class="st-svg-wrap">
        <svg class="st-svg" viewBox="{VIEW_BOX}"
             role="img"
             aria-label="Four-way intersection with North South East West traffic lights and individual cars">

            <defs>
                <clipPath id="road-clip">
                    <rect x="0" y="0" width="1500" height="1500"/>
                </clipPath>
            </defs>

            <g clip-path="url(#road-clip)">

                <!-- Grass -->
                <rect x="0" y="0" width="1500" height="1500" fill="#1d4a3c"/>

                <!-- Roads -->
                <rect x="0" y="650" width="1500" height="200" fill="#3a4452"/>
                <rect x="650" y="0" width="200" height="1500" fill="#3a4452"/>

                <!-- Road edges -->
                <g stroke="#e2e8f0" stroke-width="5" fill="none">
                    <line x1="0" y1="645" x2="650" y2="645"/>
                    <line x1="850" y1="645" x2="1500" y2="645"/>
                    <line x1="0" y1="855" x2="650" y2="855"/>
                    <line x1="850" y1="855" x2="1500" y2="855"/>
                    <line x1="645" y1="0" x2="645" y2="650"/>
                    <line x1="855" y1="0" x2="855" y2="650"/>
                    <line x1="645" y1="850" x2="645" y2="1500"/>
                    <line x1="855" y1="850" x2="855" y2="1500"/>
                </g>

                <!-- Centre markings -->
                <g stroke="#f8fafc" stroke-width="4" stroke-dasharray="24 20" opacity="0.8">
                    <line x1="0" y1="750" x2="645" y2="750"/>
                    <line x1="855" y1="750" x2="1500" y2="750"/>
                    <line x1="750" y1="0" x2="750" y2="645"/>
                    <line x1="750" y1="855" x2="750" y2="1500"/>
                </g>

                <!-- Stop lines -->
                <g stroke="#ffffff" stroke-width="7">
                    <line x1="650" y1="635" x2="850" y2="635"/>
                    <line x1="650" y1="865" x2="850" y2="865"/>
                    <line x1="635" y1="650" x2="635" y2="850"/>
                    <line x1="865" y1="650" x2="865" y2="850"/>
                </g>

                <!-- Pedestrian crossings -->
                <g fill="#f8fafc" opacity="0.85">{crossings}</g>

                <!-- Intersection centre -->
                <rect x="650" y="650" width="200" height="200" fill="#323b47" opacity="0.85"/>
                <circle cx="750" cy="750" r="43" fill="#1e293b" stroke="#94a3b8" stroke-width="3"/>
                <text x="750" y="760" fill="white" font-size="28" font-weight="bold" text-anchor="middle">ST</text>

                <!-- Compass -->
                <g transform="translate(250,270)">
                    <circle r="44" fill="#0b1220" opacity="0.65" stroke="#94a3b8" stroke-width="2"/>
                    <path d="M 0,-28 L -13,10 L 0,2 L 13,10 Z" fill="#ffffff"/>
                    <text x="0" y="-52" fill="#ffffff" font-size="26" font-weight="bold" text-anchor="middle" style="{label_style}">N</text>
                </g>

                <!-- Direction labels -->
                <g fill="#ffffff" font-size="38" font-weight="bold" style="{label_style}">
                    <text x="880" y="215">NORTH</text>
                    <text x="620" y="1325" text-anchor="end">SOUTH</text>
                    <text x="170" y="915">WEST</text>
                    <text x="1330" y="605" text-anchor="end">EAST</text>
                </g>

                <!-- Four individually labelled traffic lights -->
                {lights_svg}

                <!-- Individual cars from backend -->
                {car_svg}

            </g>

        </svg>
        </div>

        <div class="st-int-info">
            <span>Simulation time: <b>{simulation_time} s</b></span>
            <span>Individual cars: <b>{car_count}</b></span>
            <span>Remaining green time: <b>{remaining} s</b></span>
        </div>

        <div class="st-key">
            <span><span style="color:#ef4444;">●</span> RED - Stop</span>
            <span><span style="color:#eab308;">●</span> YELLOW - Transition</span>
            <span><span style="color:#22c55e;">●</span> GREEN - Active phase</span>
        </div>

        <div class="st-note">
            Signals show the actual Python simulator state.
            Car positions come from backend records.
        </div>

    </div>
    """


# =====================================================
# 13. PERFORMANCE METRICS
# =====================================================

def render_metrics(metrics=None):

    if metrics is None:
        return """
        <div class="st-panel">
            No simulation data available yet.
        </div>
        """

    return f"""
    <div class="st-grid">

        <div class="st-metric">
            <div class="st-metric-label">SIMULATION TIME</div>
            <div class="st-metric-value">{metrics["simulation_time"]} s</div>
        </div>

        <div class="st-metric">
            <div class="st-metric-label">CARS WAITING</div>
            <div class="st-metric-value">{metrics["total_vehicles_waiting"]}</div>
        </div>

        <div class="st-metric">
            <div class="st-metric-label">TOTAL ARRIVALS</div>
            <div class="st-metric-value">{metrics["total_arrivals"]}</div>
        </div>

        <div class="st-metric">
            <div class="st-metric-label">TOTAL DEPARTURES</div>
            <div class="st-metric-value">{metrics["total_departures"]}</div>
        </div>

    </div>

    <div class="st-metric-foot">
        Average waiting time: <b>{metrics["average_waiting_time"]} s</b><br>
        Individual cars: <b>{metrics.get("total_individual_cars", 0)}</b>
        &nbsp;|&nbsp;
        Stage: <b>{metrics["cycle_stage"]}</b>
    </div>
    """


# =====================================================
# 14. EXTRACT ACTUAL PYTHON RESULTS
# =====================================================

def extract_traffic_results(result):
    """
    Extract actual arguments and results from
    the two successful Python tool executions.
    """

    if not result.get("success"):
        raise ValueError("AI analysis was not completed successfully.")

    decision = result.get("decision")
    timing = result.get("timing")
    logs = result.get("tool_calls", [])

    if not isinstance(decision, dict):
        raise ValueError("Traffic priority result is missing.")

    if not isinstance(timing, dict):
        raise ValueError("Green-time result is missing.")

    if not isinstance(logs, list) or len(logs) != 2:
        raise ValueError("Both Python functions must execute successfully.")

    if any(log.get("status") != "SUCCESS" for log in logs):
        raise ValueError("One or more Python functions failed.")

    priority_log = next(
        (log for log in logs if log["function_name"] == "traffic_priority"),
        None,
    )

    green_log = next(
        (log for log in logs if log["function_name"] == "green_time"),
        None,
    )

    if priority_log is None or green_log is None:
        raise ValueError("Required Python function execution logs are missing.")

    arguments = priority_log["arguments"]

    if not isinstance(arguments, dict):
        raise ValueError("Invalid traffic function arguments.")

    vehicles = arguments.get("vehicle_counts")
    waiting = arguments.get("waiting_times")

    validate_traffic_data(vehicles, waiting)

    selected_phase = result.get("selected_phase")
    green_duration = result.get("green_duration")

    if selected_phase not in ("NS", "EW"):
        raise ValueError("Invalid selected traffic phase.")

    if (
        isinstance(green_duration, bool)
        or not isinstance(green_duration, int)
        or not 15 <= green_duration <= 60
    ):
        raise ValueError("Invalid recommended green duration.")

    if decision.get("selected_phase") != selected_phase:
        raise ValueError("Python priority result does not match.")

    if timing.get("selected_phase") != selected_phase:
        raise ValueError("Green-time phase does not match priority result.")

    if timing.get("green_time_seconds") != green_duration:
        raise ValueError("Green-time calculation does not match.")

    return (
        vehicles,
        waiting,
        selected_phase,
        green_duration,
        decision,
        timing,
    )


# =====================================================
# 15. EXECUTION LOGS
# =====================================================

def build_execution_log(result, controller=None):
    """Build execution logs with distinct request and runtime statuses."""

    logs = {
        "AI model": result.get("model"),
        "Request status": result.get(
            "request_status",
            "REFUSED" if result.get("refused") else (
                "SUCCESS" if result.get("success") else "INCOMPLETE"
            ),
        ),
        "AI execution": result.get(
            "ai_execution",
            "NOT_REQUIRED" if result.get("refused") else "COMPLETED",
        ),
        "Python calls": result.get(
            "python_call_count", len(result.get("tool_calls", []))
        ),
        "Python function calls": result.get("tool_calls", []),
        "Selected phase": result.get("selected_phase"),
        "Recommended green duration": result.get("green_duration"),
        "Simulation status": (
            controller.simulator.cycle_status
            if controller is not None and controller.simulator is not None
            else result.get("simulation_status", "NOT_STARTED")
        ),
    }

    if controller is not None:
        data = controller.get_dashboard_data()
        logs["Simulation stage"] = data["cycle_stage"]
        logs["Current signal state"] = data["traffic_status"]["current_signal_state"]
        logs["Signal history"] = data["signal_history"]
        logs["Total individual cars"] = data["total_individual_cars"]
        logs["Vehicle history"] = data["vehicle_history"]
        logs["Performance metrics"] = data["performance_metrics"]

    return logs


# =====================================================
# 16. MAIN ONE-CLICK WORKFLOW
# =====================================================

def analyse_traffic(user_request):
    global LAST_VALID_SIMULATION_DATA
    global LAST_SERVED_PHASE

    """
    Main integrated workflow.

    User -> AI -> Python tools -> Dashboard controller
         -> Traffic simulator -> Live directional intersection
    """

    # STEP 1: VALIDATE USER REQUEST.

    if (
        not isinstance(user_request, str)
        or not user_request.strip()
    ):

        yield (
            status_html(
                "warn",
                "MISSING REQUEST",
                ["Please enter a traffic request."]
            ),

            "Please provide traffic information.",

            render_intersection(),

            render_historical_cards(LAST_VALID_SIMULATION_DATA),

            render_metrics(),

            {},

            None,
        )

        return

    # STEP 2: SHOW ANALYSIS STATUS.

    yield (
        status_html(
            "warn",
            "ANALYSING TRAFFIC",
            [
                "Qwen is processing your traffic request.",
                "Python tool execution is in progress."
            ],
            loading=True,
        ),

        "SmartTraffic is analysing your traffic conditions...",

        render_intersection(),

        render_historical_cards(LAST_VALID_SIMULATION_DATA),

        render_metrics(),

        {},

        None,
    )

    # STEP 3: CALL AI AGENT.

    try:

        result = agent.process_request(
            user_input=user_request,
            previous_phase=LAST_SERVED_PHASE,
        )

    except Exception as error:

        yield (
            status_html(
                "bad",
                "AI CONNECTION ERROR",
                [str(error)]
            ),

            "SmartTraffic could not connect to the AI service.",

            render_intersection(),

            render_historical_cards(LAST_VALID_SIMULATION_DATA),

            render_metrics(),

            {},

            None,
        )

        return

    # STEP 4: HANDLE REFUSAL.

    if result.get("refused"):

        yield (
            status_html(
                "warn",
                "REQUEST REFUSED",
                [
                    f"Category: {result.get('refusal_category')}",
                    "No simulation was started."
                ]
            ),

            result.get("response", "Request refused."),

            render_intersection(),

            render_historical_cards(LAST_VALID_SIMULATION_DATA),

            render_metrics(),

            build_execution_log(result),

            None,
        )

        return

    # STEP 5: HANDLE INCOMPLETE ANALYSIS.

    if not result.get("success"):

        yield (
            status_html(
                "warn",
                "INCOMPLETE ANALYSIS",
                [
                    "Required Python calculations "
                    "were not completed.",
                    "No simulation was started."
                ]
            ),

            result.get(
                "response",
                "More information is required."
            ),

            render_intersection(),

            render_historical_cards(LAST_VALID_SIMULATION_DATA),

            render_metrics(),

            build_execution_log(result),

            None,
        )

        return

    # STEP 6: EXTRACT ACTUAL PYTHON RESULTS.

    try:

        (
            vehicles,
            waiting,
            selected_phase,
            green_duration,
            decision,
            timing,
        ) = extract_traffic_results(result)

    except Exception as error:

        yield (
            status_html(
                "bad",
                "RESULT VALIDATION FAILED",
                [
                    str(error),
                    "Simulation was not started."
                ]
            ),

            "SmartTraffic rejected the returned results.",

            render_intersection(),

            render_historical_cards(LAST_VALID_SIMULATION_DATA),

            render_metrics(),

            build_execution_log(result),

            None,
        )

        return

    # STEP 7: INITIALISE DASHBOARD CONTROLLER.

    try:

        controller = DashboardController()

        controller.initialise(
            vehicle_counts=vehicles,
            waiting_times=waiting,
            traffic_density="medium",
            random_seed=42,
        )

        # Pass actual Python decision to simulator.
        controller.start_cycle(
            selected_phase=selected_phase,
            green_duration=green_duration,
            decision=decision,
            timing=timing,
            pedestrian_active=False,
        )

    except Exception as error:

        yield (
            status_html(
                "bad",
                "SIMULATION INITIALISATION FAILED",
                [str(error)]
            ),

            result.get("response", ""),

            render_intersection(),

            render_historical_cards(LAST_VALID_SIMULATION_DATA),

            render_metrics(),

            build_execution_log(result),

            None,
        )

        return

    # STEP 8: DISPLAY INITIAL CARS AND DECISION.

    initial_data = controller.get_dashboard_data()

    initial_status = initial_data["traffic_status"]

    yield (
        status_html(
            "ok",
            "AI DECISION CONFIRMED",
            [
                f"Selected phase: {selected_phase}",
                f"Green duration: {green_duration} seconds",
                f"Individual cars created: {initial_data['total_individual_cars']}",
                "Backend decision validated.",
                "Automatic simulation starting."
            ]
        ),

        result.get("response", ""),

        render_intersection(initial_status),

        render_cards(
            initial_status["vehicle_counts"],
            initial_status["waiting_times"],
        ),

        render_metrics(
            initial_data["performance_metrics"]
        ),

        build_execution_log(result, controller),

        controller,
    )

    time.sleep(ANIMATION_DELAY)

    # STEP 9: AUTOMATIC BACKEND ANIMATION.

    while controller.simulator.cycle_status == "RUNNING":

        try:

            # Advance actual Python simulator by one second.
            controller.step(seconds=1)

            data = controller.get_dashboard_data()

            traffic_status = data["traffic_status"]

            metrics = data["performance_metrics"]

            cycle_status = data["cycle_status"]

            cycle_stage = data["cycle_stage"]

            # HANDLE SAFETY BLOCK.

            if cycle_status == "BLOCKED":

                yield (
                    status_html(
                        "warn",
                        "SAFETY BLOCKED",
                        [
                            data["last_event"],
                            "Green activation was prevented."
                        ]
                    ),

                    result.get("response", ""),

                    render_intersection(traffic_status),

                    render_cards(
                        traffic_status["vehicle_counts"],
                        traffic_status["waiting_times"],
                    ),

                    render_metrics(metrics),

                    build_execution_log(result, controller),

                    controller,
                )

                return

            # DISPLAY ACTUAL CAR POSITIONS.

            yield (
                status_html(
                    "ok",
                    "SIMULATION RUNNING",
                    [
                        f"Signal: {traffic_status['current_signal_state']}",
                        f"Selected phase: {selected_phase}",
                        f"Stage: {cycle_stage}",
                        f"Simulation time: {traffic_status['simulation_time']} seconds",
                        f"Cars currently in simulation: {len(traffic_status['vehicles'])}",
                    ]
                ),

                result.get("response", ""),

                render_intersection(traffic_status),

                render_cards(
                    traffic_status["vehicle_counts"],
                    traffic_status["waiting_times"],
                ),

                render_metrics(metrics),

                build_execution_log(result, controller),

                controller,
            )

            time.sleep(ANIMATION_DELAY)

        except Exception as error:

            current_data = controller.get_dashboard_data()

            current_status = current_data["traffic_status"]

            yield (
                status_html(
                    "bad",
                    "SIMULATION ERROR",
                    [str(error)]
                ),

                result.get("response", ""),

                render_intersection(current_status),

                render_cards(
                    current_status["vehicle_counts"],
                    current_status["waiting_times"],
                ),

                render_metrics(
                    current_data["performance_metrics"]
                ),

                build_execution_log(result, controller),

                controller,
            )

            return

    # STEP 10: FINAL SIMULATION RESULTS.

    final_data = controller.get_dashboard_data()

    final_status = final_data["traffic_status"]

    final_metrics = final_data["performance_metrics"]

    # Update historical data and previous phase
    # only after a successful completed cycle.
    if final_data["cycle_status"] == "COMPLETED":

        LAST_VALID_SIMULATION_DATA = {
            "vehicle_counts": final_status["vehicle_counts"].copy(),
            "waiting_times": final_status["waiting_times"].copy(),
            "label": (
                f"Completed simulation at "
                f"{final_status['simulation_time']} seconds"
            ),
        }

        # Save the phase successfully served.
        LAST_SERVED_PHASE = selected_phase

    yield (
        status_html(
            "ok",
            "SIMULATION COMPLETED",
            [
                f"Selected phase: {selected_phase}",
                f"Green duration: {green_duration} seconds",
                f"Total departures: {final_metrics['total_departures']}",
                f"Remaining cars: {final_metrics['total_vehicles_waiting']}",
                f"Individual car records: {final_data['total_individual_cars']}",
                "Simulation results retrieved from actual backend state."
            ]
        ),

        result.get("response", ""),

        render_intersection(final_status),

        render_cards(
            final_status["vehicle_counts"],
            final_status["waiting_times"],
        ),

        render_metrics(final_metrics),

        build_execution_log(result, controller),

        controller,
    )


# =====================================================
# 17. RESET DASHBOARD
# =====================================================

def reset_dashboard():
    global LAST_SERVED_PHASE

    # Reset the phase history for a fresh experiment.
    LAST_SERVED_PHASE = None

    return (
        "",

        IDLE_STATUS,

        "Enter traffic information to start a new analysis.",

        render_intersection(),

        render_historical_cards(LAST_VALID_SIMULATION_DATA),

        render_metrics(),

        {},

        None,
    )


# =====================================================
# 18. PRESET REQUESTS
# =====================================================

def create_preset_request(name):

    if name not in PRESETS:
        raise ValueError("Unknown traffic preset.")

    (
        north_v,
        north_w,
        south_v,
        south_w,
        east_v,
        east_w,
        west_v,
        west_w,
    ) = PRESETS[name]

    return f"""
Analyse this four-way traffic intersection.

North has {north_v} vehicles waiting {north_w} seconds.
South has {south_v} vehicles waiting {south_w} seconds.
East has {east_v} vehicles waiting {east_w} seconds.
West has {west_v} vehicles waiting {west_w} seconds.

Determine the highest-priority traffic phase
and calculate its recommended green-light duration.
"""


# =====================================================
# 19. BUILD GRADIO INTERFACE
# =====================================================
#
# Page layout (two balanced columns):
#
#   Row 1:  [1 Traffic Request + 2 Traffic Monitoring] | [3 AI Decision]
#   Row 2:  [4 Live Intersection]  | [5 Performance + 6 Execution Logs]

with gr.Blocks(
    title="SmartTraffic - AI Traffic Assistant"
) as demo:

    # Controller state.
    controller_state = gr.State(value=None)

    # HEADER.
    gr.HTML("""
    <div class="st-header">
        <div class="st-header-left">
            <div class="st-header-icon">🚦</div>
            <div>
                <h1>SmartTraffic AI Assistant</h1>
                <p>Intelligent Traffic Light Decision Agent</p>
            </div>
        </div>
        <div class="st-chips">
            <span class="st-chip">QWEN AI</span>
            <span class="st-chip">GROQ INFERENCE</span>
            <span class="st-chip">PYTHON FUNCTION CALLING</span>
        </div>
    </div>
    """)

    # ROW 1
    with gr.Row(equal_height=False, elem_classes=["st-row"]):

        # ---- LEFT: request + monitoring ----
        with gr.Column(scale=1, min_width=420, elem_classes=["st-card"]):

            gr.HTML(section_title(1, "Traffic Request"))

            user_input = gr.Textbox(
                label="Enter traffic conditions",
                placeholder=(
                    "Example:\n"
                    "North has 25 vehicles waiting 90 seconds.\n"
                    "South has 18 vehicles waiting 65 seconds.\n"
                    "East has 8 vehicles waiting 25 seconds.\n"
                    "West has 5 vehicles waiting 15 seconds.\n\n"
                    "Ask SmartTraffic to analyse the intersection."
                ),
                lines=8,
                max_lines=14,
                elem_classes=["st-input"],
            )

            gr.HTML(
                '<div class="st-subtitle">Example traffic scenarios</div>'
                '<div class="st-hint">Type your own traffic conditions above, '
                'or click a scenario to fill them in. '
                'Then press <b>Analyse Traffic</b>.</div>'
            )

            preset_buttons = []

            with gr.Row(elem_classes=["st-presets"]):
                for name in PRESETS:
                    button = gr.Button(name, size="sm")
                    preset_buttons.append((name, button))

            with gr.Row(elem_classes=["st-actions"]):
                analyse_button = gr.Button(
                    "Analyse Traffic",
                    variant="primary",
                    size="lg",
                    scale=2,
                )
                reset_button = gr.Button(
                    "Reset",
                    variant="secondary",
                    size="lg",
                    scale=1,
                )

            gr.HTML(section_title(2, "Live Traffic Monitoring"))

            traffic_cards = gr.HTML(
                render_historical_cards(LAST_VALID_SIMULATION_DATA)
            )

        # ---- RIGHT: AI decision ----
        with gr.Column(scale=1, min_width=420, elem_classes=["st-card"]):

            gr.HTML(section_title(3, "AI Decision"))

            agent_status = gr.HTML(IDLE_STATUS)

            ai_response = gr.Markdown(
                "Waiting for traffic analysis...",
                elem_id="ai-response",
            )
            
            gpu_button = gr.Button(
                "Check GPU Status",
                variant="secondary",
            )
            
            gpu_status = gr.Textbox(
                label="ZeroGPU Status",
                interactive=False,
                placeholder="GPU status has not been checked.",
            )
    # ROW 2
    with gr.Row(equal_height=False, elem_classes=["st-row"]):

        # ---- LEFT: intersection ----
        with gr.Column(scale=1, min_width=420, elem_classes=["st-card"]):

            gr.HTML(section_title(4, "Live Traffic Intersection"))

            intersection_view = gr.HTML(render_intersection())

        # ---- RIGHT: performance + logs ----
        with gr.Column(scale=1, min_width=420, elem_classes=["st-card"]):

            gr.HTML(section_title(5, "Simulation Performance"))

            performance_view = gr.HTML(render_metrics())

            gr.HTML(section_title(6, "Python Function Execution Logs"))

            execution_logs = gr.JSON(
                label="Actual Function Calls, Arguments and Results",
                value={},
                elem_classes=["st-logs"],
            )

    # FOOTER.
    gr.HTML("""
    <div class="st-footer">
        SmartTraffic is an academic simulation system.
        It does not control real-world traffic infrastructure.
    </div>
    """)

    # =================================================
    # 20. EVENT CONNECTIONS
    # =================================================

    outputs = [
        agent_status,
        ai_response,
        intersection_view,
        traffic_cards,
        performance_view,
        execution_logs,
        controller_state,
    ]

    # ZEROGPU DIAGNOSTIC.
    gpu_button.click(
        fn=check_gpu_status,
        outputs=gpu_status, 
    )

    # PRESET BUTTONS.
    for name, button in preset_buttons:
        button.click(
            fn=lambda n=name: create_preset_request(n),
            outputs=user_input,
        )

    # ONE-CLICK ANALYSIS.
    analyse_button.click(
        fn=analyse_traffic,
        inputs=user_input,
        outputs=outputs,
        show_progress="full",
    )

    # ENTER KEY SUBMISSION.
    user_input.submit(
        fn=analyse_traffic,
        inputs=user_input,
        outputs=outputs,
        show_progress="full",
    )

    # RESET.
    reset_button.click(
        fn=reset_dashboard,
        outputs=[
            user_input,
            agent_status,
            ai_response,
            intersection_view,
            traffic_cards,
            performance_view,
            execution_logs,
            controller_state,
        ],
    )


# =====================================================
# 21. LAUNCH APPLICATION
# =====================================================

def launch_app():

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":

    launch_app()
