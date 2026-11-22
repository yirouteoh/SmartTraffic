
import sys
import json

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_agent import SmartTrafficAgent

load_dotenv()


class SimulatedRateLimitError(Exception):
    status_code = 429


def make_groq_response(function_name, arguments, call_id):

    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=function_name,
            arguments=json.dumps(arguments),
        ),
    )

    message = SimpleNamespace(
        tool_calls=[tool_call],
        model_dump=lambda **kwargs: {
            "role": "assistant",
            "tool_calls": [],
        },
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def test_explanation_fallback():

    print("\n===================================")
    print("GEMINI EXPLANATION FALLBACK TEST")
    print("===================================")

    agent = SmartTrafficAgent()

    if not agent.gemini_available:
        raise RuntimeError("Gemini provider is unavailable.")

    # Mock successful Groq traffic_priority selection.
    priority_response = make_groq_response(
        "traffic_priority",
        {
            "vehicle_counts": {
                "north": 25,
                "south": 18,
                "east": 8,
                "west": 5,
            },
            "waiting_times": {
                "north": 90,
                "south": 65,
                "east": 25,
                "west": 15,
            },
            "previous_phase": None,
        },
        "priority-call",
    )

    # Mock successful Groq green_time selection.
    green_response = make_groq_response(
        "green_time",
        {
            "phase": "NS",
            "vehicle_counts": {
                "north": 25,
                "south": 18,
                "east": 8,
                "west": 5,
            },
            "waiting_times": {
                "north": 90,
                "south": 65,
                "east": 25,
                "west": 15,
            },
        },
        "green-call",
    )

    # Both tool selections succeed.
    # Groq fails only during final explanation.
    with patch.object(
        agent.client.chat.completions,
        "create",
        side_effect=[
            priority_response,
            green_response,
            SimulatedRateLimitError(
                "Simulated Groq rate limit during final explanation."
            ),
        ],
    ) as mock_groq:

        result = agent.process_request(
            user_input=(
                "Analyse this four-way traffic intersection. "
                "North has 25 vehicles waiting 90 seconds. "
                "South has 18 vehicles waiting 65 seconds. "
                "East has 8 vehicles waiting 25 seconds. "
                "West has 5 vehicles waiting 15 seconds."
            ),
            previous_phase=None,
        )

    print("\nRESULT:")
    print("Success:", result["success"])
    print("Provider:", result.get("provider"))
    print("Model:", result.get("model_used"))
    print("Request status:", result["request_status"])
    print("Python calls:", result["python_call_count"])

    print("\nFUNCTION EXECUTION:")

    for call in result["tool_calls"]:
        print(
            call["function_name"],
            "->",
            call["status"],
        )

    print("\nGROQ REQUEST COUNT:", mock_groq.call_count)

    print("\nAI EXPLANATION:")
    print(result["response"])

    # Verify both Python functions executed exactly once.
    assert result["success"] is True
    assert result["provider"] == "Gemini"
    assert result["python_call_count"] == 2
    assert mock_groq.call_count == 3

    assert [
        call["function_name"]
        for call in result["tool_calls"]
    ] == [
        "traffic_priority",
        "green_time",
    ]

    assert all(
        call["status"] == "SUCCESS"
        for call in result["tool_calls"]
    )

    assert "simulation recommendation" in (
        result["response"].lower()
    )

    print("\n===================================")
    print("EXPLANATION FALLBACK TEST PASSED!")
    print("===================================")


if __name__ == "__main__":
    test_explanation_fallback()
