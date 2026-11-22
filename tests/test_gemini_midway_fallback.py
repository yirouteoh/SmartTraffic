
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


def test_midway_fallback():

    print("\n===================================")
    print("MID-WORKFLOW GEMINI FALLBACK TEST")
    print("===================================")

    agent = SmartTrafficAgent()

    if not agent.gemini_available:
        raise RuntimeError("Gemini is unavailable.")

    # Mock a successful first Groq function call.
    first_tool_call = SimpleNamespace(
        id="mock-priority-call",
        function=SimpleNamespace(
            name="traffic_priority",
            arguments=json.dumps({
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
            }),
        ),
    )

    first_message = SimpleNamespace(
        tool_calls=[first_tool_call],
        model_dump=lambda **kwargs: {
            "role": "assistant",
            "tool_calls": [],
        },
    )

    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=first_message)
        ]
    )

    # First Groq call succeeds.
    # Second Groq call simulates a 429 error.
    with patch.object(
        agent.client.chat.completions,
        "create",
        side_effect=[
            first_response,
            SimulatedRateLimitError(
                "Simulated Groq rate limit during green_time."
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

    # Verify correct fallback behaviour.
    assert result["success"] is True
    assert result["provider"] == "Gemini"
    assert result["python_call_count"] == 2

    assert mock_groq.call_count == 2

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

    print("\n===================================")
    print("MID-WORKFLOW FALLBACK PASSED!")
    print("===================================")


if __name__ == "__main__":
    test_midway_fallback()
