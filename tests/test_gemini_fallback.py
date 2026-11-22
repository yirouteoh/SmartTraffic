
import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_agent import SmartTrafficAgent

load_dotenv()


# Simulate a Groq rate-limit error.
class SimulatedRateLimitError(Exception):

    status_code = 429


def test_gemini_fallback():

    print("\n===================================")
    print("SMARTTRAFFIC GEMINI FALLBACK TEST")
    print("===================================")

    agent = SmartTrafficAgent()

    if not agent.gemini_available:
        raise RuntimeError("Gemini provider is unavailable.")

    # Simulate Groq failing before its first function call.
    with patch.object(
        agent.client.chat.completions,
        "create",
        side_effect=SimulatedRateLimitError(
            "Simulated Groq rate limit (429)"
        ),
    ):

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

    print("\nAI EXPLANATION:")
    print(result["response"])

    # Verify the fallback completed the workflow.
    assert result["success"] is True
    assert result["provider"] == "Gemini"
    assert result["python_call_count"] == 2

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
    print("GEMINI FALLBACK TEST PASSED!")
    print("===================================")


if __name__ == "__main__":
    test_gemini_fallback()
