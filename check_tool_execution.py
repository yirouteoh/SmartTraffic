
import json

from agent.tools import (
    AVAILABLE_TOOLS,
    execute_tool,
)


# Example traffic information.
arguments = {
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
}


print("AVAILABLE TOOLS:")

for tool in AVAILABLE_TOOLS:
    print(tool["function"]["name"])


print("\nEXECUTING TRAFFIC PRIORITY TOOL...")

result = execute_tool(
    "traffic_priority",
    arguments,
)

print(json.dumps(result, indent=4))


print("\nEXECUTING GREEN TIME TOOL...")

green_result = execute_tool(
    "green_time",
    {
        "selected_phase": result["selected_phase"],
        **arguments,
    },
)

print(json.dumps(green_result, indent=4))


print("\nTESTING UNAUTHORISED FUNCTION...")

try:

    execute_tool(
        "delete_traffic_system",
        {},
    )

except ValueError as error:

    print("GUARDRAIL WORKING:", error)
