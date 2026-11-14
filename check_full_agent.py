
"""
SmartTraffic - Full LLM Agent Integration Test
"""

import json

from agent.llm_agent import SmartTrafficAgent


# Initialise agent.
agent = SmartTrafficAgent()


# Example user request.
user_request = """
Analyse this four-way intersection.

North has 25 vehicles waiting 90 seconds.
South has 18 vehicles waiting 65 seconds.
East has 8 vehicles waiting 25 seconds.
West has 5 vehicles waiting 15 seconds.

Determine the highest-priority phase and
calculate its recommended green-light duration.
"""


print("=" * 55)
print("SMARTTRAFFIC - FULL AGENT TEST")
print("=" * 55)

print("\nUSER REQUEST:")
print(user_request)


# Process request.
result = agent.process_request(user_request)


print("\nFINAL AGENT RESPONSE:")
print(result["response"])


print("\nFUNCTION EXECUTION LOGS:")

for log in result["tool_calls"]:

    print("\nFunction:", log["function_name"])

    print("Arguments:")
    print(json.dumps(log["arguments"], indent=4))

    print("Status:", log["status"])

    print("Result:")
    print(json.dumps(log["result"], indent=4))


print("\nMODEL:", result["model"])

print("\nSUCCESS:", result["success"])
