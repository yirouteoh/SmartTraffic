\---

title: SmartTraffic AI Assistant

emoji: 🚦

colorFrom: blue

colorTo: indigo

sdk: gradio

sdk_version: 6.32.0

python_version: "3.12"

app_file: main.py

pinned: false

\---



**#  SmartTraffic AI Assistant**



**\*\*An intelligent traffic decision and simulation system powered by Qwen, Groq, Python function calling, and Gradio.\*\***



SmartTraffic is an academic AI-agent project that analyses traffic conditions at a simulated four-way intersection. It evaluates vehicle demand and waiting-time pressure, selects a higher-priority traffic phase, recommends a green-light duration, and visualises the resulting traffic simulation.



\> **\*\*Important:\*\*** SmartTraffic is an academic simulation. It does not connect to or control real-world traffic lights or road infrastructure.



**##  Live Demo**



Launch the interactive application on **\*\*[Hugging Face Spaces]\(**&#x68;ttps\://huggingface.co/spaces/Rayzed0224/smarttraffic-a&#x69;**)\*\***.



**##  Dashboard Preview**



The Gradio dashboard brings traffic input, AI decisions, live monitoring, simulation performance, and Python execution logs together in one interface.



![SmartTraffic main dashboard]\(images/dashboard-overview\.png)



*\*Main dashboard: enter traffic conditions, request an analysis, and review the AI decision.\**



![SmartTraffic dashboard with traffic monitoring]\(images/dashboard-monitoring.png)



*\*Traffic monitoring and analysis results displayed alongside the request interface.\**



**### Live Four-Way Intersection**



![Live four-way traffic intersection]\(images/live-intersection.png)



The simulator visualises the North, South, East, and West approaches, vehicle movement, and traffic-signal states.



**### Simulation Metrics and Execution Logs**



![Simulation performance and Python execution logs]\(images/simulation-metrics-logs.png)



The dashboard reports simulation performance and exposes the Python function calls, arguments, and results used during the decision workflow.



**##  Key Features**



\- **\*\*AI-assisted traffic prioritisation\*\*** – evaluates North–South (NS) and East–West (EW) traffic phases using vehicle demand, waiting-time pressure, and fairness.

\- **\*\*Python function calling\*\*** – the LLM selects authorised Python tools; deterministic Python functions perform the calculations.

\- **\*\*Adaptive green-time recommendation\*\*** – recommends a green duration constrained to 15–60 seconds.

\- **\*\*Traffic simulation\*\*** – visualises a four-way intersection and tracks individual vehicles, queues, waiting times, arrivals, departures, and signal transitions.

\- **\*\*Safety guardrails\*\*** – validates traffic inputs and blocks incomplete, unsafe, unsupported, and detected prompt-injection requests.

\- **\*\*Multi-provider resilience\*\*** – uses Qwen through the Groq API as the primary provider and Gemini 3.1 Flash Lite as a fallback for recoverable Groq failures.

\- **\*\*Transparent execution logs\*\*** – displays the actual Python function names, arguments, and results.

\- **\*\*Interactive Gradio dashboard\*\*** – combines traffic requests, AI explanations, monitoring, simulation metrics, and execution logs.



**##  How It Works**



The system separates language-model orchestration from traffic calculations and safety enforcement. The LLM coordinates the authorised tools, while Python validates inputs, performs calculations, and controls the simulation workflow.



\`\`\`mermaid

flowchart TD

    A[User enters traffic conditions] --> B[Python input validation and safety checks]

    B -->|Valid request| C[Qwen through Groq API]

    B -->|Incomplete or unsafe| X[Reject request / request missing information]

    C --> D[traffic_priority Python function]

    D --> E[green_time Python function]

    C -. Recoverable Groq failure .-> G[Gemini 3.1 Flash Lite fallback]

    G --> D

    G --> E

    E --> F[Traffic simulation]

    F --> H[Gradio dashboard, explanation and execution logs]

\`\`\`



**### Traffic Priority Calculation**



The decision engine evaluates each traffic phase using a weighted priority score:



\\[

P = 0.45Q + 0.35W + 0.20F

\\]



Where:



\- **\*\*P\*\*** – overall traffic priority score.

\- **\*\*Q\*\*** – normalised vehicle demand.

\- **\*\*W\*\*** – normalised waiting-time pressure.

\- **\*\*F\*\*** – fairness component.



The system compares the NS and EW scores and selects the phase with the higher priority. The weights are 45% for vehicle demand, 35% for waiting-time pressure, and 20% for fairness.



**### Green-Time Recommendation**



The green-time function calculates a recommended duration using queue pressure and waiting-time pressure:



\\[

C = 0.60Q_p + 0.40W_p

\\]



Where **\*\*Qₚ\*\*** is queue pressure and **\*\*Wₚ\*\*** is waiting-time pressure. The recommended green duration is constrained to **\*\*15–60 seconds\*\***.



**### Sequential Phase Alternation**



When both phases have equal priority, the agent uses the previous successfully served phase to alternate fairly:



\- No previous phase: select NS.

\- Previous phase NS: select EW.

\- Previous phase EW: select NS.



The stored phase is updated only after a successful simulation.



**##  Safety and Validation**



Safety checks are enforced in Python rather than relying only on the language model.



\- **\*\*Required traffic data:\*\*** all four directions (North, South, East, and West) must include vehicle counts and waiting times.

\- **\*\*Vehicle-count validation:\*\*** values must be integers from 0 to 10,000.

\- **\*\*Waiting-time validation:\*\*** values must be from 0 to 86,400 seconds.

\- **\*\*Function allowlist:\*\*** only \`traffic_priority\` and \`green_time\` are authorised for the agent workflow.

\- **\*\*Incomplete requests:\*\*** missing values are not silently replaced with zero; processing is stopped before LLM execution, Python function calls, or simulation.

\- **\*\*Prompt-injection detection:\*\*** predefined suspicious instruction patterns are blocked before the request reaches the LLM.

\- **\*\*Unsafe signal actions:\*\*** conflicting NS and EW green phases are rejected.

\- **\*\*Signal transition validation:\*\*** the simulator enforces \`GREEN → YELLOW → ALL_RED → GREEN\` and prevents vehicle green activation during pedestrian crossings.

\- **\*\*Real-world control restriction:\*\*** requests to operate actual traffic infrastructure are not supported.



**##  AI Provider Configuration**



\| Role | Provider / Model | Purpose |

\|---|---|---|

\| Primary | Qwen via Groq API | Selects the authorised Python functions and generates traffic-analysis explanations. |

\| Fallback | Gemini 3.1 Flash Lite | Resumes the workflow when a recoverable Groq failure occurs. |



The fallback is restricted to the authorised function required at the current workflow stage. Previously completed Python results are preserved, avoiding unnecessary repeated calculations.



**##  Testing and Evaluation**



The project was evaluated using eight challenge scenarios and additional provider-fallback and regression tests.



\| Evaluation | Result |

\|---|---:|

\| Challenge scenarios | **\*\*8/8 passed (100%)\*\*** |

\| Groq-to-Gemini fallback scenarios | **\*\*3/3 passed\*\*** |

\| Automated regression test suite | **\*\*45 tests passed\*\*** |



The eight challenge scenarios covered:



1\. North–South heavy traffic.

2\. East–West heavy traffic.

3\. Balanced traffic.

4\. Quiet traffic and equal priority scores.

5\. High waiting-time pressure.

6\. Missing traffic information.

7\. Prompt-injection attempt.

8\. Unsafe simultaneous signal activation.



The three fallback tests simulated recoverable Groq failures at the initial function stage, during the workflow, and during final explanation generation. The report records successful Gemini fallback in all three cases.



**##  Technology Stack**



\- **\*\*Language:\*\*** Python

\- **\*\*Interface:\*\*** Gradio 6.32.0

\- **\*\*Primary LLM access:\*\*** Groq API with Qwen

\- **\*\*Fallback LLM:\*\*** Gemini 3.1 Flash Lite

\- **\*\*Agent capabilities:\*\*** LLM tool / function calling

\- **\*\*Simulation:\*\*** Python-based four-way traffic simulation

\- **\*\*Deployment:\*\*** Hugging Face Spaces



**##  Setup and Configuration**



**### 1. Clone the repository**



\`\`\`bash

git clone YOUR_GITHUB_REPOSITORY_URL

cd YOUR_REPOSITORY_FOLDER

\`\`\`



**### 2. Create and activate a virtual environment**



\`\`\`bash

python -m venv .venv

\`\`\`



Windows PowerShell:



\`\`\`powershell

.venv\Scripts\Activate.ps1

\`\`\`



macOS / Linux:



\`\`\`bash

source .venv/bin/activate

\`\`\`



**### 3. Install dependencies**



If the repository contains a \`requirements.txt\` file, install the project dependencies with:



\`\`\`bash

pip install -r requirements.txt

\`\`\`



**### 4. Configure API keys**



Set the required API secret in your Hugging Face Space:



\`\`\`text

GROQ_API_KEY

\`\`\`



If your deployed configuration uses Gemini fallback, configure the Gemini API key using the environment-variable name expected by your application code. Do not commit API keys or other secrets to GitHub.



**### 5. Run the application locally**



\`\`\`bash

python main.py

\`\`\`



Open the local Gradio URL printed in the terminal.



**##  Repository Structure**



The application is organised into separate components for the agent, Python traffic logic, validation, simulation, and user interface. Refer to the repository itself for the current file and folder names.



**##  Limitations**



\- The application analyses user-provided traffic information; it does not obtain live traffic data from physical road sensors.

\- Traffic behaviour is simulated and does not represent a certified traffic-control system.

\- Priority weights and simulation parameters are predefined.

\- AI-provider availability depends on the external Groq and Gemini services and their API limits.

\- The system is intended for academic demonstration, experimentation, and learning only.



**##  Project Context**



Developed for **\*\*WQF7002 – Artificial Intelligence Techniques\*\*** as an academic project on intelligent-agent design, LLM integration, Python function calling, safety guardrails, and traffic simulation.



**##  Disclaimer**



SmartTraffic AI Assistant is an academic traffic simulation system. It is not certified for operational traffic management and must not be used to control real-world traffic infrastructure.
