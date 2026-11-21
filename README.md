
---
title: SmartTraffic AI Assistant
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: main.py
pinned: false
---

# SmartTraffic AI Assistant

An intelligent traffic decision agent powered by Qwen through the Groq API.

## Features

- AI-controlled traffic priority decisions
- Python function calling
- Traffic safety guardrails
- Traffic simulation
- Individual vehicle tracking
- Interactive Gradio dashboard
- Execution logs showing actual Python function calls, arguments and results

## Technology Stack

- Python 3.11
- Gradio 6.32.0
- Groq API
- Qwen LLM

## Configuration

Set the following secret in your Hugging Face Space:

`GROQ_API_KEY`

The application uses Groq API for Qwen inference.

## Disclaimer

SmartTraffic is an academic traffic simulation system. It does not control real-world traffic infrastructure.
