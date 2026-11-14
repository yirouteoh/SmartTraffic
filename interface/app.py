
"""
SmartTraffic - Gradio Application

Features:
1. AI traffic assistant
2. Traffic analysis results
3. Python function execution logs
4. Agent status monitoring
"""

import gradio as gr

from agent.llm_agent import SmartTrafficAgent


# ==========================================
# 1. INITIALISE AGENT
# ==========================================

agent = SmartTrafficAgent()


# ==========================================
# 2. PROCESS USER REQUEST
# ==========================================

def chat_with_agent(message, history):

    if not message or not message.strip():
        return (
            history,
            "Please enter a traffic-related request.",
            []
        )

    try:

        result = agent.process_request(message)

        response = result.get(
            "response",
            "No response was generated."
        )

        logs = result.get("tool_calls", [])

        if not result.get("success", False):
            status = "INCOMPLETE - Agent could not complete request."
        else:
            status = "SUCCESS - Request completed."

        updated_history = list(history)

        updated_history.append({
            "role": "user",
            "content": message,
        })

        updated_history.append({
            "role": "assistant",
            "content": response,
        })

        return (
            updated_history,
            f"Model: {result.get('model', 'Unknown')}\n"
            f"Agent status: {status}",
            logs,
        )

    except Exception as error:

        updated_history = list(history)

        updated_history.append({
            "role": "user",
            "content": message,
        })

        updated_history.append({
            "role": "assistant",
            "content": (
                "SmartTraffic could not complete your request. "
                "Please try again later."
            ),
        })

        return (
            updated_history,
            f"Agent status: ERROR - {type(error).__name__}: {error}",
            [],
        )


# ==========================================
# 3. CLEAR CHAT
# ==========================================

def clear_chat():

    return (
        [],
        "Waiting for user request.",
        [],
        "",
    )


# ==========================================
# 4. BUILD GRADIO INTERFACE
# ==========================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}

#main-title {
    text-align: center;
}
"""


with gr.Blocks(
    title="SmartTraffic - Intelligent Traffic Agent",
) as demo:

    gr.Markdown(
        """
        # 🚦 SmartTraffic

        ### Intelligent Traffic Light Control Agent

        An AI-powered traffic management assistant that
        analyses traffic conditions, selects traffic phases
        and calculates recommended green-light durations.

        **AI Model:** Qwen3.8-27B

        **Inference Provider:** Groq
        """,
        elem_id="main-title",
    )

    with gr.Row():

        with gr.Column(scale=3):

            gr.Markdown("## 🤖 AI Traffic Assistant")

            chatbot = gr.Chatbot(
                label="SmartTraffic Conversation",
                height=450,
            )

            user_input = gr.Textbox(
                label="Enter your traffic request",
                placeholder=(
                    "Example: Analyse the traffic conditions "
                    "at a four-way intersection..."
                ),
                lines=3,
            )

            with gr.Row():

                submit_button = gr.Button(
                    "Analyse Traffic",
                    variant="primary",
                )

                clear_button = gr.Button(
                    "Clear Conversation",
                )

        with gr.Column(scale=2):

            gr.Markdown("## 📊 Agent Status")

            agent_status = gr.Textbox(
                label="Execution Status",
                value="Waiting for user request.",
                interactive=False,
                lines=3,
            )

            gr.Markdown("## ⚙️ Python Function Execution")

            execution_logs = gr.JSON(
                label="Function Calls, Arguments and Results",
                value=[],
            )

    gr.Markdown("---")

    gr.Markdown(
        """
        ### Example Traffic Request

        Analyse this four-way intersection:

        - North: 25 vehicles, waiting 90 seconds.
        - South: 18 vehicles, waiting 65 seconds.
        - East: 8 vehicles, waiting 25 seconds.
        - West: 5 vehicles, waiting 15 seconds.

        Determine the highest-priority phase and calculate
        its recommended green-light duration.
        """
    )

    # ======================================
    # 5. SUBMIT BUTTON
    # ======================================

    submit_button.click(
        fn=chat_with_agent,
        inputs=[
            user_input,
            chatbot,
        ],
        outputs=[
            chatbot,
            agent_status,
            execution_logs,
        ],
    ).then(
        fn=lambda: "",
        outputs=user_input,
    )

    # ======================================
    # 6. ENTER KEY SUBMISSION
    # ======================================

    user_input.submit(
        fn=chat_with_agent,
        inputs=[
            user_input,
            chatbot,
        ],
        outputs=[
            chatbot,
            agent_status,
            execution_logs,
        ],
    ).then(
        fn=lambda: "",
        outputs=user_input,
    )

    # ======================================
    # 7. CLEAR BUTTON
    # ======================================

    clear_button.click(
        fn=clear_chat,
        outputs=[
            chatbot,
            agent_status,
            execution_logs,
            user_input,
        ],
    )


# ==========================================
# 8. LAUNCH APPLICATION
# ==========================================

def launch_app():

    demo.launch(
        server_name="127.0.0.1",
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
