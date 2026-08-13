import json
import os
import logging
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

# =====================================================
# Logging
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# Load Notebook Backend
# =====================================================

def load_agent_from_notebook(notebook_path="answer.ipynb"):

    if not os.path.exists(notebook_path):
        raise FileNotFoundError(
            f"Could not find {notebook_path}"
        )

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_cells = []

    for cell in notebook.get("cells", []):

        if cell.get("cell_type") == "code":

            clean_lines = [

                line

                for line in cell.get("source", [])

                if not line.strip().startswith(("%", "!"))

            ]

            code_cells.append("".join(clean_lines))

    exec("\n".join(code_cells), globals())


print("Loading AI backend...")

load_agent_from_notebook("answer.ipynb")

print("Backend Loaded Successfully.")

# =====================================================
# Chat Handler
# =====================================================

def chat_handler(message: str, history):

    formatted_history = []

    for msg in history:

        role = msg.get("role")

        content = msg.get("content", "")

        if role == "user":

            formatted_history.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":

            formatted_history.append(
                AIMessage(content=content)
            )

    try:

        response = agent_executor.invoke(

            {

                "input": message,

                "chat_history": formatted_history

            }

        )

        answer = response.get(
            "output",
            "I couldn't generate a response."
        )

        logger.info(f"User : {message}")
        logger.info(f"Assistant : {answer}")

        return answer

    except Exception as e:

        logger.exception(e)

        return (
            "Sorry, something went wrong while processing your request. "
            "Please try again."
        )


# =====================================================
# Launch
# =====================================================

if __name__ == "__main__":

    demo = gr.ChatInterface(

        fn=chat_handler,
        title="Velocity Dealership Concierge Agent",
        description=(
            "Ask questions about vehicles, financing, rentals, "
            "warranty, maintenance and dealership information."
        )
    )

    demo.launch()