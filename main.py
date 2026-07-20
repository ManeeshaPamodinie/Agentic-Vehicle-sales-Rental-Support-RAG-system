import json
import os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

def laod_agent_from_notebook(notebook_path="answer.ipynb"):
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"could  not find {notebook_path}. Make sure it is in the same folder")

    with open(notebook_path, "r", encoding="utf-8")as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            clean_lines = [line for line in cell.get("source", []) if not line.strip().startswith(("%","!"))]
            code_cells.append("".join(clean_lines))

    exec("\n".join(code_cells), globals())

print("Extracting backend logic from answer.ipynb...")
laod_agent_from_notebook("answer.ipynb")



def chat_handler(message: str, history):
    formatted_history = []
    for msg in history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))

    try:
        response = agent_executor.invoke({
            "input": message,
            "chat_history": formatted_history
        })
        return response["output"]
    except Exception as e:
        return f"Pipeline Error: {str(e)}"
    
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat_handler,
        title="Velocity Dealership Concierge Agent",
        description="Ask complex multi-domain queries regarding vehicle inventory, financing and rental procedures.",    
    )
    demo.launch()