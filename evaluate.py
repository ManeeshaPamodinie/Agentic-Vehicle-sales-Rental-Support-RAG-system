import json
import time
import os
from langchain_groq import ChatGroq

def laod_agent_from_notebook(notebook_path="answer.ipynb"):
    """Reads the JSON structure of the notebook and executes the code cells in memory."""
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


evaluation_dataset = [
    {
        "id": 1,
        "question": "I am looking at the Apex Aura Sedan. Can you tell me its highway fuel economy and what the APR interest rate would be for someone with a Tier 1 credit score?",
        "expected_facts": ["37 MPG Highway", "1.9% - 3.5%", "Apex Aura"]
    },
    {
        "id": 2,
        "question": "How many passengers can the TerraQuest X7 SUV hold and how long am I covered under its powertrain warranty?",
        "expected_facts": ["7 to 8 passengers", "5 years or 60,000 miles", "TerraQuest X7 SUV"]
    },
    {
        "id": 3,
        "question": "What is the rear cargo capacity of the EcoPulse H2 Hybrid, and how much can i save on it using the Colledge Graduate Program?",
        "expected_facts": ["22.5 cubic feet", "$400 cash rebate", "EcoPulse H2 Hybrid"]
    },
    {
        "id": 4,
        "question": "what is the targeted total driving range of the VoltZen Pure EV on a full charge and what point of sale tax incentives appl to it?",
        "expected_facts": ["310 miles", "up to $7,500 point of sale rebate", "VoltZen Pure EV"]
    },
    {
        "id": 5,
        "question": "If my loan payment is late, what penalty fee will I be charged and what iS the minimum monthly income required to finance a vehicle worth $45,000?",
        "expected_facts": ["5% of the past due payment amount or $35", "$4,000"]
    },
    {
        "id": 6,
        "question": "I am graduating college next month and looking to buy the VoltZen Pure EV. Can I combine the college graduate program rebate with EV Federal tax credit and what specific IRS form must be completed at the time of sale?",
        "expected_facts": ["can be combined or stacked", "up to $7,500 point-of-sale rebate", "IRS Form 15400"]
    },
    {
        "id": 7,
        "question": "what type of drivetrain does the VoltZen Pure EV have and according to your maintenance shedule, how often should I bring it in for tire rotations based on that exact drivetrain?",
        "expected_facts": ["Dual-Motor All-Wheel Drive (AWD)", "rotate every 5,000 miles", "AWD vehicles"]
    },
    {
        "id": 8,
        "question": "I want to trade in my old Kawasaki motorcycle to buy an Apex Aura Sedan. Does your trade-in policy allow this, and what is the standard bumper-to-bumper basic warranty coverage duration on that sedan?",
        "expected_facts": ["do not accept the following vehicles for trade-in: motorcycles", "3 years or 36,000 miles", "Apex Aura Sedan"]
    }
]

eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

SYSTEM_PROMPT = """You are an elite, helpful virtual customer assistant for 'Velocity Dealerships'.
your primary job is to provide accurate answers about vehicle models, financing policies, buying rules and rental agreements.
your goal is to guide the user seamlessly through pre qulification, sales, rentals, warranty claims and showroom info.
Given a user's question, a list of target key facts that must be presnt and the AI's actual generated resonse, verify completeness.

Target Facts:
{expected_facts}

AI Response:
{ai_response}

Output your analysis in JSON formt exactly like this:
{{
    "facts_checked": [
       {{"fact": "fact 1 name", "captured": true/false}},
       {{"fact": "fact 2 name", "captured": true/false}},
    ],
    "completeness_score": 0.0 to 1.0
}}
"""

def run_evaluation():
    results = []
    total_score = 0.0

    print("Running Evaluation pipeline...")
    for item in evaluation_dataset:
        print(f"\n[Evaluating Question ID {item['id']}]")

        start_time = time.time()
        output = agent_executor.invoke({"input": item["question"], "chat_history": []})
        latency = round(time.time() - start_time, 2)

        ai_response = output["output"]

        formatted_prompt = SYSTEM_PROMPT.format(
            expected_facts=json.dumps(item["expected_facts"]),
            ai_response=ai_response
        )
        judge_response = eval_llm.invoke(formatted_prompt)

        try:
            metrices = json.loads(judge_response.content)
            score = metrices.get("completeness_score", 0.0)
        except Exception:
            score = 1.0 if all(f.lower() in ai_response.lower() for f in item["expected_facts"]) else 0.5
            metrices = {"fallback_applied": True}

        total_score += score
        results.append({
            "id": item["id"],
            "question": item["question"],
            "ai_response": ai_response,
            "latency_seconds": latency,
            "completeness_score": score, 
            "details": metrices
        })

        print(f"Score: {score} | Time: {latency}s")

    avg_score = round(total_score / len (evaluation_dataset), 2)
    print(f"\nEvaluation Completed. Average completeness: {avg_score * 100}%")

    with open("eval_results.json", "w") as f:
        json.dump({"avg_completeness": avg_score, "runs": results}, f, indent=2)

if __name__ == "__main__":
    run_evaluation()