# Agentic Vehicle sales, Rental & Support RAG system

The Vehicle sales, Rental & Support RAG system is an advanced, domain-isolated Retrieval-Augemnted Generation (RAG) assistant.

This is developed to reduce communication friction between dealership operations and customers by acting as an intelligent pre-qualification interface.

The system orchestrates a LangChain tool-calling agent to answer complex, multi-hop user queries spanning five distinct operational domain. By using strict metadata filtering within a vector database, the agent ensures absolute factual correctness, prevents domain cross-contamination and safety defaults to human handoff rather than generating dangerous hallucinations.

### Tech Stack
- chromaDB
- HuggingFace Embeddings
- Agentic framework(langchain-classic utillizing AgentExecutor and create_tool_calling_agent for multi-hop tool routing)
- Groq LLM (llama-3.1-8b-instant)
- Gradio UI

## setup instrutions

1. **clone the repository:**
git clone <your-github-repository-link>
   cd vehicle-sale-rental-rag

2. **Install dependencies:**
uv add -r requirements.txt

3. **Add groq api key:**
create a file named .env in the root directory and add your key inside:

GROQ_API_KEY=your_groq_api_key

## How to run
1. open answer.ipynb and run all cells.
2. Run the automated evaluation harness.
     uv run evaluate.py
3. Launch the user interface.
     uv run main.py