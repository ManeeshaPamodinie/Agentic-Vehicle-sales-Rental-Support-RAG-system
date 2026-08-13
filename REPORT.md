# Agentic Vehicle sales, Rental & Support RAG system

## 1. Project Overview

The Vehicle sales, Rental & Support RAG system is an advanced, domain-isolated Retrieval-Augmented Generation (RAG) assistant.

This is developed to reduce communication friction between dealership operations and customers by acting as an intelligent pre-qualification interface.

 The system ensures:
 Accurate responses grounded only in provided documents
 Source attribution is included for transparency
 Safe refusal when information is not available.

  ## 2. Tech Stack
 Document loader = langchain DirectoryLoader
 
 Text Splitting = RecursiveCharacterTextSplitter
 
 Embeddings = sentence-transformers/all-MiniLM-L6-v2(HuggingFace)
 
 Vector DB = ChromaDB
 
 LLM = Groq - openai/gpt-oss-20b
 
 UI - Gradio ChatInterface
 
 Evaluation - custom python script 

## 3. Stage 1 - Document Ingestion

All markdown files are loaded using DirectoryLoader

Metadata Extraction : Extracted the directory name as the 'category' attribute and normalized the file path as the 'source' attribute.

Outcome : 15 documents successfully loaded.

## 4. Stage 2 - Chunking Strategy

Documents are split using:
- chunk size = 600 characters
- overlap = 100 characters

Outcome : 66 text chunks created.

## 5.  Stage 3 - Embeddings & Vector store

Embedding model : sentence-transformers/all-MiniLM-L6-v2

Vector database = ChromaDB


Features:
- Stores embeddings + metadata
- Avoid recomputation using persistence
- Loads existing DB if available

Outcome : 66 Vectors in database.

## 6. Stage 4 - Routing & Retrieval System

This implements a Domain-specific tool calling architecture using Langchain @tool decorators. This ensures the LLM agent explicity orhestrates its own data retrieval by selecting the exact database slice needed for the user's intent.

Core Retrieval Mechanism
- Metadata-filtered isolation(retrieve_from_domain) : A parameterized lookup function executes a target vector search(k=3)using chromaDB metadata filters.

- source tracking & citations : The retrieval layer structrally wraps each context block inside an explict string format, feeding verifiable citation paths directly to the LLM to eliminate hallucinations.

## 7. Stage 5 - Generation (LLM layer)

System Prompt - The model restricted to retrieved context only to avoid hallucination.

## 8. Stage 6 - Gradio UI

Multi-turn conversation support
History-aware responses

## 9. Evaluation Score

"avg_completeness": 0.686

## 10. What worked well

1. strict metadata driven isolation 

- By passing explicit domain categories into the filter={"category":category} parameter the retrieval engine safely completely partitioned data slices.
- for example in id:1 the model pulled high precison facts across two fields without matching overlapping text across unrelated document clusters.

2. instant single tool retrieval velocity

- Queries requiring access to only a single tool segment or landing directly on fallback flags processed with exceptional speed.
- id:2 and id:3 achieved execution speeds under one second, showcasing high optimization when deep multi-hop interface thinking loops are not required.

3. programmatic fallback resilliency

The agent sucessfully handled out of scope prompts. id:8 correctly identifies out of bounds user parameters and applied defensive execution flags.("fallback_applied": true)


## 11. Challenges Faced

1. Granular context erasure

The primary factor lowering the metric down to 68.6% average completeness score.

2. Keyword-based scoring does not fully capture semantic correctness.

- The pipeline occasionally synthesized correct natural-language answers that were penalized with a '0.0' score simply because they omitted precise phrase anchors.

3. Minor sensitivity to retrieval chunk selection(top k-tuning)

## Hugging face deployment

link: https://manee-sha12-vehicle-rag-app.hf.space
