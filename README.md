RAG Challenge
This repository contains my complete solution to the DeepSeek RAG Challenge.The goal of this challenge is to demonstrate understanding of:
* Document preprocessing
* Chunking strategies
* Dense / Sparse / Hybrid retrieval
* RAG pipeline design
* Evaluation methodology
* Using an LLM to generate responses using retrieved context
* Building an advanced agent with LangGraph

This project implements a full Retrieval-Augmented Generation (RAG) Agent using:
LangGraph (graph-based agent framework)
Groq Llama-3.3-70B (fast inference)
Qdrant (sparse BM25 retrieval)
FastEmbed sparse embeddings
Multi-turn memory via conversation summarization
This README documents the full experimental process, findings, trade-offs, and final decisions.

Repository Structure

RAG-Challenge/
│
├── Chunking+Retrieval.ipynb        # Experiments: chunking, retrieval, evaluation
├── RAG-Agent.ipynb                 # Experiments: prototype RAG agent logic
│
├── Deepseek-r1.pdf                 # Source document
├── chunks_fixed.json               # Final chunked dataset (text + IDs)
├── ground_truth.json               # 20 labeled QA pairs for evaluation
│
├── requirements.txt                # Dependencies for notebooks
│
├── qdrant_db/                      # Local Qdrant storage (auto-created)
│
└── rag-agent/                      # Final LangGraph agent submission
    ├── data/
    │   └── chunks_fixed.json       # LangGraph API config (entrypoint)
    │
    ├── qdrant_db/                  # Qdrant DB used by LangGraph (auto-created)
    │
    ├── src/agent/
    │   ├── build_qdrant.py         # Script to build Qdrant sparse index
    │   └── graph.py                # Main LangGraph RAG Agent implementation
    │
    ├── rag3.11/                    # Local Python environment
    ├── Makefile
    ├── pyproject.toml
    └── README.md

1. Installation (LangGraph Project Environment)

Move into the agent package:
```bash
cd rag-agent
```

Create and activate Python 3.11+ environment:
```bash
python3.11 -m venv rag3.11
source rag3.11/bin/activate
pip install -r requirements.txt
```

Install LangGraph local runtime:
```bash
pip install -U "langgraph-cli[inmem]"
```

Objective
Given the DeepSeek-R1 technical report (PDF), the task is to:
1. Prepare the document (cleaning, chunking, processing)
2. Implement multiple retrieval methods
3. Evaluate retrieval performance
4. Design a RAG system based on the best method
5. Explain decisions, findings, and trade-offs


1. Data Preparation
PDF Processing

The DeepSeek-R1 technical report was extracted using pypdf, performing page-level text extraction to ensure that all paragraphs, headings, and figure captions were captured consistently. The extracted raw text was then prepared for downstream chunking and retrieval experiments.

Chunking Experiments

To understand how different segmentation strategies affect retrieval performance, we evaluated two distinct chunking approaches: fixed-size chunking and content-aware heading-based chunking.

A. Fixed-size Chunking

This method uses LangChain’s RecursiveCharacterTextSplitter to divide the document into overlapping character windows. The configuration applied was:

chunk_size = 800

chunk_overlap = 150

This approach produces chunks of consistent size, which stabilizes embedding behavior and ensures that each unit contains enough text for meaningful representation. The uniformity makes it well-suited for embedding-based retrieval methods, where vector quality depends on a balanced amount of contextual information. Additionally, it performs reliably with sparse retrieval such as BM25, because terms remain grouped within moderately sized segments rather than being scattered across very small pieces.


B. Content / Heading-Based Chunking

This approach attempts to segment the document according to its structural hierarchy. We applied a regex rule such as:

\n(?=\d[\d\.]*\s[A-Z])


to identify headings like “1 Introduction”, “2.3 DeepSeek-R1-Zero”, etc. This strategy aims to produce chunks aligned with human-interpretable sections of the paper.

Heading-based chunking better preserves the logical flow of the document. Each chunk corresponds to a meaningful conceptual unit, improving interpretability and aligning more naturally with how readers understand the material. Because the text remains semantically intact, this method is intuitively appealing for tasks requiring structured understanding.

2. Retrieval Component
Implemented and compared three retrieval strategies:

A. Dense Retrieval (Embeddings)
Model used:
* jinaai/jina-embeddings-v2-small-en
* BAAI/bge-large-en-v1.5
* sentence-transformers/all-MiniLM-L6-v2

B. Sparse Retrieval (BM25)
Model:
* Qdrant/bm25 via FastEmbed
Stored in Qdrant as a sparse vector field.

C. Hybrid (Dense + Sparse + ColBERT Reranking)
Hybrid pipeline:
* Dense vectors (Jina/sentence-transformers/all-MiniLM-L6-v2)
* Sparse BM25 vectors
* Late-interaction ColBERT reranker
* Multi-vector Qdrant collection
* Prefetch + reranking
Hybrid search did improve some semantic queries but did not justify its additional complexity.

3. Evaluation

To compare different retrieval strategies and chunking approaches, I evaluated all combinations of:

Dense retrieval (Jina embedding model)
Sparse retrieval (BM25 via Qdrant SparseVectors)
Hybrid retrieval (Dense + Sparse with ColBERT reranking)
Fixed-size chunking
Heading-based chunking

Each method was tested using a set of 20 manually verified ground-truth Q&A pairs derived from the DeepSeek-R1 technical report.
For each query, I computed:

Hit@k (k ∈ {1, 3, 5, 10})
Mean Reciprocal Rank (MRR)

3.1 Quantitative Results

The following table summarizes the retrieval accuracy across all methods:

method	chunking	hit@1	hit@3	hit@5	hit@10	MRR
Dense	fixed	0.20	0.20	0.20	0.30	0.2146
Dense	heading	0.05	0.15	0.15	0.25	0.0944
Sparse	fixed	0.30	0.35	0.45	0.55	0.3519
Sparse	heading	0.10	0.25	0.30	0.40	0.1901
Hybrid	fixed	0.20	0.25	0.30	0.50	0.2528
Hybrid	heading	0.10	0.30	0.35	0.35	0.2100

Sparse Retrieval Outperformed Dense and Hybrid

Sparse BM25 retrieval delivered:
The highest Hit@1 (0.30)
The highest Hit@5 (0.45)
The highest Hit@10 (0.55)
The highest MRR (0.3519)

This occurred because the DeepSeek-R1 PDF's characteristics strongly favor sparse term-matching.
Dense embedding models (especially small ones) sometimes blur fine-grained distinctions between technical concepts.
Hybrid retrieval, while better than dense alone, did not outperform sparse retrieval, likely due to:
ColBERT reranking requiring longer, more structured query–document matches

Final Choice for RAG System: Fixed-size chunking (800 / 150) + Sparse Retrieval (BM25)

This configuration achieves the best balance between accuracy, stability, and cost efficiency.

4. Final RAG System Architecture (LangGraph Agent)

The final system integrates retrieval, generation, memory, and tool routing into a unified agent powered by LangGraph. The agent is capable of determining when external retrieval is required, maintaining long-running conversation context, and summarizing older history to improve efficiency.

This section explains the system architecture and the logic behind key design choices.

4.1 Overall System Architecture

The final RAG pipeline consists of the following components:

1) Sparse Retrieval Tool
Uses BM25 through Qdrant’s SparseVectors to retrieve the most relevant paper chunks.

2) LLM Reasoning Layer (Groq-hosted LLM)
The assistant model decides dynamically whether a query requires external context (tool use) or can be answered directly.

3) LangGraph Execution Framework
Coordinates control flow between:
LLM reasoning
Tool execution
Memory summarization

4) Conversation Memory with Summarization
When the conversation exceeds six messages:
Older messages are summarized into a compact form
The summary is injected into the system prompt on subsequent turns
This maintains context while reducing token usage

The LangGraph agent follows a ReAct-style architecture:

1) Assistant

Core reasoning step
Determines whether a tool call is needed
Integrates conversation summary (if available)

2) Tool Node (Sparse Retrieval)

Executes BM25 search using the user's query
Returns top-k chunks from the DeepSeek-R1 paper
Output is fed back into the assistant for reasoning and final response

3) Summarization Node

Triggered automatically when len(messages) > 6
Produces a condensed summary of the full conversation so far
Older messages are replaced with the summary, reducing state size

4) MemorySaver Checkpointer

Enables persistent state across turns
Allows the conversation to continue naturally across multiple invocations