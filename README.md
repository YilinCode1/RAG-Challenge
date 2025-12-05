# RAG-Agent: Retrieval-Augmented Generation with LangGraph & LangSmith

This repository implements a complete Retrieval-Augmented Generation (RAG) pipeline using Qdrant, FastEmbed sparse embeddings, and a production-ready LangGraph agent featuring tools, memory, and conversation summarisation.
It also includes notebooks for data preparation, retrieval experimentation, and agent prototyping.

## Repository Structure

```
RAG-Agent/
│
├── data/
│   └── chunks_fixed.json            # Finalised text chunks used for Qdrant indexing
│
├── src/agent/
│   ├── __init__.py
│   ├── build_qdrant.py              # Script to build the local Qdrant DB
│   ├── graph.py                     # LangGraph agent definition (RAG tool + memory)
│   └── requirements.txt             # Dependencies required for the agent
│
├── Chunking+Retrieval.ipynb         # Notebook: chunking, sparse embedding, retrieval experiments
├── RAG-Agent.ipynb                  # Notebook: testing the LangGraph RAG agent
├── Deepseek-r1.pdf                  # Source document
├── chunks_fixed.json                # Duplicate dataset for notebook convenience
├── ground_truth.json                # 20 labelled QA pairs for evaluation
├── README.md                        # Project documentation
└── requirements.txt                 # Notebook dependencies
```


# 1. Installation

Create a clean environment (Python ≥ 3.11 recommended)

```bash
python3.11 -m venv rag-env
source rag-env/bin/activate
```
Install dependencies for the agent:

```bash
pip install -r src/agent/requirements.txt
```
Install dependencies for the notebooks:

```bash
pip install -r requirements.txt
```


# 2. Build the Qdrant Database

Before running the agent, you must build the local Qdrant index:

```bash
cd src/agent
python build_qdrant.py
```

This script:

* Loads `chunks_fixed.json`
* Computes sparse BM25 embeddings (FastEmbed)
* Creates a local Qdrant database at `qdrant_db/`
* Populates the `deepseek_sparse_fixed` collection

You only need to run this once.


# 3. Running the LangGraph RAG Agent + LangSmith Integration (Debugging + Tracing)

To enable LangSmith:

```bash
export LANGSMITH_API_KEY="YOUR_KEY"
export LANGCHAIN_TRACING_V2="true"
```

Then run:

```bash
langgraph dev
```

All runs will automatically appear at:

**[https://smith.langchain.com](https://smith.langchain.com)**

There you can inspect:

* Graph-level execution
* Tool call results
* Message state changes
* Conversation summary updates

# 4. Notebooks (Experimentation)

### Chunking+Retrieval.ipynb

This notebook demonstrates:

* Data extraction from PDF
* Text chunking strategy
* Sparse BM25 embedding generation
* Retrieval evaluation using `ground_truth.json`

### RAG-Agent.ipynb

This notebook includes:

* Local testing for your LangGraph agent
* Tool usage examples
* Memory & summarization demonstration

# Final Report — DeepSeek RAG Challenge

This repository contains my complete solution to the DeepSeek RAG Challenge.
The goal of this challenge is to demonstrate understanding of:

* Document preprocessing
* Chunking strategies
* Dense / Sparse / Hybrid retrieval
* RAG pipeline design
* Evaluation methodology
* Using an LLM to generate responses with retrieved context
* Building an advanced agent with LangGraph

# Objective

Given the DeepSeek-R1 technical report (PDF), the task is to:

1. Prepare the document (cleaning, chunking, processing)
2. Implement multiple retrieval strategies
3. Evaluate retrieval performance
4. Choose the best-performing method
5. Build a full RAG system and an advanced LangGraph agent
6. Explain decisions, findings, and trade-offs

# 1. Data Preparation

## 1.1 PDF Processing

The DeepSeek-R1 technical report was extracted using pypdf with page-level text extraction to ensure all paragraphs, headings, and figure captions were captured.
The raw text was then normalised and cleaned for chunking experiments.

## 1.2 Chunking Experiments

To study how chunking affects retrieval performance, two major chunking strategies were evaluated:

### A. Fixed-size Chunking

Using LangChain’s `RecursiveCharacterTextSplitter`:

```python
chunk_size = 800
chunk_overlap = 150
```
✔ Produces consistent chunk sizes
✔ Good for embedding-based retrieval
✔ Works extremely well with sparse retrieval
✔ Ensures meaningful semantic units without being too large

### B. Heading-Based Chunking

Chunk boundaries were derived from structural section headers using a regex:

```regex
\n(?=\d[\d\.]*\s[A-Z])
```

This matches headings like:

* `1 Introduction`
* `2.3 DeepSeek-R1-Zero`

Preserves logical document structure and Human-aligned semantic segmentation

# 2. Retrieval Component

Three retrieval strategies were implemented and compared.

## A. Dense Retrieval

Embedding models tested:

* `jinaai/jina-embeddings-v2-small-en`
* `BAAI/bge-large-en-v1.5`
* `sentence-transformers/all-MiniLM-L6-v2`

## B. Sparse Retrieval (BM25)

* Implemented using FastEmbed + Qdrant SparseVectors
* Stored directly in Qdrant as sparse embeddings
* No dense vectors required

## C. Hybrid Retrieval (Dense + Sparse + ColBERT Reranking)

Pipeline:

1. Dense vector retrieval
2. Sparse BM25 retrieval
3. ColBERT late-interaction reranker
4. Merge + rerank

Although hybrid retrieval improved certain semantic queries, it did not outperform pure sparse retrieval.

# 3. Evaluation

Each retrieval method was tested using 20 manually curated ground-truth Q&A pairs derived from the DeepSeek-R1 paper.

Metrics:

* Hit@k (k ∈ {1, 3, 5, 10})
* MRR (Mean Reciprocal Rank)

All combinations of:

* Dense / Sparse / Hybrid retrieval
* Fixed-size / Heading-based chunking

were evaluated.

## Quantitative Results
<img width="469" height="165" alt="Result" src="https://github.com/user-attachments/assets/5e5679de-1d64-49a8-bcd3-cc39ba9c2d71" />

### **Sparse Retrieval Outperformed Dense and Hybrid Approaches**

## Final Retrieval Choice

> **Fixed-size chunking (800/150) + Sparse Retrieval (BM25)**
> Best balance of accuracy, simplicity, cost, and robustness

# 4. Final RAG System Architecture (LangGraph Agent)

The final system integrates:

* Sparse retrieval (tool)
* LLM reasoning
* Conversation summarisation
* Persistent memory
* ReAct-style routing

<img width="258" height="334" alt="Screenshot 2025-12-05 at 15 59 56" src="https://github.com/user-attachments/assets/87cce866-5e5f-493f-89f9-b9c1463d3966" />

### 1) Sparse Retrieval Tool

* Uses BM25 via Qdrant
* Returns the top-k relevant chunks
* Triggered only when needed

### 2) LLM Reasoning Layer (Groq)

Responsibilities:

* Decide whether retrieval is needed
* Consume retrieved context
* Integrate conversation summary
* Produce final responses

### 3) LangGraph Controller

Handles:

* Node execution
* Conditional routing
* ReAct-style tool invocation loops
* Automatic summarisation

### 4) Memory + Summarization

When `len(messages) > 6`:

* A summary of the conversation is generated
* Older messages are removed
* Summary injected into system prompt

This enables long conversations without exceeding token limits.

## ReAct-style Interaction

1. Assistant interprets user input
2. If external information is needed → triggers `rag_search` tool
3. Tool returns JSON context
4. The assistant uses the retrieved data to answer
5. Memory module summarises when conversation grows

<img width="965" height="764" alt="Screenshot 2025-12-05 at 14 48 40" src="https://github.com/user-attachments/assets/40a8eee1-0706-4ffe-b138-d6dde365c599" />


