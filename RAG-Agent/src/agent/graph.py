from typing import Optional, List, Literal
from pydantic import BaseModel
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langgraph.prebuilt import tools_condition, ToolNode

from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
import os
import json
from langchain_core.tools import tool
from typing import Optional

class AgentState(MessagesState):
    summary: str = ""


sparse_embedder = SparseTextEmbedding("Qdrant/bm25")
client = QdrantClient(path="qdrant_db")

def retrieve(query: str, top_k: int = 5) -> str:
    """Sparse BM25 retrieval from Qdrant."""
    q_vec = next(sparse_embedder.query_embed(query))

    hits = client.query_points(
        collection_name="deepseek_sparse_fixed",  
        query=models.SparseVector(**q_vec.as_object()),
        using="bm25",
        limit=top_k,
        with_payload=True
    ).points

    if not hits:
        return "No relevant document found."

    return {
    "chunks": [
        {
            "chunk_id": hit.payload["chunk_id"],
            "text": hit.payload["text"]
        }
        for hit in hits
    ]
}

@tool
def rag_search(query: str) -> str:
    """Retrieve DeepSeek-R1 paper chunks from Qdrant."""
    result = retrieve(query)
    return json.dumps(result, ensure_ascii=False)


llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=" ")
tools = [rag_search]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content=
"""You are a DeepSeek-R1 expert assistant.
You have access to a retrieval tool `rag_search`.

When the user asks about DeepSeek-R1 paper content, citations, or details, 
you MUST respond using a tool call in the following JSON format:

<tool_call>
{"tool": "rag_search", "query": "..."}
</tool_call>

When the tool is not needed, answer normally.
"""
)

def assistant(state: AgentState):
    messages = []
    summary = state.get("summary", "")

    if summary:
        messages.append(
            SystemMessage(content=f"Summary of conversation earlier:\n{summary}")
        )

    messages += state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}




tool_node = ToolNode(tools)

def summarize(state: AgentState):

    prev_summary = state.get("summary", "")
    messages = state["messages"]

    # summary prompt
    if prev_summary:
        prompt = (
            f"Current summary:\n{prev_summary}\n\n"
            "Extend the summary using ONLY the new messages."
        )
    else:
        prompt = "Create a_summary of the conversation above."

    # let LLM produce new summary
    summary_msg = llm.invoke(messages + [HumanMessage(content=prompt)])
    new_summary = summary_msg.content

    # remove old messages but KEEP last 2 for continuity
    trimmed_messages = messages[-2:]


    return {
        "summary": new_summary,
        "messages": trimmed_messages
    }



def route_from_assistant(state: AgentState):
    tool_decision = tools_condition(state)
    if tool_decision == "tools":
        return "tools"

    if len(state["messages"]) > 6:
        return "summarize"

    return END   


builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)
builder.add_node("summarize", summarize)

# START → assistant
builder.add_edge(START, "assistant")

# assistant → tools (if tool call) OR END (if no tool)

builder.add_conditional_edges(
    "assistant",
    route_from_assistant,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END
    }
)
# tools → assistant (ReAct Loop)
builder.add_edge("tools", "assistant")

# summarize → END
builder.add_edge("summarize", END)

# enable memory
memory = MemorySaver()
graph = builder.compile()