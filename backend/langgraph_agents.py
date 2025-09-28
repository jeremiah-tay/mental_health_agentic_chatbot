# backend/langgraph_agents.py

from pydantic import BaseModel, Field
from langgraph.graph.state import StateGraph, START
from backend.utils.supabase_client import query_pgvector, client as openai_client
from typing import List, Dict, Any

# ---------------------------
# Define State Schema (Pydantic model)
# ---------------------------
class ChatState(BaseModel):
    query: str
    k: int = 5
    docs: List[Dict[str, Any]] = Field(default_factory=list)

# ---------------------------
# PGVector Node
# ---------------------------
def pgvector_node(state: ChatState):
    """Retrieve top-k chunks from Supabase."""
    query = state.query
    k = state.k
    docs = query_pgvector(query, k)
    #print("DEBUG pgvector_node docs:", docs)
    state.docs = docs
    return state

# ---------------------------
# Build Chatbot Graph
# ---------------------------
chat_graph_builder = StateGraph(ChatState)
chat_graph_builder.add_node("pgvector_node", pgvector_node)
chat_graph_builder.add_edge(START, "pgvector_node")
chat_graph = chat_graph_builder.compile()

# ---------------------------
# Helper to run chatbot easily
# ---------------------------
def run_chatbot(query: str, k: int = 5):
    state = ChatState(query=query, k=k)
    result = chat_graph.invoke(state)
    return result
