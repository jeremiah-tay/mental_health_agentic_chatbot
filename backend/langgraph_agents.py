# backend/langgraph_agents.py

from backend.agentools.rag_tool import rag_tool

# Later will need add supervising_agent and therapist_agent here
# For now, we only attach RAG tool to supervising agent
supervising_agent_tools = [rag_tool]

# Therapist agent tools will be added later
therapist_agent_tools = []
