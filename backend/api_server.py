# backend/api_server.py
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import LangGraph supervisor and other model tools
from agents.supervisor import create_supervisor_graph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Create FastAPI app
app = FastAPI(title="LangGraph Backend API", version="1.0")

# --- Initialize LLM and Graph ---
llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
supervisor_graph = create_supervisor_graph(llm)

# --- Pydantic schema for requests ---
class ChatRequest(BaseModel):
    message: str
    history: list = []  # optional list of prior messages (dicts)

# --- Root endpoint for testing ---
@app.get("/")
def root():
    return {"message": "✅ LangGraph Backend is running"}

# --- Main chat endpoint ---
@app.post("/chat")
def chat_with_graph(req: ChatRequest):
    try:
        # Convert history to LangChain message objects
        messages = []
        for m in req.history:
            if m.get("role") == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "assistant":
                messages.append(AIMessage(content=m["content"]))

        # Add new user message
        messages.append(HumanMessage(content=req.message))

        # Invoke the supervisor graph
        final_state = supervisor_graph.invoke({"messages": messages})
        final_messages = final_state["messages"]

        # Get last message content
        reply = final_messages[-1].content if final_messages else "No response."

        return {"response": reply, "messages": [m.content for m in final_messages]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


