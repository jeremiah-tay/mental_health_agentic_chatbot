# backend/api_server.py
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import LangGraph supervisor and other model tools
from agents.logger_supervisor import create_supervisor_graph
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
    history: List[Dict[str, str]] = Field(default_factory=list)  # optional list of prior messages (dicts)
    conversation_id: str  # NEW - conversation ID from frontend

# --- Pydantic schema for response ---
class ChatResponse(BaseModel):
    response: str
    messages: List[str]
    conversation_ended: bool
    risk_probability: float
    tools_called: List[str]
    tools_result: Dict
    agents_used: List[str]

# --- Root endpoint for testing ---
@app.get("/")
def root():
    return {"message": "✅ LangGraph Backend is running"}

# --- Main chat endpoint ---
@app.post("/chat", response_model=ChatResponse)
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
        
        # Prepare state for supervisor graph
        state = {
            "messages": messages,
            "conversation_ended": False,
            "tools_called": [],
            "tools_result": {},
            "agents_used": [],
            "risk_probability": 0.0
        }
        
        # Add conversation_id if provided
        if req.conversation_id:
            state["conversation_id"] = req.conversation_id
        
        # Invoke the supervisor graph
        final_state = supervisor_graph.invoke(state)
        final_messages = final_state["messages"]
        
        # Get last message content
        reply = final_messages[-1].content if final_messages else "No response."
        
        # Extract all state information
        conversation_ended = final_state.get("conversation_ended", False)
        risk_probability = final_state.get("risk_probability", 0.0)
        tools_called = final_state.get("tools_called", [])
        tools_result = final_state.get("tools_result", {})
        agents_used = final_state.get("agents_used", [])
        
        # Return comprehensive response
        return ChatResponse(
            response=reply,
            messages=[m.content for m in final_messages],
            conversation_ended=conversation_ended,
            risk_probability=round(float(risk_probability), 5), 
            tools_called=tools_called,
            tools_result=tools_result,
            agents_used=agents_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))