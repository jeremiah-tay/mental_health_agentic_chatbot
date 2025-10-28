# backend/api_server.py
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import json

# LangChain message types (used in your graphs)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Reuse your existing utilities & graphs
from backend.utils.supabase_client import query_pgvector, client as openai_client, supabase
from backend.utils.pdf_loader import extract_text_from_pdf
from tools.rag_tools import rag_tool
from tools.cbt_tools import select_cbt_tool
from agents.supervisor import create_supervisor_graph  # returns compiled LangGraph
from agents.booking import create_booking_graph

# Risk classifier
from risk_classifier.safetycheck import SafetyCheck
from risk_classifier.crisis_response import CrisisResponse

load_dotenv()

# App config
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")  # optional header protection
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
OPENAI_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")  # for ChatOpenAI instances

# Initialize FastAPI
app = FastAPI(title="Mental Health Chatbot API (LangGraph + ML endpoints)")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Create Chat LLM instance used by LangGraph graphs
llm = ChatOpenAI(model=LLM_MODEL, api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.2)

# Create graphs (compiled). These are reused across requests.
supervisor_graph = create_supervisor_graph(llm)
booking_graph = create_booking_graph(llm)

# Load risk classifier (if available)
try:
    risk_classifier = SafetyCheck(base_dir="risk_classifier/saved_models")
    print("✅ Risk classifier loaded in API server")
except Exception as e:
    print(f"⚠️ Risk classifier not loaded: {e}")
    risk_classifier = None

# ----------- Helper functions -----------
def ensure_admin(request: Request):
    if ADMIN_API_KEY:
        header_key = request.headers.get("x-admin-key")
        if header_key != ADMIN_API_KEY:
            raise HTTPException(status_code=401, detail="unauthorized: invalid admin key")

def serialize_message(msg) -> Dict[str, Any]:
    """Convert HumanMessage/AIMessage to serializable dict."""
    return {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}

def parse_messages_list(raw_messages: List[Dict[str, str]]):
    """Convert list of {'role','content'} dicts into LangChain message objects used by graphs."""
    msgs = []
    for m in raw_messages:
        role = (m.get("role") or "").lower()
        content = m.get("content", "")
        if role in ("user", "human", "humanmessage"):
            msgs.append(HumanMessage(content=content))
        elif role in ("assistant", "ai", "aimessage"):
            msgs.append(AIMessage(content=content))
        else:
            # default to HumanMessage for unknown roles
            msgs.append(HumanMessage(content=content))
    return msgs

# ----------- Pydantic request/response models -----------
class ChatAPIRequest(BaseModel):
    messages: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]

class SimpleQuery(BaseModel):
    query: str
    k: Optional[int] = 5

class SafetyRequest(BaseModel):
    text: str

class CBTRequest(BaseModel):
    user_mental_health_concern: str

# ----------- Endpoints -----------

@app.get("/health")
def health():
    return {"status": "ok"}

# Direct PGVector sub-agent (raw retrieval)
@app.post("/subagent/query")
def subagent_query(payload: SimpleQuery):
    if not payload.query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        docs = query_pgvector(payload.query, top_k=payload.k)
        return {"hits": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# RAG tool wrapper (uses your rag_tool implementation)
@app.post("/rag")
def rag_endpoint(payload: SimpleQuery):
    if not payload.query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        # rag_tool is a langchain tool function; call directly and return result
        result = rag_tool(payload.query, top_k=payload.k)
        return result  # rag_tool returns dict with answer + retrieved_docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CBT technique selection / recommendation
@app.post("/get-recommendation")
def get_recommendation(payload: CBTRequest):
    if not payload.user_mental_health_concern:
        raise HTTPException(status_code=400, detail="user_mental_health_concern is required")
    try:
        rec = select_cbt_tool(payload.user_mental_health_concern)
        return rec
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Safety assessment endpoint (wrap local risk classifier)
@app.post("/check-safety")
def check_safety(payload: SafetyRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="text is required")
    if risk_classifier is None:
        raise HTTPException(status_code=500, detail="risk classifier not available on server")
    try:
        label = risk_classifier(payload.text)
        # label interpretation depends on your implementation: adapt as needed
        return {"status": "ok", "risk_label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload PDF and insert embeddings into Supabase (admin protected)
@app.post("/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...), source: str = "pdf_upload", chunk_size: int = 1500):
    ensure_admin(request)
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract text
    try:
        content = extract_text_from_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read pdf: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not content:
        raise HTTPException(status_code=400, detail="PDF has no extractable text")

    # Chunk and embed here (simple paragraph chunker)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= chunk_size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    # Create rows and insert
    rows = []
    for idx, chunk in enumerate(chunks):
        emb_res = openai_client.embeddings.create(model=OPENAI_MODEL, input=chunk)
        emb = emb_res.data[0].embedding
        rows.append({
            "source": source,
            "content": chunk,
            "chunk_index": idx,
            "embedding": emb
        })

    insert_res = supabase.table("documents").insert(rows).execute()
    if getattr(insert_res, "error", None):
        raise HTTPException(status_code=500, detail=str(insert_res.error))

    return {"inserted": len(rows), "source": source}

# Chat endpoint that invokes LangGraph supervisor graph
@app.post("/chat")
def chat_endpoint(payload: ChatAPIRequest):
    """
    Expects: { "messages": [{"role":"user"|"ai", "content":"..."} , ...] }
    The graph will be invoked with these messages as history.
    Returns: {"messages": [...]} with serialized messages.
    """
    raw_messages = payload.messages
    if not raw_messages:
        raise HTTPException(status_code=400, detail="messages are required")

    msgs = parse_messages_list(raw_messages)
    try:
        final_state = supervisor_graph.invoke({"messages": msgs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"graph invoke failed: {e}")

    # `final_state` is likely a dict-like pydantic model; extract messages
    out_messages = []
    for m in final_state.get("messages", []):
        out_messages.append(serialize_message(m))

    return {"messages": out_messages, "conversation_ended": final_state.get("conversation_ended", False)}

# Endpoint to directly invoke the booking subgraph (useful for frontend delegation)
@app.post("/booking")
def booking_endpoint(payload: ChatAPIRequest):
    raw_messages = payload.messages
    if not raw_messages:
        raise HTTPException(status_code=400, detail="messages are required")
    msgs = parse_messages_list(raw_messages)
    try:
        result = booking_graph.invoke({"messages": msgs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"booking invoke failed: {e}")

    out = [serialize_message(m) for m in result.get("messages", [])]
    return {"messages": out}

# Simple wrapper to call supabase RPC or other administrative endpoints can be added similarly

