# backend/app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .utils.supabase_client import get_supabase
from openai import OpenAI
import os
from typing import List

# NOTE: backend/utils/supabase_client.py should call load_dotenv() and export get_supabase().

# --- config
OPENAI_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")  # optional, set in .env to protect /upload
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")  # comma-separated or "*"

# openai client
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# supabase client (uses service_role key from backend/utils/supabase_client)
supabase = get_supabase()

app = FastAPI(title="Mental Health Chatbot - Retrieval API")

# CORS
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- small helper chunker (simple, paragraph-based)
def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out

# --- request bodies
class QueryRequest(BaseModel):
    query: str
    k: int = 5

class UploadRequest(BaseModel):
    content: str
    source: str = "manual_upload"
    chunk_size: int = 1500

# --- endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_docs(payload: QueryRequest):
    if not payload.query:
        raise HTTPException(status_code=400, detail="query is required")

    # 1) make embedding
    try:
        embedding_res = openai.embeddings.create(model=OPENAI_MODEL, input=payload.query)
        query_emb = embedding_res.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding generation failed: {e}")

    # 2) call Supabase RPC (match_documents) - expects vector and match_limit
    try:
        rpc_res = supabase.rpc("match_documents", {"query_embedding": query_emb, "match_limit": payload.k}).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"supabase rpc call failed: {e}")

    if rpc_res.error:
        raise HTTPException(status_code=500, detail=str(rpc_res.error))

    # rpc_res.data is the returned rows
    return {"hits": rpc_res.data}

from fastapi import UploadFile, File
from .utils.pdf_loader import extract_text_from_pdf
import tempfile

@app.post("/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...), source: str = "pdf_upload", chunk_size: int = 1500):
    # Admin protection
    if ADMIN_API_KEY:
        header_key = request.headers.get("x-admin-key")
        if header_key != ADMIN_API_KEY:
            raise HTTPException(status_code=401, detail="unauthorized")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract text
    try:
        content = extract_text_from_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read pdf: {e}")

    if not content:
        raise HTTPException(status_code=400, detail="PDF has no extractable text")

    # Reuse your chunk + embedding + insert logic
    chunks = chunk_text(content, max_chars=chunk_size)
    rows = []
    for idx, chunk in enumerate(chunks):
        emb_res = openai.embeddings.create(model=OPENAI_MODEL, input=chunk)
        emb = emb_res.data[0].embedding
        rows.append({
            "source": source,
            "content": chunk,
            "chunk_index": idx,
            "embedding": emb
        })

    insert_res = supabase.table("documents").insert(rows).execute()
    if insert_res.error:
        raise HTTPException(status_code=500, detail=str(insert_res.error))

    return {"inserted": len(rows), "source": source}

