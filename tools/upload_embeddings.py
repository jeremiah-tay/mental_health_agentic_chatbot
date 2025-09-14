import os
import glob
from openai import OpenAI
from backend.utils.pdf_loader import extract_text_from_pdf
from utils.supabase_client import get_supabase
from tools.chunker import chunk_text


openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase = get_supabase()q

def load_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {path}")
    
def embed_text(text):
    res = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

def upload_file(filepath, source_name="local_docs"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = chunk_text(content, max_chars=1500)
    rows = []
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        rows.append({
            "source": f"{source_name}:{filepath}",
            "content": chunk,
            "chunk_index": i,
            "embedding": emb
        })
    res = supabase.table("documents").insert(rows).execute()
    if res.error:
        print("Insert error:", res.error)
    else:
        print("Inserted:", len(rows))

if __name__ == "__main__":
    for fn in glob.glob("data/*.md"):
        upload_file(fn, source_name="kb_markdown")
