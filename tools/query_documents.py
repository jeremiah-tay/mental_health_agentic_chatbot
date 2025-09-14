import os
from openai import OpenAI
from utils.supabase_client import get_supabase

openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase = get_supabase()

def embed_query(q):
    res = openai.embeddings.create(model="text-embedding-3-small", input=q)
    return res.data[0].embedding

def retrieve(query, k=5):
    emb = embed_query(query)
    res = supabase.rpc("match_documents", {"query_embedding": emb, "match_limit": k}).execute()
    if res.error:
        raise Exception(res.error)
    return res.data

if __name__ == "__main__":
    q = "How to calm down during a panic attack?"
    hits = retrieve(q, k=5)
    for h in hits:
        print(h["id"], h["source"], h["content"][:200], "...")
