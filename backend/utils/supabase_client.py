import os
from supabase import create_client
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_KEY)


def query_pgvector(question: str, top_k: int = 5):
    """Sub-agent: retrieve similar chunks from Supabase (PGVector)."""
    # Step 1: Embed the question
    embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Step 2: Call Supabase RPC (match_documents from migrations SQL)
    response = supabase.rpc(
        "match_documents",  # this was defined in 004_match_function.sql
        {"query_embedding": embedding, "match_limit": top_k}
    ).execute()

    matches = response.data
    return matches
