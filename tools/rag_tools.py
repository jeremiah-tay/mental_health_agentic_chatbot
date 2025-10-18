# backend/tools/rag_tool.py

from langchain.tools import tool
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils.supabase_client import query_pgvector, client as openai_client

@tool("rag_tool", return_direct=True)
def rag_tool(query: str, k: int = 5):
    """Retrieve information from the PGVector database and respond empathetically."""
    print(f"--- RAG TOOL: Calling RAG Tool with query: '{query}' ---")
    docs = query_pgvector(query, k)

    # Combine retrieved docs into one text
    retrieved_text = " ".join(d["content"] for d in docs)

    # Use LLM to craft empathetic response
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content":
             "You are a supportive assistant. Always sound empathetic, warm, and approachable. "
             "Ground your answer in the provided context, but speak as if to a friend."},
            {"role": "user", "content": f"User asked: {query}\n\nContext:\n{retrieved_text}"}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "retrieved_docs": docs
    }
