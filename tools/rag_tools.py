# tools/rag_tool.py
import os
import json
import pandas as pd
from datetime import datetime
from langchain.tools import tool
from pydantic import BaseModel, Field
from backend.utils.supabase_client import query_pgvector, client as openai_client

# --- Input schema for the tool ---
class RAGInput(BaseModel):
    query: str = Field(..., description="User's query to search from database")
    top_k: int = Field(5, description="Number of top documents to retrieve")

# --- Core RAG function ---
@tool
def rag_tool(query: str, top_k: int = 5):
    """Retrieve relevant context from the database and generate an empathetic answer.

    This tool is best for when a user asks for information, definitions, or explanations
    about mental health topics.

    Args:
        query (str): The user's specific question to search for in the database.
        top_k (int): The number of top documents to retrieve. Defaults to 5.
    """
    print(f"--- RAG TOOL: Calling RAG Tool with query: '{query}' ---")
    # 1. Retrieve docs using pgvector
    docs = query_pgvector(query, top_k)

    # 2. Build context text
    context_text = "\n\n".join([doc["content"] for doc in docs])

    # 3. Generate empathetic answer
    completion = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": (
                """
                You are a kind and empathetic mental health assistant.
                Your job is to answer the user's question based on the provided context.
                You must blend this factual information with a supportive, reassuring, and understanding tone to answer supportively."""
            )},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
    )
    answer = completion.choices[0].message.content

    # 4. Save outputs
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(docs)
    df.to_csv(f"outputs/rag_results_{timestamp}.csv", index=False)

    with open(f"outputs/rag_log_{timestamp}.json", "w") as f:
        json.dump({"query": query, "retrieved": docs, "answer": answer}, f, indent=2)

    print(f"Saved outputs to outputs/rag_results_{timestamp}.csv and outputs/rag_log_{timestamp}.json")

    return {"answer": answer, "retrieved_docs": docs}