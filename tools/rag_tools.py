# tools/rag_tool.py
import os
import json
import pandas as pd
from datetime import datetime
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from backend.utils.supabase_client import query_pgvector, client as openai_client

# --- Input schema for the tool ---
class RAGInput(BaseModel):
    query: str = Field(..., description="User's query to search from database")
    top_k: int = Field(5, description="Number of top documents to retrieve")

# --- Core RAG function ---
def run_rag(query: str, top_k: int = 5):
    # 1. Retrieve docs using pgvector
    docs = query_pgvector(query, top_k)

    # 2. Build context text
    context_text = "\n\n".join([doc["content"] for doc in docs])

    # 3. Generate empathetic answer
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a kind and empathetic mental health assistant. "
                "Use the provided context to answer supportively, offering reassurance and understanding."
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

# --- Define as LangGraph-compatible tool ---
rag_query = StructuredTool.from_function(
    func=run_rag,
    name="rag_query",
    description="Retrieve relevant context from database and generate an empathetic answer",
    args_schema=RAGInput
)
