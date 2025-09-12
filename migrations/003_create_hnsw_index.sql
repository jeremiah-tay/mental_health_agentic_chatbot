create index if not exists documents_embedding_hnsw
  on documents using hnsw (embedding vector_cosine_ops);
