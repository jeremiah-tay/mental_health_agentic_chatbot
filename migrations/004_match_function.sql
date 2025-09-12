create or replace function match_documents(
  query_embedding vector(1536),
  match_limit int default 5
)
returns table (
  id bigint,
  source text,
  content text,
  distance float
)
language sql
as $$
  select id, source, content, embedding <=> query_embedding as distance
  from documents
  order by distance
  limit match_limit;
$$;
