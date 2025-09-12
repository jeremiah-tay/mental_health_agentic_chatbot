create table if not exists documents (
  id bigserial primary key,
  source text,
  content text not null,
  chunk_index int default 0,
  embedding vector(1536) not null,
  created_at timestamptz default now()
);
