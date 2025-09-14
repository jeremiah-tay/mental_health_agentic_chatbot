def chunk_text(text, max_chars=2000):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur: out.append(cur)
            cur = p
    if cur: out.append(cur)
    return out
