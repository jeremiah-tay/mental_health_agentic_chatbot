def chunk_text(text: str, max_words: int = 150):
    words = text.split()
    out = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        out.append(chunk)
    return out
