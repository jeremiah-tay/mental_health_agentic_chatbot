import os
from tools.upload_embeddings import process_file

PDF_DIR = "pdf/"

for fname in os.listdir(PDF_DIR):
    if fname.lower().endswith(".pdf"):
        path = os.path.join(PDF_DIR, fname)
        print(f"Processing {path} ...")
        process_file(path)
