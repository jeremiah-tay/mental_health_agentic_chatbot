import os
from dotenv import load_dotenv 
import json
import fitz  # PyMuPDF
import psycopg2
import requests
from bs4 import BeautifulSoup
from psycopg2.extras import execute_values
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def chunk_pdf(pdf_path: str) -> list[Document]:
    """
    Opens a PDF, extracts its text, and splits it into processed chunks.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list[Document]: A list of LangChain Document objects, each representing a
                        chunk of text with populated metadata.
    """
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)

    # Extract all text from the PDF into a single string
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n\n"
    
    doc.close() # Close the document after extracting text

    # Create a single Document object for the entire text
    doc_for_splitting = Document(page_content = full_text)

    # 3. Split the document into chunks
    chunks = splitter.split_documents([doc_for_splitting])

    # 4. Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = os.path.basename(pdf_path)
        chunk.metadata["chunk_number"] = i + 1

    return chunks


def insert_chunk_to_db(pdf_path: str, db_url: str):
    doc_chunks = chunk_pdf(pdf_path)
    data_to_insert = []

    for chunk in doc_chunks:
        content = chunk.page_content
        metadata = chunk.metadata
        source = metadata.get('source')
        chunk_number = metadata.get('chunk_number')

        data_to_insert.append((
            json.dumps(metadata),
            source,
            chunk_number,
            content
        ))

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                query = "INSERT INTO rag_text_chunks (metadata, source, chunk_number, content) VALUES %s"
                execute_values(cur, query, data_to_insert)
            
            # Explicitly commit the transaction
            conn.commit()
            
            # Only print success AFTER the commit is successful
            print(f"Successfully committed {len(data_to_insert)} chunks from {os.path.basename(pdf_path)}.")

    except psycopg2.Error as e:
        # We add a clearer error message here
        print(f"Database error for {os.path.basename(pdf_path)}: {e}")
        # Rollback the transaction on error
        if 'conn' in locals() and conn:
            conn.rollback()

if __name__ == "__main__":
    pdf_directory = "pdf"
    pdf_path = "pdf/5-action-steps-help-someone-having-thoughts-suicide.pdf"
    db_url = os.getenv('DATABASE_URL')
    print(db_url)
    
    if not db_url:
        print("Error: DATABASE_URL environment variable not set.")
    elif not os.path.isdir(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' not found.")
    else:
        print(f"Starting to process PDF files in '{pdf_directory}' directory...")
        # Loop through all files in the specified directory
        for filename in os.listdir(pdf_directory):
            # Check if the file has a .pdf extension (case-insensitive)
            if filename.lower().endswith(".pdf"):
                # Construct the full path to the PDF file
                full_pdf_path = os.path.join(pdf_directory, filename)
                
                # Call the function to process and insert this single PDF
                insert_chunk_to_db(full_pdf_path, db_url)
        print("Processing complete.")