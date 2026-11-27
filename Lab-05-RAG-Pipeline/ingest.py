import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "docs"
DB_DIR = "chroma_db"

def ingest_documents():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        print("No PDF files found in docs/ folder. Please add at least one PDF.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"Successfully stored embeddings in '{DB_DIR}'!")

if __name__ == "__main__":
    ingest_documents()
