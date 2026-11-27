import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

DB_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    max_new_tokens=180,
    return_full_text=False
)

def ask_rag(query):
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": "You are a helpful University AI Assistant. Answer strictly using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    return generator(messages)[0]["generated_text"].strip()

if __name__ == "__main__":
    while True:
        query = input("\nAsk a university question (or exit): ")
        if query.strip().lower() == "exit":
            break
        print("\nAnswer:", ask_rag(query))


