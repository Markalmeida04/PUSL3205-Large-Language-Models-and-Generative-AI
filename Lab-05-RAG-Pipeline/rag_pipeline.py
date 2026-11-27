import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

DB_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)

def ask_rag(query):
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"Answer the university question strictly using the provided context. "
        f"If the answer is not in the context, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    return generator(prompt)["generated_text"].strip()

if __name__ == "__main__":
    while True:
        query = input("\nAsk a university question (or exit): ")
        if query.strip().lower() == "exit":
            break
        print("\nAnswer:", ask_rag(query))

