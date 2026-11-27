import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import sqlite3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline


conn = sqlite3.connect("agent_memory.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT)")
conn.commit()

def save_memory(fact: str):
    cursor.execute("INSERT INTO memories (fact) VALUES (?)", (fact,))
    conn.commit()
    print(f"[SAVED TO SQLITE MEMORY] {fact}")

def search_memory() -> str:
    cursor.execute("SELECT fact FROM memories")
    rows = [r[0] for r in cursor.fetchall()]
    return "; ".join(rows) if rows else "No stored user memories yet."


def calculator_tool(expression: str) -> str:
    clean = re.sub(r"[^0-9+\-*/().]", "", expression)
    try:
        return f"Calculation result: {eval(clean)}"
    except Exception:
        return "Could not evaluate expression."


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_db_path = os.path.join("..", "Lab-05-RAG-Pipeline", "chroma_db")
vectorstore = Chroma(persist_directory=rag_db_path, embedding_function=embeddings) if os.path.exists(rag_db_path) else None

def rag_search_tool(query: str) -> str:
    if vectorstore is None:
        return "University RAG database not found. Run Lab 05 ingest.py first."
    docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    return "\n".join(d.page_content for d in docs)


llm = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=150, return_full_text=False)

def agent_chat(user_input: str) -> str:
    
    if "my name is" in user_input.lower() or "i am" in user_input.lower() or "i study" in user_input.lower():
        save_memory(user_input)

    q = user_input.lower()
    tool_context = ""

    
    if any(op in q for op in ["+", "-", "*", "/", "calculate"]):
        print("[TOOL SELECTED] calculator_tool")
        tool_context = calculator_tool(user_input)
    elif "my name" in q or "remember" in q or "who am i" in q:
        print("[TOOL SELECTED] search_memory (SQLite)")
        tool_context = f"Stored User Memory: {search_memory()}"
    elif any(k in q for k in ["university", "module", "lab", "assignment", "course", "subject", "brief"]):
        print("[TOOL SELECTED] rag_search_tool (Lab 05 ChromaDB)")
        tool_context = f"University Knowledge Base:\n{rag_search_tool(user_input)}"
    else:
        tool_context = f"Stored User Memory: {search_memory()}"

    print(f"[TOOL OUTPUT] {tool_context[:200]}...")

    messages = [
        {"role": "system", "content": "You are an Intelligent Student AI Agent with SQLite memory, calculator, and university RAG tools. Answer concisely using the tool context."},
        {"role": "user", "content": f"Tool Context:\n{tool_context}\n\nUser Question: {user_input}"}
    ]
    return llm(messages)[0]["generated_text"].strip()

if __name__ == "__main__":
    print("=== INTELLIGENT STUDENT AI AGENT (SQLite Memory + Calculator + RAG) ===")
    while True:
        user_msg = input("\nYou (or 'exit'): ")
        if user_msg.strip().lower() == "exit":
            break
        print("Agent:", agent_chat(user_msg))
