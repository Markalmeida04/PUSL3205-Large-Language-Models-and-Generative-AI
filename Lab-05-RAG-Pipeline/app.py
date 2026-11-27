import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents or upload your own PDFs.")

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_new_tokens=180,
        return_full_text=False
    )
    return embeddings, generator

embeddings, generator = load_models()

# Sidebar for dynamic multi-PDF upload
with st.sidebar:
    st.header("📄 Upload University PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Process Uploaded PDFs"):
        with st.spinner("Indexing uploaded PDFs..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
                os.remove(tmp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(all_docs)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks, embedding=embeddings
            )
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s)!")

# Load pre-ingested ChromaDB by default
if "vectorstore" not in st.session_state:
    if os.path.exists("chroma_db"):
        st.session_state.vectorstore = Chroma(
            persist_directory="chroma_db", embedding_function=embeddings
        )
    else:
        st.info("Please upload one or more PDF files in the sidebar to begin.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask your university question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    messages = [
        {"role": "system", "content": "You are a helpful University AI Assistant. Answer strictly using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = generator(messages)
            answer = result[0]["generated_text"].strip()
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
