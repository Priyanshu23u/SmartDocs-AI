import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

from src.config import settings
from src.utils import validate_required_env, get_logger
from src.ingestion import parse_pdf, to_langchain_documents
from src.index import split_documents, index_into_qdrant
from src.retriever import build_retriever
from src.qa import build_qa_chain

logger = get_logger(__name__)
load_dotenv()

st.set_page_config(page_title="AI Q&A over PDF", page_icon="📄")
st.title("📄 AI Q&A over Uploaded PDF")

# Sidebar
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
use_llamaparse = st.sidebar.checkbox(
    "Use LlamaParse (if API key provided)", value=False
)

# Env check
missing = validate_required_env(["GROQ_API_KEY"])
if missing:
    st.error(f"Missing required environment variables: {', '.join(missing)}")
    st.stop()

if uploaded_file:
    try:
        with st.spinner("📄 Parsing PDF..."):
            # Save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            parsed = parse_pdf(tmp_path, use_llamaparse=use_llamaparse)
            docs = to_langchain_documents(parsed)
            if not docs:
                st.error("❌ No text extracted from PDF.")
                st.stop()

        with st.spinner("🔪 Chunking and embedding..."):
            chunks = split_documents(docs)
            if not chunks:
                st.error("❌ No valid text chunks after splitting.")
                st.stop()

            vector_store = index_into_qdrant(chunks)
            retriever = build_retriever(vector_store)
            qa_chain = build_qa_chain(retriever)

        st.success("✅ Document processed! Ask questions below.")
        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("🤖 Generating answer..."):
                result = qa_chain.invoke({"query": query})
                answer = result.get("result", "").strip()
                sources: List = result.get("source_documents", [])

            if answer:
                st.markdown("### Answer")
                st.write(answer)

            if sources:
                st.markdown("### Sources")
                for i, d in enumerate(sources, start=1):
                    meta = d.metadata or {}
                    src = meta.get("source", "uploaded.pdf")
                    page = meta.get("page_number", "?")
                    snippet = (
                        d.page_content[:400].strip().replace("\n", " ")
                        if d.page_content
                        else ""
                    )
                    st.write(f"{i}. {src} — page {page}")
                    if snippet:
                        st.caption(snippet)

    except Exception as e:
        logger.exception("Processing failed")
        st.error(f"❌ An error occurred: {e}")

else:
    st.info("⬅️ Upload a PDF from the sidebar to get started.")
