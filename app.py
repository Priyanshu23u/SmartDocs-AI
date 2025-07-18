import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import fitz  # PyMuPDF

# Load environment variables from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Streamlit app settings
st.set_page_config(page_title="📄 AI Q&A over PDF", page_icon="📄")
st.title("📄 AI Q&A over Uploaded PDF")

st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to analyze", type=["pdf"])

if uploaded_file:
    # Check if API keys are set
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it and restart the app.")
        st.stop()
    if not qdrant_url or not qdrant_api_key:
        st.error("⚠️ QDRANT_URL or QDRANT_API_KEY not found in .env file. Please add them and restart the app.")
        st.stop()

    with st.spinner("📄 Parsing and indexing the PDF..."):
        try:
            # Save uploaded PDF temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Parse text from PDF using PyMuPDF
            with fitz.open(temp_path) as pdf:
                full_text = ""
                for page in pdf:
                    text = page.get_text()
                    if text:
                        full_text += text + "\n"

            if not full_text.strip():
                st.error("❌ No text found in the PDF. Maybe it's scanned or image-only.")
                st.stop()

            # Wrap into LangChain Document object
            documents = [Document(page_content=full_text)]

            # Split into smaller chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(documents)

            # Filter out empty chunks
            docs = [doc for doc in docs if doc.page_content.strip()]
            if not docs:
                st.error("❌ Document split failed: No valid text chunks to embed.")
                st.stop()

            # Create embeddings & store in Qdrant Cloud
            embeddings = FastEmbedEmbeddings()
            vector_store = Qdrant.from_documents(
                docs,
                embeddings,
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=False  # use HTTP for simplicity
            )

            # Create retriever with contextual compression (reranking)
            retriever = ContextualCompressionRetriever(
                base_compressor=FlashrankRerank(top_n=4),
                base_retriever=vector_store.as_retriever()
            )

            # Create QA chain using Groq LLM
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        except Exception as e:
            st.error(f"❌ An error occurred while processing: {e}")
            st.stop()

    st.success("✅ Document processed! You can now ask questions below:")

    # User input for question
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("🤖 Generating answer..."):
            answer = qa_chain({"query": query})
            st.markdown(f"**Answer:** {answer['result']}")

else:
    st.info("⬅️ Please upload a PDF from the sidebar to get started.")
