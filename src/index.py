from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document

from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    # Ensure useful metadata on each chunk
    for c in chunks:
        c.metadata.setdefault("source", c.metadata.get("source", "uploaded.pdf"))
        c.metadata.setdefault("page_number", c.metadata.get("page_number"))

    return [c for c in chunks if c.page_content.strip()]


def build_embeddings():
    return FastEmbedEmbeddings(model_name=settings.EMBEDDING_MODEL)


def index_into_qdrant(chunks: List[Document]) -> Qdrant:
    embeddings = build_embeddings()
    logger.info(
        f"Indexing {len(chunks)} chunks into Qdrant collection '{settings.QDRANT_COLLECTION}'"
    )
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.QDRANT_COLLECTION,
        url=settings.QDRANT_URL if settings.QDRANT_URL else None,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        path=(None if settings.QDRANT_URL else settings.QDRANT_LOCAL_PATH),
        prefer_grpc=settings.PREFER_GRPC,
    )
    return vector_store
