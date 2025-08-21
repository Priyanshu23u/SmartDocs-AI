from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores import Qdrant

from.config import settings


def build_retriever(vector_store: Qdrant) -> ContextualCompressionRetriever:
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.TOP_K}
    )
    reranker = FlashrankRerank(
        model=settings.RERANK_MODEL,
        top_n=settings.RERANK_TOP_N
    )
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker,
    )
    return compression_retriever
