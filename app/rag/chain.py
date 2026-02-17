from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from app.prompts.rag_prompt import RAG_PROMPT


def build_rag_chain(db: Chroma, llm: Ollama) -> object:
    """
    Build the RAG retrieval chain with hybrid search (vector + BM25) and reranking.
    """
    # Vector retriever
    vector_retriever = db.as_retriever(
        search_kwargs={"k": 20})

    total_docs = db._collection.count()

    if total_docs == 0:
        retriever = vector_retriever
    else:
        # Load all documents once for BM25
        data = db.get(include=["documents", "metadatas", "embeddings"])
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(data["documents"], data["metadatas"])
        ]

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 20

        # Lean on semantic but keep keyword boost
        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

        # Rerank the combined top-k
        reranker = FlashrankRerank(top_n=6)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_retriever,
        )

    # control context size via reranker
    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=RAG_PROMPT)

    return create_retrieval_chain(retriever, stuff_chain)
