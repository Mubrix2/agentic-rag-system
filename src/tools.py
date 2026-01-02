from langchain.tools import tool
from retriever import get_retriever
from vectorstore import build_vectorstore
from embeddings import get_embeddings
from chunking import chunk_documents
from loaders import load_documents


@tool
def search_documents(query: str) -> str:
    """
    Searches internal documents for relevant information.
    Use this tool when you need factual information from documents.
    """
    docs = load_documents("data/documents/RAG_Hands_On_Practical_Document.pdf")

    if not docs:
        return "No documents available."

    chunks = chunk_documents(docs)
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    results = retriever.get_relevant_documents(query)

    if not results:
        return "No relevant information found."

    return "\n".join([doc.page_content for doc in results])
