from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

@tool
def rag_tool_func(query) -> dict:
    """use the RAG tool to retrive the infromtion from the pdf document and give the result for the query related to mcp"""
    doc_loation = PyPDFLoader("Resources/MCP_Server.pdf")
    docs = doc_loation.load()

    doc_split = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = doc_split.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = QdrantVectorStore.from_documents(documents=chunks, embedding=embeddings, path = "./qdrant_db", collection_name = "mcp_embeddings")
    retriver_helper = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 3})

    result = retriver_helper.invoke(query)

    return {
        "query" : query,
        "result" : [docs.page_content for docs in result],
        "metadata" : [docs.metadata for docs in result]
    }

