from langchain.tools import tool
from vector_stores.vector_helper import VectorHelper
from schemas.rag_tool_parameters import PatientSearchInput
import constants

vectorstore = VectorHelper()

@tool("patient_document_search", description="Retrieve patient information regarding the all medical history",args_schema=PatientSearchInput, return_direct=True)
def rag_patient_retrieval(query: str, file_path: str) -> str:
    """
    Retrieve patient information from PGVector
    """
    print("call patient_document_search...")
    similarity_threshold = 0.8
    all_chunks = []

    print(f"Retrieval Query: {query}")
    print(f"file_path: {file_path}")

    docs = vectorstore.search_with_cosine_similarity(query, k=constants.TOP_K, filter={"source": file_path})
    print(f"Found {len(docs)} documents for query: {query}")
        
    for doc, score in docs:
        if score is not None and score >= similarity_threshold:
            all_chunks.append(doc)

    final_context = "\n\n".join([chunk.page_content for chunk in all_chunks])
    return final_context