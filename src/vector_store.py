import os
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

def initialize_vector_db(chroma_dir = "chromadb"):
    os.makedirs(chroma_dir, exist_ok=True)
    document_store = ChromaDocumentStore(
        persist_path=chroma_dir,
        distance_function="cosine"
    )
    return document_store
