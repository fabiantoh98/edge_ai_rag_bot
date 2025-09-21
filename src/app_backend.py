from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any

from src.vector_store import initialize_vector_db
from src.index_pipeline import HaystackIndexer, DocumentLoader
from src.rag import HaystackRAG

app = FastAPI()

# Initialize once at startup
document_store = initialize_vector_db()
indexer = HaystackIndexer(document_store=document_store)
rag_pipeline = HaystackRAG(document_store=document_store)
doc_loader = DocumentLoader()

class QueryRequest(BaseModel):
    query: str

class AddDocsRequest(BaseModel):
    documents: List[Dict[str, Any]]

@app.post("/query")
def query_rag(request: QueryRequest):
    answer = rag_pipeline.get_generative_answer(request.query)
    return {"answer": answer}

@app.post("/add-docs")
def add_documents(request: AddDocsRequest):
    # Expecting documents as list of dicts with 'content' and 'meta'
    docs = []
    for doc in request.documents:
        # You may want to validate/normalize here
        docs.append(doc)
    indexer.index(raw_docs=docs)
    return {"status": "success", "num_docs_added": len(docs)}

@app.post("/add-from-folder")
def add_from_folder():
    # Loads and indexes all docs from the folder using your loader
    docs = doc_loader.load_documents()
    indexer.index(raw_docs=docs)
    return {"status": "success", "num_docs_added": len(docs)}