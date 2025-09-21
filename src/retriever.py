from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline

class Retreiver():
    def __init__(self, document_store):
        self.retreiver_pipeline = Pipeline()
        retriever = ChromaEmbeddingRetriever(document_store, top_k=5)
        embedder = SentenceTransformersTextEmbedder(
                    model="thenlper/gte-large"
                    )

        self.retreiver_pipeline.add_component("text_embedder", embedder)
        self.retreiver_pipeline.add_component("retriever", retriever)
        self.retreiver_pipeline.connect("text_embedder", "retriever")
    
    def get_documents(self, query):
        results = self.retreiver_pipeline.run({
            "text_embedder": {"text": query}
            })
        
        return results["retriever"]["documents"]