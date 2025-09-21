import os
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from src.data_loader import AudioVideoExtractor, PPTXExtractor, CSVExtractor, ImageExtractor

class HaystackIndexer:
    def __init__(self, document_store, model_name="thenlper/gte-large"):
        self.pipeline = Pipeline()
        self.pipeline.add_component("cleaner", DocumentCleaner())
        self.pipeline.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=2))
        self.pipeline.add_component("doc_embedder", SentenceTransformersDocumentEmbedder(model=model_name, meta_fields_to_embed=["title"]))
        self.pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
        self.pipeline.connect("cleaner", "splitter")
        self.pipeline.connect("splitter", "doc_embedder")
        self.pipeline.connect("doc_embedder", "writer")

    def index(self, raw_docs):
        return self.pipeline.run({"cleaner": {"documents": raw_docs}})

class DocumentLoader:
    def __init__(self, data_dir="corpus/edge_ai"):
        self.data_dir = data_dir
        self.av_extractor = AudioVideoExtractor()
        self.csv_extractor = CSVExtractor()
        self.pptx_extractor = PPTXExtractor()
        self.img_extractor = ImageExtractor()
        self.pdf_converter = PyPDFToDocument()
        self.text_converter = TextFileToDocument()

    def load_documents(self):
        raw_docs = []
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            file_ext = file_name.split(".")[-1].lower()
            if file_ext == "pdf":
                docs = self.pdf_converter.run(sources=[file_path])["documents"]
                for doc in docs:
                    doc.meta["file_type"] = "pdf"
                    doc.meta["file_path"] = file_path
                    if "page" not in doc.meta and "page_number" in doc.meta:
                        doc.meta["page"] = doc.meta["page_number"]
                    raw_docs.append(doc)
            elif file_ext == "txt":
                docs = self.text_converter.run(sources=[file_path])["documents"]
                for doc in docs:
                    doc.meta["file_type"] = "txt"
                    doc.meta["file_path"] = file_path
                    doc.meta["page"] = 1
                    raw_docs.append(doc)
            elif file_ext in ["mp3", "mp4"]:
                texts = self.av_extractor.extract(file_path)
                for i, text in enumerate(texts):
                    raw_docs.append(Document(content=text, meta={
                        "file_type": file_ext,
                        "file_path": file_path,
                        "page": 0
                    }))
            elif file_ext in ["jpg", "png"]:
                texts = self.img_extractor.extract(file_path=file_path)
                raw_docs.append(Document(content=texts, meta={
                    "file_type": file_ext,
                    "file_path": file_path,
                    "page": 0
                }))
            elif file_ext == "pptx":
                texts = self.pptx_extractor.extract(file_path)
                for i, text in enumerate(texts):
                    raw_docs.append(Document(content=text, meta={
                        "file_type": file_ext,
                        "file_path": file_path,
                        "page": i+1
                    }))
            elif file_ext == "csv":
                texts = self.csv_extractor.extract(file_path)
                for i, text in enumerate(texts):
                    raw_docs.append(Document(content=text, meta={
                        "file_type": file_ext,
                        "file_path": file_path,
                        "page": i+1
                    }))
        return raw_docs