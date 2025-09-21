import logging
import hydra
import os
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from time import perf_counter

from src.vector_store import initialize_vector_db
from src.index_pipeline import HaystackIndexer, DocumentLoader
from src.rag import HaystackRAG
from src.data_loader import load_qa_from_json
from src.utils import setup_logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.1")
def main(cfg):
    original_dir = hydra.utils.get_original_cwd()
    setup_logging(
        logging_config_path=os.path.join(
            original_dir, "conf", "logging.yaml"
        ),
        log_dir=os.path.join(original_dir, cfg.log_dir),
    )
    # Setup or load chroma db
    chroma_dir = os.path.join(original_dir, cfg.chromedb_dir)
    document_store = initialize_vector_db(chroma_dir = chroma_dir)
    logging.info("Successfully initialize chroma db")
    
    if cfg.indexing:
        corpus_dir = os.path.join(original_dir, cfg.corpus_dir)
        raw_docs = DocumentLoader(data_dir=corpus_dir).load_documents()
        indexer = HaystackIndexer(document_store=document_store, model_name=cfg.rag_embedding_model_name)
        logging.info("Starting to index, duplicate documents would be updated")
        indexer.index(raw_docs=raw_docs)
        logging.info("Completed indexing of documents.")
    
    rag_pipeline = HaystackRAG(document_store=document_store)
    
    # Load test questions
    print("test")
    logging.info("Begin evaluation, loading question and answers")
    test_file_path = os.path.join(original_dir, cfg.test_file_path)
    questions, ground_truths = load_qa_from_json(test_file_path)
    responses = []
    # contexts =[]
    logging.info("Generating answers for respective questions")
    for question in questions:
        res = rag_pipeline.get_generative_answer_with_context(question)
        responses.append(res['answer'])
        # contexts.append(res["contexts"])
        
    rag_results = rag_pipeline.evaluate_rag(responses=responses, ground_truths=ground_truths)
    logging.info(f"RAG Results: {rag_results}")

if __name__ == "__main__":
    start_time = perf_counter()
    main()
    logging.info(f"Total Time to run pipeline {(perf_counter()-start_time)/60:.3g} mins")