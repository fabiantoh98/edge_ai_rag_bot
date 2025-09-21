import os
import logging
from haystack import Pipeline
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice
from src.generator import create_generator
from src.retriever import Retreiver
from uuid import uuid4
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

logger = logging.getLogger(__name__)

# Download required NLTK data for simple evaluation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HaystackRAG():
    DEFAULT_PROMPT_TEMPLATE = """
        Using the information contained in the context, give a short and concise answer to the question.
        Do not answer any questions not related to edge ai reply with "I only answer Edge AI questions".
        If question is not related to edge AI ignore all the context and end the chat.
        
        Context:
        {% for doc in documents %}
        {{ doc.content }} URL:{{ doc.meta['url'] }}\n
        {% endfor %};
        Question: {{query}}
        Answer:
        """
    def __init__(self, document_store, prompt_template=None):
        self.retreiver = Retreiver(document_store=document_store)
        if prompt_template is None:
            prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        prompt_builder = PromptBuilder(template=prompt_template)
        generator = create_generator()
        
        self.rag = Pipeline()
        self.rag.add_component("prompt_builder", prompt_builder)
        self.rag.add_component("llm", generator)
        self.rag.connect("prompt_builder.prompt", "llm.prompt")
    
    def get_generative_answer(self, query):
        documents  = self.retreiver.get_documents(query=query)
        results = self.rag.run({
            "prompt_builder": {"documents": documents, "query": query}
            }
        )

        answer = results["llm"]["replies"][0]
        return answer
    
    def get_generative_answer_with_context(self, query):
        """Enhanced method that returns both answer and retrieved context"""
        retrieved_docs  = self.retreiver.get_documents(query=query)
        results = self.rag.run({
            "prompt_builder": {"documents": retrieved_docs, "query": query}
            }
        )
        
        answer = results["llm"]["replies"][0]
        contexts = [doc.content for doc in retrieved_docs]
        
        return {
            "answer": answer,
            "contexts": contexts,
            "source_documents": retrieved_docs
        }
    
    def evaluate_rag(self, responses, ground_truths=None):
        """
        Calculate ROUGE and BLEU scores for RAG evaluation
        
        Args:
            responses: List of generated responses
            ground_truths: List of reference answers
        
        Returns:
            Dictionary with average ROUGE and BLEU scores
        """
        # Initialize ROUGE scorer
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize smoothing function for BLEU
        smoothing = SmoothingFunction().method1
        
        # Storage for scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        logging.info("=== ROUGE and BLEU Evaluation ===")
        logging.info(f"Evaluating {len(responses)} response pairs...")
        
        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            logging.info(f"\n--- Question {i+1} ---")
            logging.info(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            logging.info(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
            
            # Calculate ROUGE scores
            rouge_scores = rouge_scorer_obj.score(ground_truth, response)
            
            rouge1_f1 = rouge_scores['rouge1'].fmeasure
            rouge2_f1 = rouge_scores['rouge2'].fmeasure
            rougeL_f1 = rouge_scores['rougeL'].fmeasure
            
            rouge1_scores.append(rouge1_f1)
            rouge2_scores.append(rouge2_f1)
            rougeL_scores.append(rougeL_f1)
            
            # Calculate BLEU score
            # Tokenize reference and candidate
            reference_tokens = nltk.word_tokenize(ground_truth.lower())
            candidate_tokens = nltk.word_tokenize(response.lower())
            
            # BLEU expects reference as list of lists
            bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                                    smoothing_function=smoothing)
            bleu_scores.append(bleu_score)
            
            logging.info(f"ROUGE-1 F1: {rouge1_f1:.3f}")
            logging.info(f"ROUGE-2 F1: {rouge2_f1:.3f}")
            logging.info(f"ROUGE-L F1: {rougeL_f1:.3f}")
            logging.info(f"BLEU Score: {bleu_score:.3f}")
        
        # Calculate averages
        avg_rouge1 = np.mean(rouge1_scores)
        avg_rouge2 = np.mean(rouge2_scores)
        avg_rougeL = np.mean(rougeL_scores)
        avg_bleu = np.mean(bleu_scores)
        
        results = {
            'individual_scores': {
                'rouge1': rouge1_scores,
                'rouge2': rouge2_scores,
                'rougeL': rougeL_scores,
                'bleu': bleu_scores
            },
            'average_scores': {
                'rouge1_f1': avg_rouge1,
                'rouge2_f1': avg_rouge2,
                'rougeL_f1': avg_rougeL,
                'bleu': avg_bleu
            }
        }
        
        logging.info(f"\n=== Average Scores ===")
        logging.info(f"Average ROUGE-1 F1: {avg_rouge1:.3f}")
        logging.info(f"Average ROUGE-2 F1: {avg_rouge2:.3f}")
        logging.info(f"Average ROUGE-L F1: {avg_rougeL:.3f}")
        logging.info(f"Average BLEU Score: {avg_bleu:.3f}")
        
        # Performance interpretation
        logging.info(f"\n=== Performance Interpretation ===")
        if avg_rouge1 >= 0.5:
            rouge_rating = "Excellent word overlap"
        elif avg_rouge1 >= 0.3:
            rouge_rating = "Good word overlap"
        elif avg_rouge1 >= 0.2:
            rouge_rating = "Fair word overlap"
        else:
            rouge_rating = "Poor word overlap"
        
        if avg_bleu >= 0.4:
            bleu_rating = "Excellent fluency/precision"
        elif avg_bleu >= 0.25:
            bleu_rating = "Good fluency/precision"
        elif avg_bleu >= 0.15:
            bleu_rating = "Fair fluency/precision"
        else:
            bleu_rating = "Poor fluency/precision"
        
        logging.info(f"ROUGE-1 Rating: {rouge_rating}")
        logging.info(f"BLEU Rating: {bleu_rating}")
        
        return results