"""
MM Reranker Evaluation Package

A unified package for evaluating multimodal reranking models across various retrieval tasks.
Supports text, image, and mixed modality queries and documents.
"""

from mm_reranker_eval.reranker.factory import MMReranker
from mm_reranker_eval.evaluation.evaluator import Evaluator
from mm_reranker_eval.data.types import Query, Document, EvalSample

__version__ = "0.1.0"
__all__ = ["MMReranker", "Evaluator", "Query", "Document", "EvalSample"]

