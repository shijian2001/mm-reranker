"""Evaluation metrics for reranking tasks."""

from typing import List, Dict, Any
import numpy as np


def recall_at_k(ranked_indices: List[int], relevant_idx: int, k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        ranked_indices: List of document indices ranked by score
        relevant_idx: Index of the relevant/ground truth document
        k: Top-k to consider
        
    Returns:
        1.0 if relevant document is in top-k, 0.0 otherwise
    """
    if k <= 0:
        return 0.0
    
    top_k = ranked_indices[:k]
    return 1.0 if relevant_idx in top_k else 0.0


def mean_reciprocal_rank(ranked_indices: List[int], relevant_idx: int) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a single query.
    
    Args:
        ranked_indices: List of document indices ranked by score
        relevant_idx: Index of the relevant/ground truth document
        
    Returns:
        Reciprocal rank of the relevant document
    """
    try:
        rank = ranked_indices.index(relevant_idx) + 1  # 1-indexed
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(ranked_indices: List[int], relevant_idx: int, k: int) -> float:
    """
    Calculate NDCG@K for binary relevance (single relevant document).
    
    Args:
        ranked_indices: List of document indices ranked by score
        relevant_idx: Index of the relevant/ground truth document
        k: Top-k to consider
        
    Returns:
        NDCG@K score
    """
    if k <= 0:
        return 0.0
    
    top_k = ranked_indices[:k]
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(top_k):
        if idx == relevant_idx:
            # relevance = 1 for relevant doc, 0 otherwise
            # DCG += (2^rel - 1) / log2(i + 2)
            dcg = 1.0 / np.log2(i + 2)
            break
    
    # IDCG (ideal DCG) - relevant doc at position 1
    idcg = 1.0 / np.log2(2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    ranked_indices: List[int],
    relevant_idx: int,
    recall_k_values: List[int] = [1, 3, 5, 7, 10],
    ndcg_k_values: List[int] = None
) -> Dict[str, float]:
    """
    Compute multiple metrics for a ranking result.
    
    Args:
        ranked_indices: List of document indices ranked by score
        relevant_idx: Index of the relevant/ground truth document
        recall_k_values: List of K values for Recall@K
        ndcg_k_values: List of K values for NDCG@K (optional)
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Recall@K
    for k in recall_k_values:
        metrics[f"recall@{k}"] = recall_at_k(ranked_indices, relevant_idx, k)
    
    # MRR
    metrics["mrr"] = mean_reciprocal_rank(ranked_indices, relevant_idx)
    
    # NDCG@K (if requested)
    if ndcg_k_values:
        for k in ndcg_k_values:
            metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_indices, relevant_idx, k)
    
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries.
    
    Args:
        all_metrics: List of metric dicts from individual queries
        
    Returns:
        Dictionary with averaged metrics
    """
    if not all_metrics:
        return {}
    
    # Get all metric names
    metric_names = all_metrics[0].keys()
    
    # Average each metric
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        aggregated[name] = np.mean(values)
    
    return aggregated

