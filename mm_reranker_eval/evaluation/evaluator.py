"""Evaluator for multimodal reranker models with parallel GPU support."""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from mm_reranker_eval.data.types import Query, Document, EvalSample
from mm_reranker_eval.evaluation.metrics import compute_metrics, aggregate_metrics
from mm_reranker_eval.reranker.factory import MMReranker

logger = logging.getLogger(__name__)


def _evaluate_single_query(
    query_idx: int,
    eval_sample: EvalSample,
    candidate_docs: List[Document],
    model_name: str,
    model_kwargs: Dict[str, Any],
    device: str,
    rank_kwargs: Dict[str, Any],
    metric_kwargs: Dict[str, Any]
) -> Tuple[int, Dict[str, float]]:
    """
    Evaluate a single query (worker function for parallel processing).
    
    Args:
        query_idx: Query index
        eval_sample: Evaluation sample with query and ground truth
        candidate_docs: List of candidate documents (including the ground truth)
        model_name: Model name for initialization
        model_kwargs: Model initialization kwargs
        device: Device to use
        rank_kwargs: Ranking kwargs
        metric_kwargs: Metric computation kwargs
        
    Returns:
        Tuple of (query_idx, metrics_dict)
    """
    # Initialize model in this process
    reranker = MMReranker(model_name, device=device, **model_kwargs)
    
    # Find ground truth index
    gt_idx = None
    for i, doc in enumerate(candidate_docs):
        if _is_same_document(doc, eval_sample.match):
            gt_idx = i
            break
    
    if gt_idx is None:
        logger.warning(f"Ground truth not found in candidates for query {query_idx}")
        return query_idx, {}
    
    # Rank documents
    rank_result = reranker.rank(eval_sample.query, candidate_docs, **rank_kwargs)
    
    # Compute metrics
    metrics = compute_metrics(
        ranked_indices=rank_result.ranked_indices,
        relevant_idx=gt_idx,
        **metric_kwargs
    )
    
    return query_idx, metrics


def _is_same_document(doc1: Document, doc2: Document) -> bool:
    """Check if two documents are the same."""
    return (
        doc1.text == doc2.text and
        doc1.image == doc2.image and
        doc1.video == doc2.video
    )


class Evaluator:
    """
    Evaluator for multimodal reranker models.
    
    Supports parallel evaluation across multiple GPUs and comprehensive metrics.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        num_gpus: Optional[int] = None,
        **model_kwargs
    ):
        """
        Initialize evaluator.
        
        Args:
            model_name: Name of the reranker model
            device: Device type ('cuda' or 'cpu')
            num_gpus: Number of GPUs to use (None = auto-detect from CUDA_VISIBLE_DEVICES)
            **model_kwargs: Additional model initialization arguments
        """
        self.model_name = model_name
        self.device_type = device
        self.model_kwargs = model_kwargs
        
        # Determine available GPUs
        if device == "cuda" and num_gpus is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                self.num_gpus = len(cuda_visible.split(","))
            else:
                import torch
                self.num_gpus = torch.cuda.device_count()
        elif device == "cuda":
            self.num_gpus = num_gpus
        else:
            self.num_gpus = 0
        
        logger.info(f"Initialized evaluator with {self.num_gpus} GPU(s)")
    
    def evaluate(
        self,
        eval_data_path: str,
        candidate_docs: List[Document],
        output_dir: str,
        recall_k: List[int] = [1, 3, 5, 7, 10],
        ndcg_k: Optional[List[int]] = None,
        max_queries: Optional[int] = None,
        rank_kwargs: Optional[Dict[str, Any]] = None,
        save_per_query: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate reranker on a dataset.
        
        Args:
            eval_data_path: Path to JSONL file with evaluation samples
            candidate_docs: List of candidate documents for reranking
            output_dir: Directory to save results
            recall_k: List of K values for Recall@K
            ndcg_k: List of K values for NDCG@K (optional)
            max_queries: Maximum number of queries to evaluate (None = all)
            rank_kwargs: Additional kwargs for rank() method
            save_per_query: Whether to save per-query results
            
        Returns:
            Dictionary with aggregated metrics and metadata
        """
        # Load evaluation data
        eval_samples = self._load_eval_data(eval_data_path, max_queries)
        logger.info(f"Loaded {len(eval_samples)} evaluation samples")
        
        # Prepare metric kwargs
        metric_kwargs = {
            "recall_k_values": recall_k,
            "ndcg_k_values": ndcg_k
        }
        
        rank_kwargs = rank_kwargs or {}
        
        # Evaluate
        if self.num_gpus > 1 and len(eval_samples) > 1:
            all_metrics = self._evaluate_parallel(
                eval_samples,
                candidate_docs,
                rank_kwargs,
                metric_kwargs
            )
        else:
            all_metrics = self._evaluate_sequential(
                eval_samples,
                candidate_docs,
                rank_kwargs,
                metric_kwargs
            )
        
        # Aggregate results
        aggregated_metrics = aggregate_metrics(all_metrics)
        
        # Prepare results
        results = {
            "model_name": self.model_name,
            "num_queries": len(eval_samples),
            "num_candidates": len(candidate_docs),
            "metrics": aggregated_metrics,
            "config": {
                "recall_k": recall_k,
                "ndcg_k": ndcg_k,
                "rank_kwargs": rank_kwargs,
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated results
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save per-query results if requested
        if save_per_query:
            per_query_results = [
                {
                    "query_idx": i,
                    "dataset": eval_samples[i].dataset,
                    "id": eval_samples[i].id,
                    "metrics": all_metrics[i]
                }
                for i in range(len(eval_samples))
            ]
            with open(output_path / "per_query_results.json", "w") as f:
                json.dump(per_query_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        self._print_results(aggregated_metrics)
        
        return results
    
    def _evaluate_sequential(
        self,
        eval_samples: List[EvalSample],
        candidate_docs: List[Document],
        rank_kwargs: Dict[str, Any],
        metric_kwargs: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Evaluate queries sequentially (single GPU or CPU)."""
        device = f"cuda:0" if self.device_type == "cuda" else "cpu"
        reranker = MMReranker(self.model_name, device=device, **self.model_kwargs)
        
        all_metrics = []
        
        for eval_sample in tqdm(eval_samples, desc="Evaluating"):
            # Find ground truth index
            gt_idx = None
            for i, doc in enumerate(candidate_docs):
                if _is_same_document(doc, eval_sample.match):
                    gt_idx = i
                    break
            
            if gt_idx is None:
                logger.warning(f"Ground truth not found for sample {eval_sample.id}")
                all_metrics.append({})
                continue
            
            # Rank
            rank_result = reranker.rank(eval_sample.query, candidate_docs, **rank_kwargs)
            
            # Compute metrics
            metrics = compute_metrics(
                ranked_indices=rank_result.ranked_indices,
                relevant_idx=gt_idx,
                **metric_kwargs
            )
            all_metrics.append(metrics)
        
        return all_metrics
    
    def _evaluate_parallel(
        self,
        eval_samples: List[EvalSample],
        candidate_docs: List[Document],
        rank_kwargs: Dict[str, Any],
        metric_kwargs: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Evaluate queries in parallel across multiple GPUs."""
        num_workers = min(self.num_gpus, len(eval_samples))
        
        # Prepare tasks
        tasks = []
        for i, eval_sample in enumerate(eval_samples):
            gpu_id = i % num_workers
            device = f"cuda:{gpu_id}"
            
            tasks.append((
                i,
                eval_sample,
                candidate_docs,
                self.model_name,
                self.model_kwargs,
                device,
                rank_kwargs,
                metric_kwargs
            ))
        
        # Execute in parallel
        all_metrics = [None] * len(eval_samples)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_evaluate_single_query, *task): task[0]
                for task in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                query_idx, metrics = future.result()
                all_metrics[query_idx] = metrics
        
        return all_metrics
    
    def _load_eval_data(
        self,
        data_path: str,
        max_samples: Optional[int] = None
    ) -> List[EvalSample]:
        """Load evaluation data from JSONL file."""
        samples = []
        
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                data = json.loads(line)
                sample = EvalSample.from_json(data)
                samples.append(sample)
        
        return samples
    
    def _print_results(self, metrics: Dict[str, float]) -> None:
        """Print results in a formatted way."""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:20s}: {value:.4f}")
        
        print("=" * 50 + "\n")

