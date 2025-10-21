"""Evaluator for multimodal reranker models with parallel GPU support."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from mm_reranker_eval.data.types import Query, Document, EvalSample
from mm_reranker_eval.evaluation.metrics import compute_metrics, aggregate_metrics
from mm_reranker_eval.reranker.factory import MMReranker
from mm_reranker_eval.reranker.base import BaseReranker

logger = logging.getLogger(__name__)


def _evaluate_single_query(
    reranker: BaseReranker,
    eval_sample: EvalSample,
    candidate_docs: List[Document],
    rank_kwargs: Dict[str, Any],
    metric_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single query using a pre-initialized reranker.
    
    Args:
        reranker: Pre-initialized reranker instance
        eval_sample: Evaluation sample with query and ground truth
        candidate_docs: List of candidate documents (including ground truth)
        rank_kwargs: Ranking kwargs
        metric_kwargs: Metric computation kwargs
        
    Returns:
        Result dict with metrics and ranking info
    """
    # Find ground truth index
    gt_idx = None
    for i, doc in enumerate(candidate_docs):
        if _is_same_document(doc, eval_sample.match):
            gt_idx = i
            break
    
    if gt_idx is None:
        logger.warning(f"Ground truth not found for sample {eval_sample.id}")
        return {"metrics": {}}
    
    # Rank documents
    rank_result = reranker.rank(eval_sample.query, candidate_docs, **rank_kwargs)
    
    # Compute metrics
    metrics = compute_metrics(
        ranked_indices=rank_result.ranked_indices,
        relevant_idx=gt_idx,
        **metric_kwargs
    )
    
    return {
        "metrics": metrics,
        "ranked_indices": rank_result.ranked_indices,
        "scores": rank_result.scores or [],
        "gt_idx": gt_idx
    }




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
        
        # Set number of GPUs to use
        if device == "cuda":
            self.num_gpus = num_gpus if num_gpus is not None else 8
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
        save_per_query: bool = False,
        base_dir: Optional[str] = None
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
        eval_samples = self._load_eval_data(eval_data_path, max_queries, base_dir)
        logger.info(f"Loaded {len(eval_samples)} evaluation samples")
        
        # Prepare metric kwargs
        metric_kwargs = {
            "recall_k_values": recall_k,
            "ndcg_k_values": ndcg_k
        }
        
        rank_kwargs = rank_kwargs or {}
        
        # Evaluate
        if self.num_gpus > 1 and len(eval_samples) > 1:
            all_results = self._evaluate_parallel(
                eval_samples,
                candidate_docs,
                rank_kwargs,
                metric_kwargs
            )
        else:
            all_results = self._evaluate_sequential(
                eval_samples,
                candidate_docs,
                rank_kwargs,
                metric_kwargs
            )
        
        # Aggregate metrics (filter out None results)
        all_metrics = [r.get("metrics", {}) for r in all_results if r is not None]
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
            per_query_results = []
            for i in range(len(eval_samples)):
                sample = eval_samples[i]
                result = all_results[i]
                
                # Handle None results (failed queries)
                if result is None:
                    query_result = {
                        "query_idx": i,
                        "dataset": sample.dataset,
                        "id": sample.id,
                        "query": {
                            "text": sample.query.text,
                            "image": sample.query.image,
                            "video": sample.query.video
                        },
                        "match": {
                            "text": sample.match.text,
                            "image": sample.match.image,
                            "video": sample.match.video
                        },
                        "metrics": {},
                        "ranked_indices": [],
                        "scores": [],
                        "gt_idx": None,
                        "failed": True
                    }
                else:
                    query_result = {
                        "query_idx": i,
                        "dataset": sample.dataset,
                        "id": sample.id,
                        "query": {
                            "text": sample.query.text,
                            "image": sample.query.image,
                            "video": sample.query.video
                        },
                        "match": {
                            "text": sample.match.text,
                            "image": sample.match.image,
                            "video": sample.match.video
                        },
                        "metrics": result.get("metrics", {}),
                        "ranked_indices": result.get("ranked_indices", []),
                        "scores": result.get("scores", []),
                        "gt_idx": result.get("gt_idx")
                    }
                per_query_results.append(query_result)
            
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
    ) -> List[Dict[str, Any]]:
        """Evaluate queries sequentially (single GPU or CPU)."""
        from tqdm import tqdm
        
        device = f"cuda:0" if self.device_type == "cuda" else "cpu"
        reranker = MMReranker(self.model_name, device=device, **self.model_kwargs)
        
        all_results = []
        for eval_sample in tqdm(eval_samples, desc="Evaluating"):
            result = _evaluate_single_query(
                reranker, eval_sample, candidate_docs, rank_kwargs, metric_kwargs
            )
            all_results.append(result)
        
        return all_results
    
    def _evaluate_parallel(
        self,
        eval_samples: List[EvalSample],
        candidate_docs: List[Document],
        rank_kwargs: Dict[str, Any],
        metric_kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate queries in parallel across multiple GPUs using Accelerate.
        
        Strategy:
        1. Use Accelerate to automatically manage multi-GPU setup
        2. Each process loads its own model instance
        3. Data is automatically split across processes
        4. Results are gathered automatically
        
        This approach is elegant, minimal, and leverages Accelerate's robust infrastructure.
        """
        try:
            from accelerate import Accelerator
            from accelerate.utils import gather_object
        except ImportError:
            logger.error(
                "Accelerate is required for multi-GPU evaluation. "
                "Install it with: pip install accelerate"
            )
            # Fallback to sequential
            return self._evaluate_sequential(eval_samples, candidate_docs, rank_kwargs, metric_kwargs)
        
        # Check if we're already in an Accelerate context
        try:
            accelerator = Accelerator()
        except Exception as e:
            logger.warning(f"Failed to initialize Accelerator: {e}. Falling back to sequential evaluation.")
            return self._evaluate_sequential(eval_samples, candidate_docs, rank_kwargs, metric_kwargs)
        
        # Split data across processes
        with accelerator.split_between_processes(eval_samples, apply_padding=False) as process_samples:
            # Each process initializes its own model
            # Accelerate automatically assigns the correct GPU
            device = accelerator.device
            
            if accelerator.is_main_process:
                logger.info(
                    f"Starting parallel evaluation across {accelerator.num_processes} GPU(s), "
                    f"{len(process_samples)} queries per process"
                )
            
            # Load model on this process's assigned device
            reranker = MMReranker(self.model_name, device=str(device), **self.model_kwargs)
            
            # Process assigned queries with progress bar
            # Each GPU gets its own progress bar at a different position
            from tqdm import tqdm
            process_results = []
            
            # Create progress bar for this GPU
            # position parameter ensures each GPU's progress bar doesn't overlap
            pbar = tqdm(
                process_samples,
                desc=f"GPU {accelerator.process_index}",
                position=accelerator.process_index,
                leave=True
            )
            
            for eval_sample in pbar:
                result = _evaluate_single_query(
                    reranker, eval_sample, candidate_docs, rank_kwargs, metric_kwargs
                )
                process_results.append(result)
        
        # Gather results from all processes
        all_results = gather_object(process_results)
        
        if accelerator.is_main_process:
            logger.info(f"Parallel evaluation completed, processed {len(all_results)} queries total")
        
        return all_results
    
    def _load_eval_data(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        base_dir: Optional[str] = None
    ) -> List[EvalSample]:
        """Load evaluation data from JSONL file."""
        samples = []
        
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                data = json.loads(line)
                sample = EvalSample.from_json(data, base_dir=base_dir)
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

