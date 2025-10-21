"""Evaluator for multimodal reranker models with parallel GPU support."""

import os
import json
import logging
import tempfile
import pickle
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

from mm_reranker_eval.data.types import Query, Document, EvalSample
from mm_reranker_eval.evaluation.metrics import compute_metrics, aggregate_metrics
from mm_reranker_eval.reranker.factory import MMReranker
from mm_reranker_eval.reranker.base import BaseReranker

logger = logging.getLogger(__name__)

# Set spawn method for proper multiprocessing isolation
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


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


def _gpu_worker(
    gpu_id: int,
    query_batch_file: str,
    candidate_docs_file: str,
    model_name: str,
    model_kwargs: Dict[str, Any],
    rank_kwargs: Dict[str, Any],
    metric_kwargs: Dict[str, Any],
    output_file: str
):
    """
    Worker function that runs on a single GPU.
    
    This function loads a model on the specified GPU, processes its assigned
    batch of queries, and saves results to a file.
    
    Args:
        gpu_id: GPU device ID (0, 1, 2, ...)
        query_batch_file: Path to pickle file with query batch
        candidate_docs_file: Path to pickle file with candidate documents
        model_name: Model name for initialization
        model_kwargs: Model initialization kwargs
        rank_kwargs: Ranking kwargs
        metric_kwargs: Metric computation kwargs
        output_file: Path to save results
    """
    import time
    
    start_time = time.time()
    print(f"[GPU {gpu_id}] ========== Worker started at {time.strftime('%H:%M:%S')} ==========", flush=True)
    
    # Load data from shared files
    t0 = time.time()
    with open(query_batch_file, 'rb') as f:
        query_batch = pickle.load(f)
    
    with open(candidate_docs_file, 'rb') as f:
        candidate_docs = pickle.load(f)
    
    load_time = time.time() - t0
    print(f"[GPU {gpu_id}] Loaded {len(query_batch)} queries and {len(candidate_docs)} candidates in {load_time:.1f}s", flush=True)
    
    # Load model directly on the specified GPU
    t1 = time.time()
    device = f"cuda:{gpu_id}"
    reranker = MMReranker(model_name, device=device, **model_kwargs)
    model_time = time.time() - t1
    print(f"[GPU {gpu_id}] Model loaded in {model_time:.1f}s, starting evaluation", flush=True)
    
    # Process all queries in this batch with progress tracking
    results = []
    total = len(query_batch)
    eval_start = time.time()
    
    for i, (query_idx, eval_sample) in enumerate(query_batch):
        result = _evaluate_single_query(
            reranker, eval_sample, candidate_docs, rank_kwargs, metric_kwargs
        )
        results.append((query_idx, result))
        
        # Print progress every 10 queries or at first query
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - eval_start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"[GPU {gpu_id}] Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) | "
                  f"Speed: {speed:.2f} q/s | ETA: {eta/60:.1f}min", flush=True)
    
    # Save results to file
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    total_time = time.time() - start_time
    eval_time = time.time() - eval_start
    avg_speed = total / eval_time if eval_time > 0 else 0
    print(f"[GPU {gpu_id}] ========== Completed in {total_time/60:.1f}min "
          f"(eval: {eval_time/60:.1f}min, {avg_speed:.2f} q/s) ==========", flush=True)


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
        
        # Aggregate metrics
        all_metrics = [r.get("metrics", {}) for r in all_results]
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
        Evaluate queries in parallel across multiple GPUs.
        
        Strategy:
        1. Split data evenly across GPUs
        2. Save batches and candidate_docs to temporary files
        3. Spawn one process per GPU, each loads model on its assigned GPU
        4. Each process saves results to its own file
        5. Merge results and cleanup
        
        This approach ensures one model per GPU with zero inter-process communication.
        """
        num_workers = min(self.num_gpus, len(eval_samples))
        
        # Create temporary directory for all intermediate files
        temp_dir = tempfile.mkdtemp(prefix="mm_reranker_eval_")
        
        try:
            # Save candidate documents to shared file (read by all workers)
            candidate_docs_file = Path(temp_dir) / "candidates.pkl"
            with open(candidate_docs_file, 'wb') as f:
                pickle.dump(candidate_docs, f)
            
            # Split queries evenly across GPUs
            batch_size = (len(eval_samples) + num_workers - 1) // num_workers
            processes = []
            
            logger.info(f"Starting {num_workers} GPU workers, {batch_size} queries per GPU")
            
            for gpu_id in range(num_workers):
                start_idx = gpu_id * batch_size
                end_idx = min(start_idx + batch_size, len(eval_samples))
                
                if start_idx >= len(eval_samples):
                    break
                
                # Create query batch with (query_idx, eval_sample) tuples
                query_batch = [
                    (i, eval_samples[i]) 
                    for i in range(start_idx, end_idx)
                ]
                
                # Save batch to file
                batch_file = Path(temp_dir) / f"batch_{gpu_id}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(query_batch, f)
                
                # Output file for this GPU
                output_file = Path(temp_dir) / f"results_{gpu_id}.pkl"
                
                # Start worker process
                p = mp.Process(
                    target=_gpu_worker,
                    args=(
                        gpu_id,
                        str(batch_file),
                        str(candidate_docs_file),
                        self.model_name,
                        self.model_kwargs,
                        rank_kwargs,
                        metric_kwargs,
                        str(output_file)
                    )
                )
                p.start()
                processes.append((gpu_id, p, output_file))
            
            logger.info(f"All {len(processes)} workers started, waiting for completion...")
            
            # Wait for all processes to complete
            for gpu_id, p, _ in processes:
                p.join()
                if p.exitcode != 0:
                    logger.error(f"GPU {gpu_id} worker failed with exit code {p.exitcode}")
            
            # Merge results from all GPUs
            all_results = [None] * len(eval_samples)
            for gpu_id, _, output_file in processes:
                if output_file.exists():
                    with open(output_file, 'rb') as f:
                        batch_results = pickle.load(f)
                        for query_idx, result in batch_results:
                            all_results[query_idx] = result
            
            return all_results
            
        finally:
            # Cleanup temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
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

