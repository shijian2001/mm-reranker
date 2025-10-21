"""Evaluator for multimodal reranker models with parallel GPU support."""

import os
import json
import logging
import tempfile
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


def _evaluate_query_batch(
    gpu_id: int,
    worker_id: int,
    query_batch: List[Tuple[int, EvalSample]],
    candidate_docs: List[Document],
    model_name: str,
    model_kwargs: Dict[str, Any],
    rank_kwargs: Dict[str, Any],
    metric_kwargs: Dict[str, Any],
    temp_dir: str,
    progress_queue: Optional[mp.Queue] = None
) -> str:
    """
    Evaluate a batch of queries on an isolated GPU (worker process function).
    
    This function runs in a separate process. It sets CUDA_VISIBLE_DEVICES first
    to isolate the GPU, then loads the model once and processes all queries.
    
    Args:
        gpu_id: Physical GPU ID to isolate
        worker_id: Worker ID for identification
        query_batch: List of (query_idx, eval_sample) tuples to evaluate
        candidate_docs: List of candidate documents (including ground truth)
        model_name: Model name for initialization
        model_kwargs: Model initialization kwargs
        rank_kwargs: Ranking kwargs
        metric_kwargs: Metric computation kwargs
        temp_dir: Directory to save temporary results
        progress_queue: Optional queue for progress updates
        
    Returns:
        Path to temporary result file
    """
    # CRITICAL: Set GPU isolation BEFORE any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Load model once on the isolated GPU
    reranker = MMReranker(model_name, device="cuda:0", **model_kwargs)
    
    # Process queries with progress tracking
    results = []
    total = len(query_batch)
    for idx, (query_idx, eval_sample) in enumerate(query_batch):
        result = _evaluate_single_query(
            reranker, eval_sample, candidate_docs, rank_kwargs, metric_kwargs
        )
        results.append((query_idx, result))
        
        # Send progress update
        if progress_queue is not None:
            progress_queue.put((worker_id, gpu_id, idx + 1, total))
    
    # Save results to temporary file
    temp_file = Path(temp_dir) / f"worker_{worker_id}.json"
    with open(temp_file, "w") as f:
        json.dump(results, f)
    
    return str(temp_file)


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
        
        Each GPU gets an isolated batch of queries and loads the model exactly once.
        Uses multiprocessing.Process to ensure proper GPU isolation before CUDA init.
        Progress is displayed with one tqdm bar per GPU showing query-level progress.
        """
        num_workers = min(self.num_gpus, len(eval_samples))
        
        # Split queries evenly across GPUs (handles uneven division)
        batch_size = (len(eval_samples) + num_workers - 1) // num_workers
        query_batches = []
        
        for worker_id in range(num_workers):
            start_idx = worker_id * batch_size
            end_idx = min(start_idx + batch_size, len(eval_samples))
            
            if start_idx >= len(eval_samples):
                break
            
            # Create batch with (query_idx, eval_sample) tuples
            batch = [(i, eval_samples[i]) for i in range(start_idx, end_idx)]
            query_batches.append((worker_id, batch))
        
        # Create temporary directory for worker results
        temp_dir = tempfile.mkdtemp(prefix="mm_reranker_eval_")
        
        # Create progress queue for real-time updates
        progress_queue = mp.Queue()
        
        try:
            # Launch worker processes (one per GPU with isolated environment)
            processes = []
            for worker_id, batch in query_batches:
                p = mp.Process(
                    target=_evaluate_query_batch,
                    args=(
                        worker_id,  # GPU ID = worker ID
                        worker_id,
                        batch,
                        candidate_docs,
                        self.model_name,
                        self.model_kwargs,
                        rank_kwargs,
                        metric_kwargs,
                        temp_dir,
                        progress_queue
                    )
                )
                p.start()
                processes.append(p)
            
            # Monitor progress with multiple tqdm bars
            progress_bars = {}
            for worker_id, batch in query_batches:
                pbar = tqdm(
                    total=len(batch),
                    desc=f"GPU {worker_id}",
                    position=worker_id,
                    leave=True
                )
                progress_bars[worker_id] = pbar
            
            # Update progress bars as workers report
            completed_workers = set()
            while len(completed_workers) < len(query_batches):
                try:
                    worker_id, gpu_id, current, total = progress_queue.get(timeout=0.1)
                    progress_bars[worker_id].n = current
                    progress_bars[worker_id].refresh()
                    
                    if current >= total:
                        completed_workers.add(worker_id)
                except:
                    # Check if any process finished
                    for idx, p in enumerate(processes):
                        if not p.is_alive() and idx not in completed_workers:
                            completed_workers.add(idx)
            
            # Close progress bars
            for pbar in progress_bars.values():
                pbar.close()
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            # Merge results from temporary files
            all_results = [None] * len(eval_samples)
            for worker_id, _ in query_batches:
                temp_file = Path(temp_dir) / f"worker_{worker_id}.json"
                with open(temp_file, "r") as f:
                    batch_results = json.load(f)
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

