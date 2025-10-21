"""Independent worker script for GPU-parallel evaluation."""

import sys
import os
from pathlib import Path

# Ensure mm_reranker_eval can be imported
# Add project root to path if not already there
script_dir = Path(__file__).parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import pickle
import time


def main():
    """Main worker function that runs in a separate process."""
    if len(sys.argv) != 9:
        print("Usage: _worker.py <gpu_id> <query_batch_file> <candidate_docs_file> "
              "<model_name> <model_kwargs_file> <rank_kwargs_file> <metric_kwargs_file> <output_file>")
        sys.exit(1)
    
    gpu_id = int(sys.argv[1])
    query_batch_file = sys.argv[2]
    candidate_docs_file = sys.argv[3]
    model_name = sys.argv[4]
    model_kwargs_file = sys.argv[5]
    rank_kwargs_file = sys.argv[6]
    metric_kwargs_file = sys.argv[7]
    output_file = sys.argv[8]
    
    start_time = time.time()
    print(f"[GPU {gpu_id}] ========== Worker started at {time.strftime('%H:%M:%S')} ==========", flush=True)
    
    # Load all parameters
    with open(model_kwargs_file, 'rb') as f:
        model_kwargs = pickle.load(f)
    with open(rank_kwargs_file, 'rb') as f:
        rank_kwargs = pickle.load(f)
    with open(metric_kwargs_file, 'rb') as f:
        metric_kwargs = pickle.load(f)
    
    # Load data
    t0 = time.time()
    with open(query_batch_file, 'rb') as f:
        query_batch = pickle.load(f)
    with open(candidate_docs_file, 'rb') as f:
        candidate_docs = pickle.load(f)
    load_time = time.time() - t0
    print(f"[GPU {gpu_id}] Loaded {len(query_batch)} queries and {len(candidate_docs)} candidates in {load_time:.1f}s", flush=True)
    
    # Load model on the assigned GPU
    t1 = time.time()
    from mm_reranker_eval.reranker.factory import MMReranker
    device = f"cuda:{gpu_id}"
    reranker = MMReranker(model_name, device=device, **model_kwargs)
    model_time = time.time() - t1
    print(f"[GPU {gpu_id}] Model loaded in {model_time:.1f}s, starting evaluation", flush=True)
    
    # Import evaluation function
    from mm_reranker_eval.evaluation.evaluator import _evaluate_single_query
    
    # Process queries
    results = []
    total = len(query_batch)
    eval_start = time.time()
    
    for i, (query_idx, eval_sample) in enumerate(query_batch):
        result = _evaluate_single_query(
            reranker, eval_sample, candidate_docs, rank_kwargs, metric_kwargs
        )
        results.append((query_idx, result))
        
        # Print progress every 10 queries
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - eval_start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"[GPU {gpu_id}] Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) | "
                  f"Speed: {speed:.2f} q/s | ETA: {eta/60:.1f}min", flush=True)
    
    # Save results
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    total_time = time.time() - start_time
    eval_time = time.time() - eval_start
    avg_speed = total / eval_time if eval_time > 0 else 0
    print(f"[GPU {gpu_id}] ========== Completed in {total_time/60:.1f}min "
          f"(eval: {eval_time/60:.1f}min, {avg_speed:.2f} q/s) ==========", flush=True)


if __name__ == '__main__':
    main()

