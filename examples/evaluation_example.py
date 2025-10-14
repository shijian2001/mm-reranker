"""Example of running evaluation with the Evaluator class."""

import json
from pathlib import Path
from mm_reranker_eval import Evaluator, Document

def create_sample_data():
    """Create sample evaluation data for demonstration."""
    
    # Create data directory
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample evaluation data (JSONL)
    eval_samples = [
        {
            "query": "A beach town in Mexico",
            "match": "images/beach1.jpg",
            "dataset": "Example",
            "id": 1
        },
        {
            "query": "images/soldier.jpg",
            "match": "A soldier on patrol",
            "dataset": "Example",
            "id": 2
        },
        {
            "query": {"text": "What aircraft is this?", "image": "images/aircraft.jpg"},
            "match": {"text": "Boeing 747 commercial airliner", "image": "images/747.jpg"},
            "dataset": "Example",
            "id": 3
        }
    ]
    
    with open(data_dir / "eval.jsonl", "w") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Create sample candidate documents (JSONL)
    candidates = [
        {"text": "A soldier on patrol"},
        {"image": "images/beach1.jpg"},
        {"image": "images/beach2.jpg"},
        {"text": "Mountain landscape"},
        {"text": "Boeing 747 commercial airliner", "image": "images/747.jpg"},
        {"text": "City skyline at night"},
    ]
    
    with open(data_dir / "candidates.jsonl", "w") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate) + "\n")
    
    print(f"Sample data created in {data_dir}/")
    return data_dir


def load_candidates(path: str):
    """Load candidate documents from JSONL file."""
    docs = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            doc = Document.from_raw(data)
            docs.append(doc)
    return docs


def main():
    """Run evaluation example."""
    
    # Create sample data
    print("Creating sample data...")
    data_dir = create_sample_data()
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = Evaluator(
        model_name="jinaai/jina-reranker-m0",
        device="cuda",
        num_gpus=1,
        use_flash_attention=True
    )
    
    # Load candidates
    candidates = load_candidates(str(data_dir / "candidates.jsonl"))
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(
        eval_data_path=str(data_dir / "eval.jsonl"),
        candidate_docs=candidates,
        output_dir="results/example",
        recall_k=[1, 3, 5],
        max_queries=3,
        save_per_query=True
    )
    
    print("\nEvaluation completed!")
    print(f"Results saved to: results/example/")


if __name__ == "__main__":
    main()

