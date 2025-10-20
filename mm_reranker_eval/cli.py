"""Command-line interface for mm_reranker_eval."""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional

from mm_reranker_eval.config import EvalConfig
from mm_reranker_eval.evaluation.evaluator import Evaluator
from mm_reranker_eval.data.types import Document


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: If True, set log level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_candidate_docs(path: str, base_dir: Optional[str] = None) -> list[Document]:
    """
    Load candidate documents from JSONL file.
    
    Args:
        path: Path to JSONL file with documents
        base_dir: Base directory to prepend to image/video paths
        
    Returns:
        List of Document objects
        
    Expected format (one per line):
        {"text": "some text"}
        {"image": "path/to/image.jpg"}
        {"text": "caption", "image": "path/to/image.jpg"}
        
        # With base_dir, relative paths are resolved:
        # base_dir: "/data/images"
        # {"image": "img1.jpg"} -> "/data/images/img1.jpg"
    """
    docs = []
    
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            doc = Document.from_raw(data, base_dir=base_dir)
            docs.append(doc)
    
    return docs


def eval_command(args: argparse.Namespace) -> int:
    """
    Run evaluation command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load configuration
        config = EvalConfig.from_yaml(args.config)
        
        # Validate configuration
        config.validate()
        
        # Load candidate documents
        candidate_docs = load_candidate_docs(config.data.candidate_docs_path, config.data.base_dir)
        logging.info(f"Loaded {len(candidate_docs)} candidate documents")
        
        # Initialize evaluator
        evaluator = Evaluator(
            model_name=config.model.name,
            device=config.model.device,
            num_gpus=config.model.num_gpus,
            use_flash_attention=config.model.use_flash_attention,
            **config.model.kwargs
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            eval_data_path=config.data.eval_data_path,
            candidate_docs=candidate_docs,
            output_dir=config.output_dir,
            recall_k=config.metrics.recall_k,
            ndcg_k=config.metrics.ndcg_k,
            max_queries=config.data.max_queries,
            rank_kwargs=config.rank_kwargs,
            save_per_query=config.save_per_query,
            base_dir=config.data.base_dir
        )
        
        logging.info("Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def create_config_command(args: argparse.Namespace) -> int:
    """
    Create a sample configuration file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    sample_config = {
        "model": {
            "name": "jinaai/jina-reranker-m0",
            "device": "cuda",
            "num_gpus": None,
            "use_flash_attention": True,
            "kwargs": {}
        },
        "data": {
            "eval_data_path": "data/eval.jsonl",
            "candidate_docs_path": "data/candidates.jsonl",
            "max_queries": None
        },
        "metrics": {
            "recall_k": [1, 3, 5, 7, 10],
            "ndcg_k": None
        },
        "output_dir": "results",
        "rank_kwargs": {
            "max_length": 2048
        },
        "save_per_query": False
    }
    
    import yaml
    with open(args.output, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration saved to: {args.output}")
    return 0


def main() -> int:
    """
    Main entry point for CLI.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="mmranker",
        description="Multimodal Reranker Evaluation Tool"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Create-config command
    config_parser = subparsers.add_parser(
        "create-config",
        help="Create a sample configuration file"
    )
    config_parser.add_argument(
        "-o", "--output",
        type=str,
        default="config.yaml",
        help="Output path for configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == "eval":
        return eval_command(args)
    elif args.command == "create-config":
        return create_config_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

