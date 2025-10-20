"""Utility functions for mm_reranker_eval."""

import json
from pathlib import Path
from typing import List, Union, Optional

from mm_reranker_eval.data.types import Document, EvalSample


def load_documents(path: Union[str, Path], base_dir: Optional[str] = None) -> List[Document]:
    """
    Load documents from JSONL file.
    
    Args:
        path: Path to JSONL file
        base_dir: Base directory to prepend to image/video paths
        
    Returns:
        List of Document objects
        
    Example format:
        {"text": "some text"}
        {"image": "path/to/image.jpg"}
        {"text": "caption", "image": "path/to/image.jpg"}
    """
    documents = []
    
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                doc = Document.from_raw(data, base_dir=base_dir)
                documents.append(doc)
    
    return documents


def load_eval_samples(path: Union[str, Path], max_samples: int = None, base_dir: Optional[str] = None) -> List[EvalSample]:
    """
    Load evaluation samples from JSONL file.
    
    Args:
        path: Path to JSONL file
        max_samples: Maximum number of samples to load (None = all)
        base_dir: Base directory to prepend to image/video paths
        
    Returns:
        List of EvalSample objects
    """
    samples = []
    
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            if line.strip():
                data = json.loads(line)
                sample = EvalSample.from_json(data, base_dir=base_dir)
                samples.append(sample)
    
    return samples


def save_documents(documents: List[Document], path: Union[str, Path]) -> None:
    """
    Save documents to JSONL file.
    
    Args:
        documents: List of Document objects
        path: Output path
    """
    with open(path, "w") as f:
        for doc in documents:
            data = {}
            if doc.text is not None:
                data["text"] = doc.text
            if doc.image is not None:
                data["image"] = doc.image
            if doc.video is not None:
                data["video"] = doc.video
            
            f.write(json.dumps(data) + "\n")

