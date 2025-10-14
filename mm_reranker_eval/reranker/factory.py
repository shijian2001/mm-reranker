"""Factory for creating reranker instances."""

from typing import Optional
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.reranker.jina import JinaMMReranker


# Registry mapping model names to reranker classes
RERANKER_REGISTRY = {
    "jinaai/jina-reranker-m0": JinaMMReranker,
    "jina-reranker-m0": JinaMMReranker,
}


def MMReranker(
    model_name: str,
    device: str = "cuda",
    **kwargs
) -> BaseReranker:
    """
    Factory function to create appropriate reranker based on model name.
    
    Args:
        model_name: Name or path of the model
        device: Device to run on ('cuda' or 'cpu')
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized reranker instance
        
    Raises:
        ValueError: If model_name is not recognized
        
    Example:
        >>> reranker = MMReranker("jinaai/jina-reranker-m0", device="cuda")
        >>> results = reranker.rank(query, documents)
    """
    # Check registry for exact match
    if model_name in RERANKER_REGISTRY:
        reranker_class = RERANKER_REGISTRY[model_name]
        return reranker_class(model_name=model_name, device=device, **kwargs)
    
    # Check for partial matches (e.g., local paths to known models)
    for key, reranker_class in RERANKER_REGISTRY.items():
        if key in model_name or model_name in key:
            return reranker_class(model_name=model_name, device=device, **kwargs)
    
    # Model not found
    raise ValueError(
        f"Unknown model: '{model_name}'. "
        f"Supported models: {list(RERANKER_REGISTRY.keys())}"
    )


def register_reranker(model_name: str, reranker_class: type) -> None:
    """
    Register a new reranker class for a model name.
    
    This allows users to extend the package with custom rerankers.
    
    Args:
        model_name: Model name to register
        reranker_class: Reranker class (must inherit from BaseReranker)
        
    Example:
        >>> from mm_reranker_eval.reranker import register_reranker, BaseReranker
        >>> class MyReranker(BaseReranker):
        ...     # Implementation
        ...     pass
        >>> register_reranker("my-model", MyReranker)
    """
    if not issubclass(reranker_class, BaseReranker):
        raise ValueError(f"{reranker_class} must inherit from BaseReranker")
    
    RERANKER_REGISTRY[model_name] = reranker_class

