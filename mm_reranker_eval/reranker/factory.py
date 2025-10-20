"""Factory for creating reranker instances."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Callable, List
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.reranker.jina import JinaMMReranker, JinaClipReranker
from mm_reranker_eval.reranker.dse_qwen2_mrl import DseQwen2Mrl
from mm_reranker_eval.reranker.bge_vl_mllm import BgeVlMllmReranker
from mm_reranker_eval.reranker.gme_qwen2_vl import GmeQwen2VL
from mm_reranker_eval.reranker.colqwen2 import ColQwen2Reranker


# Registry mapping model names/patterns to reranker classes
RERANKER_REGISTRY = {
    "jinaai/jina-reranker-m0": JinaMMReranker,
    "jinaai/jina-clip-v2": JinaClipReranker,
    "MrLight/dse-qwen2-2b-mrl-v1": DseQwen2Mrl,
    "BAAI/BGE-VL-MLLM-S1": BgeVlMllmReranker,
    "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct": GmeQwen2VL,
    "vidore/colqwen2-v1.0": ColQwen2Reranker,
}

# Registry for model type detection patterns
# Maps pattern keywords to reranker classes
MODEL_TYPE_PATTERNS = {
    "jina-reranker": JinaMMReranker,
    "jina-clip": JinaClipReranker,
    "dse-qwen2": DseQwen2Mrl,
    "bge-vl-mllm": BgeVlMllmReranker,
    "gme-qwen2-vl": GmeQwen2VL,
    "colqwen2": ColQwen2Reranker,
}


def _match_patterns_in_text(text: str, patterns: List[str]) -> bool:
    """Check if any pattern exists in text (case-insensitive)."""
    text_lower = str(text).lower()
    return any(pattern.lower() in text_lower for pattern in patterns)


def _detect_model_type_from_path(model_path: str) -> Optional[type]:
    """
    Detect model type from local path using multiple strategies.
    
    Args:
        model_path: Local path to model directory
        
    Returns:
        Reranker class type or None if cannot detect
    """
    model_path = Path(model_path)
    
    # Strategy 1: Check directory name against registered patterns
    dir_name = model_path.name
    for pattern, reranker_class in MODEL_TYPE_PATTERNS.items():
        if pattern.lower() in dir_name.lower():
            return reranker_class
    
    # Strategy 2: Try to read config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Collect all searchable fields
            searchable_fields = [
                config.get("model_type", ""),
                config.get("_name_or_path", ""),
                config.get("name_or_path", ""),
            ] + config.get("architectures", [])
            
            # Check each pattern against config fields
            for pattern, reranker_class in MODEL_TYPE_PATTERNS.items():
                if _match_patterns_in_text(" ".join(str(f) for f in searchable_fields), [pattern]):
                    return reranker_class
                
        except (json.JSONDecodeError, IOError):
            pass
    
    # Strategy 3: Check against registry by partial model name match
    for registered_name, reranker_class in RERANKER_REGISTRY.items():
        if registered_name in dir_name or dir_name in registered_name:
            return reranker_class
    
    return None


def MMReranker(
    model_name: str,
    device: str = "cuda",
    **kwargs
) -> BaseReranker:
    """
    Factory function to create appropriate reranker based on model name or path.
    
    Supports both remote model names (e.g., HuggingFace) and local paths.
    For local paths, the model type is automatically detected from:
    - Directory name patterns
    - Model config files (config.json)
    - Registered model patterns
    
    Args:
        model_name: Name or path of the model
                   - Remote model name: "jinaai/jina-reranker-m0"
                   - Absolute path: "/path/to/model"
                   - Relative path: "./models/my-model"
        device: Device to run on ('cuda' or 'cpu')
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized reranker instance
        
    Raises:
        ValueError: If model cannot be identified or loaded
        
    Examples:
        >>> # Using remote model name
        >>> reranker = MMReranker("jinaai/jina-reranker-m0", device="cuda")
        
        >>> # Using local path (type auto-detected)
        >>> reranker = MMReranker("/path/to/model", device="cuda")
        >>> reranker = MMReranker("./models/my-model", device="cuda")
    """
    # Strategy 1: Check if it's a local path
    if os.path.exists(model_name) or os.path.isdir(model_name):
        reranker_class = _detect_model_type_from_path(model_name)
        
        if reranker_class:
            return reranker_class(model_name=model_name, device=device, **kwargs)
        else:
            # Cannot detect type from local path
            raise ValueError(
                f"Cannot detect model type from path: '{model_name}'. "
                f"Ensure the directory name or config.json contains recognizable model identifiers, "
                f"or use register_reranker() to manually register this path."
            )
    
    # Strategy 2: Check registry for exact match
    if model_name in RERANKER_REGISTRY:
        reranker_class = RERANKER_REGISTRY[model_name]
        return reranker_class(model_name=model_name, device=device, **kwargs)
    
    # Strategy 3: Check for partial matches in model name
    for key, reranker_class in RERANKER_REGISTRY.items():
        if key in model_name or model_name in key:
            return reranker_class(model_name=model_name, device=device, **kwargs)
    
    # Model not found
    raise ValueError(
        f"Unknown model: '{model_name}'. "
        f"Supported models: {list(RERANKER_REGISTRY.keys())}. "
        f"To add support for a new model, use register_reranker()."
    )


def register_reranker(
    model_name: str, 
    reranker_class: type,
    pattern: Optional[str] = None
) -> None:
    """
    Register a new reranker class for a model name or pattern.
    
    This allows users to extend the package with custom rerankers and
    enable automatic detection for local models.
    
    Args:
        model_name: Model name or identifier to register
        reranker_class: Reranker class (must inherit from BaseReranker)
        pattern: Optional pattern keyword for auto-detection in local paths
                 e.g., "mymodel" will match directories containing "mymodel"
        
    Examples:
        >>> from mm_reranker_eval.reranker import register_reranker, BaseReranker
        >>> 
        >>> class MyReranker(BaseReranker):
        ...     # Implementation
        ...     pass
        >>> 
        >>> # Register for specific model name
        >>> register_reranker("my-model-v1", MyReranker)
        >>> 
        >>> # Register with pattern for auto-detection
        >>> register_reranker("my-model-v1", MyReranker, pattern="mymodel")
        >>> # Now "./models/mymodel-finetuned" will be auto-detected
    """
    if not issubclass(reranker_class, BaseReranker):
        raise ValueError(f"{reranker_class} must inherit from BaseReranker")
    
    RERANKER_REGISTRY[model_name] = reranker_class
    
    # Register pattern for auto-detection if provided
    if pattern:
        MODEL_TYPE_PATTERNS[pattern] = reranker_class

