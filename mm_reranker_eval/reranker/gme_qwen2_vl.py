"""GME-Qwen2-VL multimodal reranker implementation."""

from typing import List, Set
import torch
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, Modality


class GmeQwen2VL(BaseReranker):
    """
    GME-Qwen2-VL multimodal reranker (Alibaba-NLP/gme-Qwen2-VL-7B-Instruct).
    
    Supports single-modal (text, image) and fused-modal (text+image) retrieval
    with optional task instructions.
    """
    
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        instruction: str = None,
        **kwargs
    ):
        """
        Initialize GME-Qwen2-VL reranker.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda' or 'cpu')
            instruction: Task instruction for retrieval (e.g., 'Find an image that matches the given text.')
            **kwargs: Additional model arguments
        """
        super().__init__(model_name, device, **kwargs)
        self.instruction = instruction
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the GME-Qwen2-VL model using SentenceTransformer."""
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
    
    def _format(self, item: Query | Document) -> dict:
        """
        Convert Query or Document to GME-Qwen2-VL format.
        
        Returns a dict with 'text', 'image', and optionally 'prompt' keys.
        """
        result = {}
        
        if item.text is not None:
            result["text"] = item.text
        
        if item.image is not None:
            result["image"] = item.image
        
        return result
    
    def _compute_scores(
        self,
        query_str: dict,
        doc_strs: List[dict],
        query_type: str,
        doc_type: str,
        instruction: str = None,
        **kwargs
    ) -> List[float]:
        """
        Compute scores using GME-Qwen2-VL embeddings.
        
        Args:
            query_str: Formatted query dict with 'text' and/or 'image'
            doc_strs: List of formatted document dicts
            query_type: Query type (not used by GME-Qwen2-VL)
            doc_type: Document type (not used by GME-Qwen2-VL)
            instruction: Task instruction for retrieval
            **kwargs: Additional arguments
        """
        # Use instruction from parameter or instance default
        task_instruction = instruction if instruction is not None else self.instruction
        
        # Prepare query input with instruction if available
        query_input = query_str.copy()
        if task_instruction and "text" in query_input:
            query_input["prompt"] = task_instruction
        
        # Encode query
        query_emb = self.model.encode(
            [query_input],
            convert_to_tensor=True
        )
        
        # Encode documents
        doc_embs = self.model.encode(
            doc_strs,
            convert_to_tensor=True
        )
        
        # Compute similarity scores (cosine similarity via dot product)
        scores = torch.matmul(query_emb, doc_embs.T)
        
        # Convert to list
        scores = scores.squeeze(0).cpu().float().tolist()
        
        if not isinstance(scores, list):
            scores = [scores]
        
        return scores
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for GME-Qwen2-VL.
        
        GME-Qwen2-VL supports:
        - text to text
        - text to image
        - image to text
        - image to image
        - multimodal (text+image) combinations
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        text_image = frozenset([Modality.TEXT, Modality.IMAGE])
        
        return {
            (text, text),
            (text, image),
            (image, text),
            (image, image),
            (text_image, text),
            (text_image, image),
            (text_image, text_image),
            (text, text_image),
            (image, text_image),
        }

