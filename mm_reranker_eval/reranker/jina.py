"""Jina AI multimodal reranker implementation."""

from typing import List, Set
import torch
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, RankResult, Modality


class JinaMMReranker(BaseReranker):
    """
    Jina AI multimodal reranker (jina-reranker-m0).
    
    Supports text-to-image, image-to-text, and mixed modality retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-m0",
        device: str = "cuda",
        use_flash_attention: bool = True,
        **kwargs
    ):
        """
        Initialize Jina multimodal reranker.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda' or 'cpu')
            use_flash_attention: Whether to use flash attention 2 (requires compatible GPU)
            **kwargs: Additional model arguments
        """
        super().__init__(model_name, device, **kwargs)
        self.use_flash_attention = use_flash_attention and device == "cuda"
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Jina reranker model."""
        from transformers import AutoModel
        
        attn_impl = "flash_attention_2" if self.use_flash_attention else None
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def rank(
        self,
        query: Query,
        documents: List[Document],
        max_length: int = 2048,
        **kwargs
    ) -> RankResult:
        """
        Rank documents given a query.
        
        Args:
            query: Query object
            documents: List of documents to rank
            max_length: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            RankResult with ranked indices and scores
        """
        if len(documents) == 0:
            return RankResult(ranked_indices=[], scores=[])
        
        # Validate modalities
        self.validate_modalities(query, documents)
        
        # Convert to Jina format
        query_str = self._to_jina_format(query)
        doc_strs = [self._to_jina_format(doc) for doc in documents]
        
        # Create pairs
        pairs = [[query_str, doc_str] for doc_str in doc_strs]
        
        # Determine doc_type based on document modalities
        doc_type = self._infer_doc_type(documents[0])
        
        # Compute scores
        with torch.no_grad():
            scores = self.model.compute_score(
                pairs,
                max_length=max_length,
                doc_type=doc_type
            )
        
        # Ensure scores is a list
        if not isinstance(scores, list):
            scores = scores.tolist() if hasattr(scores, 'tolist') else [scores]
        
        # Rank by scores (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_scores = [scores[i] for i in ranked_indices]
        
        return RankResult(ranked_indices=ranked_indices, scores=ranked_scores)
    
    def _to_jina_format(self, item: Query | Document) -> str:
        """
        Convert Query or Document to Jina format.
        
        For text-only: returns the text
        For image-only: returns the image path/URL
        For mixed: returns text + image path/URL combined
        
        Args:
            item: Query or Document object
            
        Returns:
            Formatted string for Jina model
        """
        parts = []
        
        if item.text is not None:
            parts.append(item.text)
        
        if item.image is not None:
            parts.append(item.image)
        
        if item.video is not None:
            parts.append(item.video)
        
        # For single modality, return as-is
        # For mixed, combine with space
        return " ".join(parts) if len(parts) > 1 else parts[0]
    
    def _infer_doc_type(self, document: Document) -> str:
        """
        Infer doc_type parameter for Jina model.
        
        Args:
            document: Document to infer type from
            
        Returns:
            'text', 'image', or 'auto'
        """
        modalities = document.get_modalities()
        
        if Modality.IMAGE in modalities and Modality.TEXT not in modalities:
            return "image"
        elif Modality.TEXT in modalities and Modality.IMAGE not in modalities:
            return "text"
        else:
            # Mixed or other
            return "auto"
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for Jina reranker.
        
        Jina reranker supports:
        - text to text
        - text to image
        - image to text
        - image to image
        - mixed to mixed
        - and other combinations
        
        Returns:
            Set of supported (query_modalities, doc_modalities) tuples
        """
        # Jina reranker is quite flexible, supporting most combinations
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

