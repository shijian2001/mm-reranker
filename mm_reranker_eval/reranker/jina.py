"""Jina AI multimodal reranker implementation."""

from typing import List, Set, Union
import torch
import numpy as np
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
    
    def _format(self, item: Query | Document) -> str:
        """
        Convert Query or Document to Jina format.
        
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
        
        # For single modality, return as-is; for mixed, combine with space
        return " ".join(parts) if len(parts) > 1 else parts[0]
    
    def _compute_scores(
        self,
        query_str: str,
        doc_strs: List[str],
        query_type: str,
        doc_type: str,
        max_length: int = 2048,
        **kwargs
    ) -> List[float]:
        """
        Compute scores using Jina model API.
        
        Args:
            query_str: Formatted query string
            doc_strs: List of formatted document strings
            query_type: Query type ('text', 'image', 'auto')
            doc_type: Document type ('text', 'image', 'auto')
            max_length: Maximum sequence length
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of relevance scores
        """
        # Create query-document pairs
        pairs = [[query_str, doc_str] for doc_str in doc_strs]
        
        # Compute scores with Jina model
        with torch.no_grad():
            scores = self.model.compute_score(
                pairs,
                max_length=max_length,
                query_type=query_type,
                doc_type=doc_type
            )
        
        # Ensure scores is a list
        if not isinstance(scores, list):
            scores = scores.tolist() if hasattr(scores, 'tolist') else [scores]
        
        return scores
    
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


class JinaClipReranker(BaseReranker):
    """
    Jina CLIP v2 multimodal reranker (jina-clip-v2).
    
    Uses embedding-based similarity for text-to-image, image-to-text, 
    text-to-text, and image-to-image retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-clip-v2",
        device: str = "cuda",
        truncate_dim: int = None,
        **kwargs
    ):
        """
        Initialize Jina CLIP v2 reranker.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda' or 'cpu')
            truncate_dim: Matryoshka dimension (None for full 1024-dim vectors)
            **kwargs: Additional model arguments
        """
        super().__init__(model_name, device, **kwargs)
        self.truncate_dim = truncate_dim
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Jina CLIP v2 model."""
        from transformers import AutoModel
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def _format(self, item: Query | Document) -> Union[str, List[str]]:
        """
        Convert Query or Document to Jina CLIP v2 format.
        
        For text-only items, returns a string.
        For image-only items, returns a list with image path/URL.
        For mixed items, returns a list with both text and image.
        
        Args:
            item: Query or Document object
            
        Returns:
            Formatted string or list for Jina CLIP v2 model
        """
        modalities = item.get_modalities()
        
        # Pure text
        if Modality.TEXT in modalities and Modality.IMAGE not in modalities:
            return item.text
        
        # Pure image
        if Modality.IMAGE in modalities and Modality.TEXT not in modalities:
            return item.image
        
        # Mixed modality - for now, prioritize based on what's available
        # Note: jina-clip-v2 doesn't natively support mixed inputs in a single embedding
        # We'll return the image if available, otherwise text
        if item.image is not None:
            return item.image
        elif item.text is not None:
            return item.text
        
        raise ValueError(f"Item has no valid text or image content: {item}")
    
    def _encode_text(self, texts: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """
        Encode text using Jina CLIP v2.
        
        Args:
            texts: Single text or list of texts
            is_query: Whether this is a query (uses 'retrieval.query' task)
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            if is_query:
                embeddings = self.model.encode_text(
                    texts,
                    task='retrieval.query',
                    truncate_dim=self.truncate_dim
                )
            else:
                embeddings = self.model.encode_text(
                    texts,
                    truncate_dim=self.truncate_dim
                )
        
        # Convert to numpy if it's a tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def _encode_image(self, images: Union[str, List[str]]) -> np.ndarray:
        """
        Encode images using Jina CLIP v2.
        
        Args:
            images: Single image or list of image paths/URLs
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(images, str):
            images = [images]
        
        with torch.no_grad():
            embeddings = self.model.encode_image(
                images,
                truncate_dim=self.truncate_dim
            )
        
        # Convert to numpy if it's a tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def _compute_scores(
        self,
        query_str: Union[str, List[str]],
        doc_strs: List[Union[str, List[str]]],
        query_type: str,
        doc_type: str,
        **kwargs
    ) -> List[float]:
        """
        Compute scores using Jina CLIP v2 embeddings and dot product similarity.
        
        Args:
            query_str: Formatted query (text or image path)
            doc_strs: List of formatted documents (text or image paths)
            query_type: Query type ('text' or 'image')
            doc_type: Document type ('text' or 'image')
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of similarity scores
        """
        # Encode query
        if query_type == "text":
            query_embedding = self._encode_text(query_str, is_query=True)
        elif query_type == "image":
            query_embedding = self._encode_image(query_str)
        else:
            # Auto: try to determine from content
            if isinstance(query_str, str) and (
                query_str.startswith('http://') or 
                query_str.startswith('https://') or
                query_str.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
            ):
                query_embedding = self._encode_image(query_str)
            else:
                query_embedding = self._encode_text(query_str, is_query=True)
        
        # Encode documents
        if doc_type == "text":
            doc_embeddings = self._encode_text(doc_strs, is_query=False)
        elif doc_type == "image":
            doc_embeddings = self._encode_image(doc_strs)
        else:
            # Auto: try to determine from content
            # Check first document to determine type
            if doc_strs and isinstance(doc_strs[0], str) and (
                doc_strs[0].startswith('http://') or 
                doc_strs[0].startswith('https://') or
                doc_strs[0].endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
            ):
                doc_embeddings = self._encode_image(doc_strs)
            else:
                doc_embeddings = self._encode_text(doc_strs, is_query=False)
        
        # Ensure query_embedding is 1D for single query
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
        
        # Compute dot product similarity
        # query_embedding shape: (dim,)
        # doc_embeddings shape: (num_docs, dim)
        scores = np.dot(doc_embeddings, query_embedding)
        
        # Convert to list
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [scores]
        
        return scores
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for Jina CLIP v2.
        
        Jina CLIP v2 supports:
        - text to text
        - text to image
        - image to text
        - image to image
        
        Returns:
            Set of supported (query_modalities, doc_modalities) tuples
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        
        return {
            (text, text),
            (text, image),
            (image, text),
            (image, image),
        }

