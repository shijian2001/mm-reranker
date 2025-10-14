"""Base abstract class for multimodal rerankers."""

from abc import ABC, abstractmethod
from typing import List, Set
from mm_reranker_eval.data.types import Query, Document, RankResult, Modality


class BaseReranker(ABC):
    """
    Abstract base class for all multimodal reranker models.
    
    Subclasses must implement:
    - rank(): Rerank documents given a query
    - supported_modalities(): Return supported modality combinations
    """
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """
        Initialize base reranker.
        
        Args:
            model_name: Name/path of the model
            device: Device to run on ('cuda' or 'cpu')
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        
    @abstractmethod
    def rank(self, query: Query, documents: List[Document], **kwargs) -> RankResult:
        """
        Rank documents given a query.
        
        Args:
            query: Query object (can be text, image, video, or mixed)
            documents: List of Document objects to rank
            **kwargs: Additional ranking arguments (e.g., max_length)
            
        Returns:
            RankResult containing ranked indices and scores
            
        Raises:
            ValueError: If query/document modalities are not supported
        """
        pass
    
    @abstractmethod
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported query-document modality combinations.
        
        Returns:
            Set of tuples (query_modalities, doc_modalities)
            Each element is a frozenset of Modality enums
            
        Example:
            {
                (frozenset([Modality.TEXT]), frozenset([Modality.IMAGE])),  # txt2img
                (frozenset([Modality.IMAGE]), frozenset([Modality.TEXT])),  # img2txt
            }
        """
        pass
    
    def validate_modalities(self, query: Query, documents: List[Document]) -> None:
        """
        Validate that query and document modalities are supported.
        
        Args:
            query: Query to validate
            documents: Documents to validate
            
        Raises:
            ValueError: If modalities are not supported by this model
        """
        query_mods = frozenset(query.get_modalities())
        
        # Check each document
        for doc in documents:
            doc_mods = frozenset(doc.get_modalities())
            
            # Check if this combination is supported
            supported = self.supported_modalities()
            is_supported = any(
                query_mods == q_mods and doc_mods == d_mods
                for q_mods, d_mods in supported
            )
            
            if not is_supported:
                raise ValueError(
                    f"Model '{self.model_name}' does not support "
                    f"query modalities {query_mods} with document modalities {doc_mods}. "
                    f"Supported combinations: {supported}"
                )
    
    def to(self, device: str) -> "BaseReranker":
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda' or 'cpu')
            
        Returns:
            Self for chaining
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"

