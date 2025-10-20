"""Base abstract class for multimodal rerankers."""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional
from mm_reranker_eval.data.types import Query, Document, RankResult, Modality


class BaseReranker(ABC):
    """
    Abstract base class for all multimodal reranker models.
    
    Uses template method pattern: base class handles the rank() flow,
    subclasses only implement model-specific logic.
    
    Subclasses must implement:
    - _format(): Convert Query/Document to model input format
    - _compute_scores(): Call model API to compute scores
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
    
    def rank(self, query: Query, documents: List[Document], **kwargs) -> RankResult:
        """
        Rank documents given a query (template method).
        
        This method handles the complete ranking flow:
        1. Validate modalities
        2. Group documents by type
        3. Compute scores for each group
        4. Merge and rank results
        
        Args:
            query: Query object (can be text, image, video, or mixed)
            documents: List of Document objects to rank
            **kwargs: Additional ranking arguments passed to _compute_scores
            
        Returns:
            RankResult containing ranked indices and scores
            
        Raises:
            ValueError: If query/document modalities are not supported
        """
        if len(documents) == 0:
            return RankResult(ranked_indices=[], scores=[])
        
        # Validate modalities
        self.validate_modalities(query, documents)
        
        # Format query once
        query_str = self._format(query)
        
        # Group documents by type to handle mixed modality candidates
        doc_groups = self._group_documents_by_type(documents, self._infer_doc_type)
        
        # Compute scores for each document type group
        all_scores = [0.0] * len(documents)
        
        for doc_type, group in doc_groups.items():
            # Extract indices and format documents
            indices = [idx for idx, _ in group]
            doc_strs = [self._format(doc) for _, doc in group]
            
            # Compute scores for this group
            group_scores = self._compute_scores(
                query_str, doc_strs, doc_type, **kwargs
            )
            
            # Assign scores back to original positions
            for idx, score in zip(indices, group_scores):
                all_scores[idx] = score
        
        # Rank by scores (descending)
        ranked_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
        ranked_scores = [all_scores[i] for i in ranked_indices]
        
        return RankResult(ranked_indices=ranked_indices, scores=ranked_scores)
    
    @abstractmethod
    def _format(self, item: Query | Document) -> str:
        """
        Convert Query or Document to model-specific input format.
        
        Args:
            item: Query or Document object
            
        Returns:
            Formatted string for model input
        """
        pass
    
    @abstractmethod
    def _compute_scores(
        self,
        query_str: str,
        doc_strs: List[str],
        doc_type: str,
        **kwargs
    ) -> List[float]:
        """
        Compute relevance scores for documents.
        
        This is where the actual model inference happens. Subclasses should
        call their model API here.
        
        Args:
            query_str: Formatted query string
            doc_strs: List of formatted document strings
            doc_type: Document type ('text', 'image', 'auto', etc.)
            **kwargs: Additional model-specific arguments (e.g., max_length)
            
        Returns:
            List of scores, one per document
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
    
    def _infer_doc_type(self, document: Document) -> str:
        """
        Infer document type for grouping and model parameter selection.
        
        Default implementation returns 'text', 'image', or 'auto'.
        Subclasses can override for custom logic.
        
        Args:
            document: Document to classify
            
        Returns:
            Document type string ('text', 'image', 'auto', etc.)
        """
        modalities = document.get_modalities()
        
        if Modality.IMAGE in modalities and Modality.TEXT not in modalities:
            return "image"
        elif Modality.TEXT in modalities and Modality.IMAGE not in modalities:
            return "text"
        else:
            # Mixed or other modalities
            return "auto"
    
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
    
    def _group_documents_by_type(
        self,
        documents: List[Document],
        type_classifier
    ) -> Dict[str, List[Tuple[int, Document]]]:
        """
        Group documents by type for batch processing.
        
        Internal helper for handling mixed modality candidates.
        
        Args:
            documents: List of documents to group
            type_classifier: Function that returns the type/category for each document
            
        Returns:
            Dictionary mapping type to list of (original_index, document) tuples
        """
        groups: Dict[str, List[Tuple[int, Document]]] = {}
        
        for i, doc in enumerate(documents):
            doc_type = type_classifier(doc)
            if doc_type not in groups:
                groups[doc_type] = []
            groups[doc_type].append((i, doc))
        
        return groups
    
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

