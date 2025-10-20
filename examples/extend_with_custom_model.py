"""Example of extending mm_reranker_eval with a custom reranker model."""

from typing import List, Set
from mm_reranker_eval.reranker import BaseReranker, register_reranker
from mm_reranker_eval.data.types import Query, Document, RankResult, Modality
from mm_reranker_eval import MMReranker


class CustomReranker(BaseReranker):
    """
    Example custom reranker implementation.
    
    The base class handles the ranking flow (template method pattern).
    You only need to implement 3 methods:
    - _format(): Convert Query/Document to model input format
    - _compute_scores(): Call your model API to compute scores
    - supported_modalities(): Declare supported modality combinations
    
    The base class automatically handles:
    - Document grouping by type
    - Score merging and ranking
    - Modality validation
    """
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """Initialize custom reranker."""
        super().__init__(model_name, device, **kwargs)
        
        # Load your model here
        print(f"Loading custom model: {model_name}")
        # self.model = load_your_model(model_name)
        # self.model.to(device)
    
    def _format(self, item: Query | Document) -> str:
        """
        Convert Query or Document to model input format.
        
        This method is called for both queries and documents.
        
        Args:
            item: Query or Document object
            
        Returns:
            Formatted string for model input
        """
        # Simple example: just return text or image path
        if item.text is not None:
            return item.text
        elif item.image is not None:
            return item.image
        elif item.video is not None:
            return item.video
        else:
            return ""
    
    def _compute_scores(
        self,
        query_str: str,
        doc_strs: List[str],
        doc_type: str,
        **kwargs
    ) -> List[float]:
        """
        Compute scores using your model.
        
        Args:
            query_str: Formatted query string
            doc_strs: List of formatted document strings
            doc_type: Document type ('text', 'image', 'auto')
            **kwargs: Additional model-specific arguments
            
        Returns:
            List of scores, one per document
        """
        # This is a dummy implementation for demonstration
        # Replace with actual model inference:
        # scores = self.model.compute_similarity(query_str, doc_strs)
        
        # Dummy scores for demo
        import random
        scores = [random.random() for _ in doc_strs]
        return scores
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Declare supported modality combinations.
        
        Returns:
            Set of (query_modalities, doc_modalities) tuples
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        
        return {
            (text, text),      # text to text
            (text, image),     # text to image
            (image, text),     # image to text
            (image, image),    # image to image
        }


def main():
    """Demonstrate custom model registration and usage."""
    
    # Register your custom reranker
    # Option 1: Register for specific model name
    register_reranker("my-custom-model", CustomReranker)
    
    # Option 2: Register with pattern for auto-detection of local paths
    register_reranker("my-custom-model-v2", CustomReranker, pattern="mycustom")
    # Now any local path with "mycustom" in name will auto-detect this reranker:
    # MMReranker("./models/mycustom-finetuned") -> uses CustomReranker
    
    # Now you can use it with the MMReranker factory
    reranker = MMReranker("my-custom-model", device="cuda")
    
    # Use it like any other reranker
    query = Query(text="example query")
    documents = [
        Document(text="document 1"),
        Document(text="document 2"),
        Document(text="document 3"),
    ]
    
    result = reranker.rank(query, documents)
    print(f"Ranked indices: {result.ranked_indices}")
    print(f"Scores: {result.scores}")
    
    print("\nCustom model registration successful!")
    print("\nWith pattern registration, you can now use:")
    print("  reranker = MMReranker('./models/mycustom-v1', device='cuda')")


if __name__ == "__main__":
    main()

