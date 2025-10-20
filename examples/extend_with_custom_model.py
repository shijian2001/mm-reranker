"""Example of extending mm_reranker_eval with a custom reranker model."""

from typing import List, Set
from mm_reranker_eval.reranker import BaseReranker, register_reranker
from mm_reranker_eval.data.types import Query, Document, RankResult, Modality
from mm_reranker_eval import MMReranker


class CustomReranker(BaseReranker):
    """
    Example custom reranker implementation.
    
    This is a dummy implementation for demonstration purposes.
    Replace with your actual model logic.
    """
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """Initialize custom reranker."""
        super().__init__(model_name, device, **kwargs)
        
        # Load your model here
        print(f"Loading custom model: {model_name}")
        # self.model = load_your_model(model_name)
        # self.model.to(device)
    
    def rank(
        self,
        query: Query,
        documents: List[Document],
        **kwargs
    ) -> RankResult:
        """
        Rank documents given a query.
        
        Args:
            query: Query object
            documents: List of documents to rank
            **kwargs: Additional ranking arguments
            
        Returns:
            RankResult with ranked indices and scores
        """
        # Validate modalities
        self.validate_modalities(query, documents)
        
        # Implement your ranking logic here
        # This is a dummy implementation
        scores = [0.5 + i * 0.1 for i in range(len(documents))]
        
        # Sort by scores (descending)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        ranked_scores = [scores[i] for i in ranked_indices]
        
        return RankResult(ranked_indices=ranked_indices, scores=ranked_scores)
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations.
        
        Returns:
            Set of (query_modalities, doc_modalities) tuples
        """
        # Define which modality combinations your model supports
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

