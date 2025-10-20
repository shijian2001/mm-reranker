"""
Example usage of GLM-4.1V-9B-Thinking reranker with three different scoring modes.

This example demonstrates:
1. logits_pyes: P(Yes) extraction from Yes/No token logits (fast, efficient)
2. generative_ranking: Generate ranking sequence (interpretable, flexible)
3. listwise_scores: Generate list-wise scores (fine-grained control)
"""

import torch
from mm_reranker_eval.reranker import GLM4VThinkingReranker
from mm_reranker_eval.data.types import Query, Document


def example_text_to_image_logits_pyes():
    """Example: Text-to-Image ranking using logits_pyes mode."""
    print("=" * 80)
    print("Example 1: Text-to-Image ranking with logits_pyes mode")
    print("=" * 80)
    
    # Initialize reranker with logits_pyes mode
    reranker = GLM4VThinkingReranker(
        model_name="THUDM/GLM-4.1V-9B-Thinking",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_mode="logits_pyes",
        torch_dtype=torch.bfloat16
    )
    
    # Create a text query
    query = Query(text="a cat sitting on a sofa")
    
    # Create image documents (replace with your actual image paths)
    documents = [
        Document(image="path/to/cat_on_sofa.jpg"),
        Document(image="path/to/dog_running.jpg"),
        Document(image="path/to/empty_room.jpg"),
    ]
    
    # Rank documents
    result = reranker.rank(query, documents)
    
    print(f"\nQuery: {query.text}")
    print("\nRanking results:")
    for rank, (idx, score) in enumerate(zip(result.ranked_indices, result.scores), 1):
        print(f"  Rank {rank}: Document {idx} (score: {score:.4f})")
    
    print("\n")


def example_image_to_text_generative_ranking():
    """Example: Image-to-Text ranking using generative_ranking mode."""
    print("=" * 80)
    print("Example 2: Image-to-Text ranking with generative_ranking mode")
    print("=" * 80)
    
    # Initialize reranker with generative_ranking mode
    reranker = GLM4VThinkingReranker(
        model_name="THUDM/GLM-4.1V-9B-Thinking",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_mode="generative_ranking",
        torch_dtype=torch.bfloat16,
        max_new_tokens=256
    )
    
    # Create an image query
    query = Query(image="path/to/beach_sunset.jpg")
    
    # Create text documents
    documents = [
        Document(text="A beautiful sunset over the ocean with golden light."),
        Document(text="People playing volleyball on a sandy beach."),
        Document(text="A mountain landscape covered with snow."),
    ]
    
    # Rank documents
    result = reranker.rank(query, documents)
    
    print(f"\nQuery: [Image: {query.image}]")
    print("\nRanking results:")
    for rank, (idx, score) in enumerate(zip(result.ranked_indices, result.scores), 1):
        doc_text = documents[idx].text[:50] + "..." if len(documents[idx].text) > 50 else documents[idx].text
        print(f"  Rank {rank}: Document {idx} (score: {score:.4f})")
        print(f"    Text: {doc_text}")
    
    print("\n")


def example_text_to_text_listwise_scores():
    """Example: Text-to-Text ranking using listwise_scores mode."""
    print("=" * 80)
    print("Example 3: Text-to-Text ranking with listwise_scores mode")
    print("=" * 80)
    
    # Initialize reranker with listwise_scores mode
    reranker = GLM4VThinkingReranker(
        model_name="THUDM/GLM-4.1V-9B-Thinking",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_mode="listwise_scores",
        torch_dtype=torch.bfloat16,
        max_new_tokens=256
    )
    
    # Create a text query
    query = Query(text="What are the benefits of deep learning?")
    
    # Create text documents
    documents = [
        Document(text="Deep learning enables automatic feature extraction from raw data, "
                     "eliminating the need for manual feature engineering."),
        Document(text="Machine learning is a subset of artificial intelligence that "
                     "uses statistical techniques to give computers the ability to learn."),
        Document(text="Neural networks with multiple layers can learn hierarchical "
                     "representations of data, achieving state-of-the-art results."),
        Document(text="Python is a popular programming language for data science."),
    ]
    
    # Rank documents
    result = reranker.rank(query, documents)
    
    print(f"\nQuery: {query.text}")
    print("\nRanking results:")
    for rank, (idx, score) in enumerate(zip(result.ranked_indices, result.scores), 1):
        doc_text = documents[idx].text[:60] + "..." if len(documents[idx].text) > 60 else documents[idx].text
        print(f"  Rank {rank}: Document {idx} (score: {score:.4f})")
        print(f"    Text: {doc_text}")
    
    print("\n")


def example_multimodal_query():
    """Example: Multimodal (text+image) query ranking."""
    print("=" * 80)
    print("Example 4: Multimodal query with logits_pyes mode")
    print("=" * 80)
    
    # Initialize reranker
    reranker = GLM4VThinkingReranker(
        model_name="THUDM/GLM-4.1V-9B-Thinking",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_mode="logits_pyes",
        torch_dtype=torch.bfloat16
    )
    
    # Create a multimodal query (text + image)
    query = Query(
        text="Find similar architectural styles",
        image="path/to/building.jpg"
    )
    
    # Create mixed documents
    documents = [
        Document(
            text="Gothic architecture with pointed arches",
            image="path/to/gothic_building.jpg"
        ),
        Document(text="Modern minimalist architecture with clean lines"),
        Document(image="path/to/classical_building.jpg"),
    ]
    
    # Rank documents
    result = reranker.rank(query, documents)
    
    print(f"\nQuery: {query.text} + [Image]")
    print("\nRanking results:")
    for rank, (idx, score) in enumerate(zip(result.ranked_indices, result.scores), 1):
        doc = documents[idx]
        doc_desc = []
        if doc.text:
            doc_desc.append(f"Text: {doc.text[:40]}...")
        if doc.image:
            doc_desc.append(f"Image: {doc.image}")
        
        print(f"  Rank {rank}: Document {idx} (score: {score:.4f})")
        print(f"    {' | '.join(doc_desc)}")
    
    print("\n")


def example_factory_usage():
    """Example: Using the factory pattern to create reranker."""
    print("=" * 80)
    print("Example 5: Using MMReranker factory")
    print("=" * 80)
    
    from mm_reranker_eval.reranker import MMReranker
    
    # Create reranker using factory (auto-detects GLM4V)
    reranker = MMReranker(
        model_name="THUDM/GLM-4.1V-9B-Thinking",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_mode="logits_pyes",  # Can specify scoring mode
        torch_dtype=torch.bfloat16
    )
    
    print(f"Created reranker: {reranker}")
    print(f"Supported modalities: {reranker.supported_modalities()}")
    
    print("\n")


def main():
    """Run all examples."""
    print("\n")
    print("GLM-4.1V-9B-Thinking Reranker Examples")
    print("=" * 80)
    print("\nNote: Replace 'path/to/...' with actual image paths to run these examples.")
    print("\n")
    
    # Example 1: Text-to-Image with logits_pyes
    example_text_to_image_logits_pyes()
    
    # Example 2: Image-to-Text with generative_ranking
    example_image_to_text_generative_ranking()
    
    # Example 3: Text-to-Text with listwise_scores
    example_text_to_text_listwise_scores()
    
    # Example 4: Multimodal query
    example_multimodal_query()
    
    # Example 5: Factory usage
    example_factory_usage()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

