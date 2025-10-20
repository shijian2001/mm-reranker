"""Example usage of BGE-VL-MLLM-S1 multimodal reranker."""

from mm_reranker_eval.reranker import MMReranker, BgeVlMllmReranker
from mm_reranker_eval.data.types import Query, Document


def main():
    # Example 1: Using factory function with instruction
    print("=" * 80)
    print("Example 1: Text-to-Image Retrieval with Instruction")
    print("=" * 80)
    
    reranker = MMReranker(
        "BAAI/BGE-VL-MLLM-S1",
        device="cuda",
        instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
    )
    
    # Create query with text and image (multimodal)
    query = Query(
        text="Make the background dark, as if the camera has taken the photo at night",
        image="./assets/cir_query.png"
    )
    
    # Create candidate documents (images)
    documents = [
        Document(image="./assets/cir_candi_1.png"),
        Document(image="./assets/cir_candi_2.png"),
    ]
    
    # Rank documents
    results = reranker.rank(query, documents)
    print(f"Ranked indices: {results.ranked_indices}")
    print(f"Scores: {results.scores}")
    print()
    
    # Example 2: Direct instantiation
    print("=" * 80)
    print("Example 2: Direct Instantiation")
    print("=" * 80)
    
    reranker = BgeVlMllmReranker(
        model_name="BAAI/BGE-VL-MLLM-S1",
        device="cuda"
    )
    
    query = Query(text="A cat sitting on a couch")
    documents = [
        Document(image="./images/cat1.jpg"),
        Document(image="./images/cat2.jpg"),
        Document(image="./images/dog1.jpg"),
    ]
    
    results = reranker.rank(query, documents)
    print(f"Ranked indices: {results.ranked_indices}")
    print(f"Scores: {results.scores}")
    print()
    
    # Example 3: Demonstrating parameter warning mechanism
    print("=" * 80)
    print("Example 3: Parameter Warning Mechanism")
    print("=" * 80)
    
    # instruction is supported by BGE-VL-MLLM but not by Jina models
    from mm_reranker_eval.reranker import JinaClipReranker
    
    # This will print a warning about instruction not being used
    jina_reranker = JinaClipReranker(
        instruction="This parameter is not used by Jina CLIP v2",
        use_flash_attention=True  # This will also trigger a warning
    )
    
    query = Query(text="test query")
    documents = [Document(text="test document")]
    
    results = jina_reranker.rank(
        query,
        documents,
        instruction="This will also trigger a warning in compute_scores"
    )
    print(f"Results (with warnings): {results.ranked_indices}")
    print()
    
    # Example 4: Multimodal combinations
    print("=" * 80)
    print("Example 4: Various Modality Combinations")
    print("=" * 80)
    
    reranker = BgeVlMllmReranker()
    
    print("Supported modality combinations:")
    for q_mod, d_mod in sorted(reranker.supported_modalities()):
        print(f"  Query: {q_mod} -> Document: {d_mod}")


if __name__ == "__main__":
    main()

