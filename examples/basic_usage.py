"""Basic usage example for mm_reranker_eval."""

from mm_reranker_eval import MMReranker, Query, Document

def main():
    """Demonstrate basic usage of the package."""
    
    # Initialize reranker with remote model name
    print("Initializing reranker...")
    reranker = MMReranker(
        model_name="jinaai/jina-reranker-m0",
        device="cuda",
        use_flash_attention=True
    )
    
    # Or use a local model path (model type auto-detected):
    # reranker = MMReranker(
    #     model_name="/path/to/local/model",
    #     device="cuda",
    #     use_flash_attention=True
    # )
    # reranker = MMReranker(
    #     model_name="./models/my-model",
    #     device="cuda"
    # )
    
    # Example 1: Text to Image
    print("\n=== Example 1: Text to Image ===")
    query = Query(text="slm markdown")
    documents = [
        Document(image="https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"),
        Document(image="https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"),
        Document(image="https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/wired-preview.png"),
    ]
    
    # The reranker automatically infers:
    #   - query_type="text" (pure text query)
    #   - doc_type="image" (image documents)
    result = reranker.rank(query, documents)
    print(f"Ranked indices: {result.ranked_indices}")
    print(f"Scores: {result.scores}")
    
    # Example 2: Image to Text
    print("\n=== Example 2: Image to Text ===")
    query = Query(image="path/to/query_image.jpg")
    documents = [
        Document(text="A beautiful sunset over the ocean"),
        Document(text="A cat sleeping on a couch"),
        Document(text="Mountain landscape with snow"),
    ]
    
    # Note: This will work if you have actual image files
    # The reranker would automatically infer:
    #   - query_type="image" (image query)
    #   - doc_type="text" (text documents)
    # result = reranker.rank(query, documents)
    # print(f"Ranked indices: {result.ranked_indices}")
    
    # Example 3: Mixed Modality
    print("\n=== Example 3: Mixed Modality ===")
    query = Query(text="What is this building?", image="path/to/building.jpg")
    documents = [
        Document(text="The Eiffel Tower in Paris", image="path/to/eiffel.jpg"),
        Document(text="The Statue of Liberty in New York", image="path/to/statue.jpg"),
        Document(text="Big Ben in London", image="path/to/bigben.jpg"),
    ]
    
    # Note: This will work if you have actual image files
    # The reranker would automatically infer:
    #   - query_type="auto" (mixed: text + image query)
    #   - doc_type="auto" (mixed: text + image documents)
    # result = reranker.rank(query, documents)
    # print(f"Ranked indices: {result.ranked_indices}")
    
    # Example 4: Using base_dir for relative paths
    print("\n=== Example 4: Relative Paths with base_dir ===")
    base_dir = "/path/to/images"  # Your image directory
    
    # Create documents with relative paths
    query_with_base = Query.from_raw("relative/path/query.jpg", base_dir=base_dir)
    docs_with_base = [
        Document.from_raw({"image": "img1.jpg"}, base_dir=base_dir),
        Document.from_raw({"image": "img2.jpg"}, base_dir=base_dir),
    ]
    
    print(f"Query image path: {query_with_base.image}")
    print(f"Document image paths: {[doc.image for doc in docs_with_base]}")
    
    # Note: This will work if you have actual image files
    # result = reranker.rank(query_with_base, docs_with_base)
    # print(f"Ranked indices: {result.ranked_indices}")
    
    print("\nBasic usage examples completed!")


if __name__ == "__main__":
    main()

