"""ColQwen2 multimodal reranker implementation."""
## TODO: env management

from typing import List, Set
import torch
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, Modality


class ColQwen2Reranker(BaseReranker):
    """
    ColQwen2 multimodal reranker (vidore/colqwen2-v1.0).
    
    A vision-language model for document retrieval using multi-vector embeddings.
    Supports text-to-image retrieval with ColPali architecture.
    
    Requirements:
        - colpali-engine >= 0.3.4 (or install from source)
        - transformers > 4.46.1
    """
    
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str = "cuda",
        torch_dtype=None,
        use_flash_attention: bool = True,
        **kwargs
    ):
        """
        Initialize ColQwen2 reranker.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda', 'mps', or 'cpu')
            torch_dtype: Torch dtype for model (default: bfloat16)
            use_flash_attention: Whether to use flash attention 2 if available
            **kwargs: Additional model arguments
        """
        super().__init__(model_name, device, **kwargs)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.use_flash_attention = use_flash_attention
        self._check_dependencies()
        self._load_model()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed and compatible."""
        try:
            import colpali_engine
            import transformers
            
            # Check transformers version
            from packaging import version
            tf_version = version.parse(transformers.__version__)
            if tf_version <= version.parse("4.46.1"):
                raise ImportError(
                    f"transformers version {transformers.__version__} is not supported. "
                    "Please upgrade to > 4.46.1: pip install transformers>=4.46.2"
                )
            
            # Check colpali-engine version
            colpali_version = version.parse(colpali_engine.__version__)
            if colpali_version < version.parse("0.3.4"):
                raise ImportError(
                    f"colpali-engine version {colpali_engine.__version__} is not supported. "
                    "Please upgrade to >= 0.3.4 or install from source:\n"
                    "pip install git+https://github.com/illuin-tech/colpali"
                )
                
        except ImportError as e:
            if "colpali_engine" in str(e):
                raise ImportError(
                    "colpali-engine is not installed. Please install it:\n"
                    "pip install git+https://github.com/illuin-tech/colpali\n"
                    "Or install from PyPI (>= 0.3.4): pip install colpali-engine>=0.3.4"
                ) from e
            elif "packaging" in str(e):
                raise ImportError(
                    "packaging is required for version checking. Please install it:\n"
                    "pip install packaging"
                ) from e
            else:
                raise

    def _load_model(self) -> None:
        """Load the ColQwen2 model and processor."""
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from transformers.utils.import_utils import is_flash_attn_2_available
        
        # Determine attention implementation
        attn_implementation = None
        if self.use_flash_attention and is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"
        
        # Load model
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device,
        }
        
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        
        self.model = ColQwen2.from_pretrained(
            self.model_name,
            **load_kwargs
        ).eval()
        
        # Load processor
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)
    
    def _format(self, item: Query | Document) -> dict:
        """
        Convert Query or Document to ColQwen2 format.
        
        Returns a dict with 'text' and 'image' keys.
        """
        result = {"text": None, "image": None}
        
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
        **kwargs
    ) -> List[float]:
        """
        Compute scores using ColQwen2 multi-vector embeddings.
        
        Args:
            query_str: Formatted query dict with 'text' and 'image'
            doc_strs: List of formatted document dicts
            query_type: Query type ('text', 'image', etc.)
            doc_type: Document type ('text', 'image', etc.)
            **kwargs: Additional arguments
        """
        with torch.no_grad():
            # Process queries
            queries = []
            if query_str["text"] is not None:
                queries.append(query_str["text"])
            
            # Process documents (images or text)
            images = []
            doc_texts = []
            
            for doc in doc_strs:
                if doc["image"] is not None:
                    images.append(doc["image"])
                elif doc["text"] is not None:
                    doc_texts.append(doc["text"])
            
            # ColQwen2 expects queries (text) and images
            # Handle different scenarios
            if queries and images:
                # Text-to-Image retrieval (primary use case)
                batch_queries = self.processor.process_queries(queries).to(self.model.device)
                batch_images = self.processor.process_images(images).to(self.model.device)
                
                # Get embeddings
                query_embeddings = self.model(**batch_queries)
                image_embeddings = self.model(**batch_images)
                
                # Compute multi-vector scores
                scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
                
                # Convert to list
                # scores shape: (num_queries, num_images)
                scores = scores.squeeze(0).cpu().float().tolist()
                if not isinstance(scores, list):
                    scores = [scores]
                
            elif queries and doc_texts:
                # Text-to-Text: Not the primary use case for ColQwen2
                # We'll process documents as "pseudo-images" by encoding text
                # This is a fallback and may not perform optimally
                batch_queries = self.processor.process_queries(queries).to(self.model.device)
                batch_doc_queries = self.processor.process_queries(doc_texts).to(self.model.device)
                
                query_embeddings = self.model(**batch_queries)
                doc_embeddings = self.model(**batch_doc_queries)
                
                # Compute multi-vector scores
                scores = self.processor.score_multi_vector(query_embeddings, doc_embeddings)
                scores = scores.squeeze(0).cpu().float().tolist()
                if not isinstance(scores, list):
                    scores = [scores]
                    
            else:
                # Unsupported combination
                raise ValueError(
                    f"Unsupported query-document combination. "
                    f"ColQwen2 primarily supports text queries with image documents."
                )
        
        return scores
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for ColQwen2.
        
        ColQwen2 is primarily designed for text-to-image retrieval,
        but can also handle text-to-text as a fallback.
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        
        return {
            (text, image),  # Primary use case: text query -> image documents
            (text, text),   # Fallback: text query -> text documents
        }
    
    def to(self, device: str) -> "ColQwen2Reranker":
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda', 'mps', or 'cpu')
            
        Returns:
            Self for chaining
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

