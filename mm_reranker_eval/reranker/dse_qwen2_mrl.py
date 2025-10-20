"""DSE Qwen2 MRL multimodal reranker implementation."""

from typing import List, Set, Union
import torch
import numpy as np
from PIL import Image
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, Modality


class DseQwen2Mrl(BaseReranker):
    """
    DSE Qwen2 MRL multimodal reranker (dse-qwen2-2b-mrl-v1).
    
    Uses Qwen2VL for multimodal embedding-based retrieval.
    Supports text-to-image, text-to-text, image-to-text, and image-to-image retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "MrLight/dse-qwen2-2b-mrl-v1",
        device: str = "cuda",
        use_flash_attention: bool = True,
        min_pixels: int = 1 * 28 * 28,
        max_pixels: int = 2560 * 28 * 28,
        embedding_dim: int = 1536,
        instruction: str = None,
        **kwargs
    ):
        """
        Initialize DSE Qwen2 MRL reranker.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda' or 'cpu')
            use_flash_attention: Whether to use flash attention 2
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
            embedding_dim: Embedding dimension (e.g., 512, 1536)
            instruction: Task instruction (not used by DSE Qwen2 MRL)
            **kwargs: Additional model arguments
        """
        super().__init__(model_name, device, **kwargs)
        self.use_flash_attention = use_flash_attention and device == "cuda"
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.embedding_dim = embedding_dim
        if instruction is not None:
            self._warn_unused_param("instruction", instruction)
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the DSE Qwen2 MRL model and processor."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.processor.tokenizer.padding_side = "left"
        
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            attn_implementation=attn_impl, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
        ).eval()
        self.model.padding_side = "left"
    
    def _get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Extract and normalize embeddings from hidden states."""
        reps = last_hidden_state[:, -1, :self.embedding_dim]
        return torch.nn.functional.normalize(reps, p=2, dim=-1)
    
    def _load_image(self, image_input: str) -> Image.Image:
        """Load image from path or URL."""
        if image_input.startswith(('http://', 'https://')):
            import requests
            from io import BytesIO
            return Image.open(BytesIO(requests.get(image_input).content))
        return Image.open(image_input)
    
    def _encode(self, contents: List[str], is_query: bool, is_image: bool) -> torch.Tensor:
        """
        Unified encoding function for both queries and documents.
        
        Args:
            contents: List of text strings or image paths
            is_query: Whether encoding queries (vs documents)
            is_image: Whether content is images (vs text)
        """
        from qwen_vl_utils import process_vision_info
        
        messages = []
        for content in contents:
            if is_image:
                img = self._load_image(content)
                text = 'Query: What is shown in this image?' if is_query else 'What is shown in this image?'
                message_content = [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': text}
                ]
            else:
                # Text: use dummy 1x1 image
                prefix = 'Query: ' if is_query else 'Document: '
                message_content = [
                    {'type': 'image', 'image': Image.new('RGB', (28, 28)), 
                     'resized_height': 1, 'resized_width': 1},
                    {'type': 'text', 'text': f'{prefix}{content}'}
                ]
            messages.append([{'role': 'user', 'content': message_content}])
        
        # Process messages
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding='longest', return_tensors='pt'
        ).to(self.device)
        
        cache_position = torch.arange(0, len(texts))
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False
        )
        
        with torch.no_grad():
            output = self.model(**inputs, return_dict=True, output_hidden_states=True)
        
        return self._get_embedding(output.hidden_states[-1])
    
    def _format(self, item: Query | Document) -> str:
        """Convert Query or Document to content string."""
        modalities = item.get_modalities()
        if Modality.TEXT in modalities and Modality.IMAGE not in modalities:
            return item.text
        if Modality.IMAGE in modalities:
            return item.image
        if item.text:
            return item.text
        raise ValueError(f"Item has no valid text or image content: {item}")
    
    def _compute_scores(
        self, query_str: str, doc_strs: List[str],
        query_type: str, doc_type: str, instruction: str = None, **kwargs
    ) -> List[float]:
        """Compute scores using embeddings and cosine similarity."""
        if instruction is not None:
            self._warn_unused_param("instruction", instruction)
        # Determine if inputs are images
        def is_image_input(s: str) -> bool:
            return s.startswith(('http://', 'https://')) or s.endswith(
                ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
            )
        
        query_is_image = query_type == "image" or (query_type == "auto" and is_image_input(query_str))
        doc_is_image = doc_type == "image" or (doc_type == "auto" and doc_strs and is_image_input(doc_strs[0]))
        
        # Encode query and documents
        query_embedding = self._encode([query_str], is_query=True, is_image=query_is_image)
        doc_embeddings = self._encode(doc_strs, is_query=False, is_image=doc_is_image)
        
        # Compute cosine similarity
        num_docs = doc_embeddings.size(0)
        query_expanded = query_embedding.expand(num_docs, -1)
        similarities = torch.nn.functional.cosine_similarity(query_expanded, doc_embeddings)
        
        return similarities.cpu().float().tolist()
    
    def supported_modalities(self) -> Set[tuple]:
        """Get supported modality combinations."""
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        return {(text, text), (text, image), (image, text), (image, image)}
    
    def to(self, device: str) -> "DseQwen2Mrl":
        """
        Move model to specified device.
        
        Note: This model uses device_map during loading, so it's already on the device.
        This method only updates the device attribute.
        
        Args:
            device: Target device ('cuda' or 'cpu')
            
        Returns:
            Self for chaining
        """
        self.device = device
        # Model is already on device via device_map, no need to move
        return self
