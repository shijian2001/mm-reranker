"""MonoQwen2-VL multimodal reranker implementation.

Note: MonoQwen2-VL requires the 'peft' library to be installed.
The model is based on Qwen2-VL-2B-Instruct with LoRA adapters.
"""

from typing import List, Set
import torch
from PIL import Image
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, Modality


class MonoQwen2VL(BaseReranker):
    """
    MonoQwen2-VL multimodal reranker (lightonai/MonoQwen2-VL-v0.1).
    
    A pointwise reranker that uses Qwen2-VL-2B-Instruct to determine relevance
    between queries and documents by generating True/False answers.
    
    Requirements:
        - peft >= 0.7.0 (for LoRA adapter support)
        - transformers >= 4.47.3
    
    Supported modalities:
        - Text query → Image document (primary use case)
        - Text query → Text document
        - Text query → Mixed (text+image) document
    """
    
    def __init__(
        self,
        model_name: str = "lightonai/MonoQwen2-VL-v0.1",
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize MonoQwen2-VL reranker.
        
        Args:
            model_name: HuggingFace model name/path (default: lightonai/MonoQwen2-VL-v0.1)
            device: Device to run on ('cuda' or 'cpu')
            **kwargs: Additional model arguments (e.g., torch_dtype, attn_implementation)
        """
        super().__init__(model_name, device, **kwargs)
        self.model_kwargs = kwargs
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the MonoQwen2-VL model and processor."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        # Load processor (always from Qwen2-VL-2B-Instruct)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        
        # Prepare model loading kwargs
        model_load_kwargs = {"device_map": self.device}
        
        # Add optional kwargs like torch_dtype, attn_implementation
        if "torch_dtype" in self.model_kwargs:
            model_load_kwargs["torch_dtype"] = self.model_kwargs["torch_dtype"]
        if "attn_implementation" in self.model_kwargs:
            model_load_kwargs["attn_implementation"] = self.model_kwargs["attn_implementation"]
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_load_kwargs
        )
        
        # Get True/False token IDs
        self.true_token_id = self.processor.tokenizer.convert_tokens_to_ids("True")
        self.false_token_id = self.processor.tokenizer.convert_tokens_to_ids("False")
    
    def _format(self, item: Query | Document) -> dict:
        """
        Convert Query or Document to MonoQwen2-VL format.
        
        Returns a dict with 'text' and/or 'image' keys.
        """
        result = {}
        
        if item.text is not None:
            result["text"] = item.text
        
        if item.image is not None:
            # Ensure image is a PIL Image
            if isinstance(item.image, str):
                result["image"] = Image.open(item.image)
            else:
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
        Compute relevance scores using MonoQwen2-VL.
        
        Args:
            query_str: Formatted query dict with 'text' and/or 'image'
            doc_strs: List of formatted document dicts
            query_type: Query type ('text', 'image', 'auto')
            doc_type: Document type ('text', 'image', 'auto')
            **kwargs: Additional arguments
        
        Returns:
            List of relevance scores (probability of True)
        """
        scores = []
        
        # Extract query text
        query_text = query_str.get("text", "")
        
        for doc_str in doc_strs:
            # Prepare messages based on document type
            content = []
            
            # Determine document type for prompt
            has_image = "image" in doc_str
            has_text = "text" in doc_str
            
            # Add document content
            if has_image:
                content.append({"type": "image", "image": doc_str["image"]})
            
            # Construct appropriate prompt based on document type
            if has_image and not has_text:
                # Pure image document
                prompt = (
                    "Assert the relevance of the previous image document to the following query, "
                    "answer True or False. The query is: {query}"
                ).format(query=query_text)
            elif has_text and not has_image:
                # Pure text document
                doc_text = doc_str["text"]
                prompt = (
                    "Assert the relevance of the following document to the query, "
                    "answer True or False.\n\n"
                    "Document: {doc}\n\n"
                    "Query: {query}"
                ).format(doc=doc_text, query=query_text)
            else:
                # Mixed document (text + image)
                doc_text = doc_str["text"]
                prompt = (
                    "Assert the relevance of the previous document (image and text) to the following query, "
                    "answer True or False.\n\n"
                    "Document text: {doc}\n\n"
                    "Query: {query}"
                ).format(doc=doc_text, query=query_text)
            
            # Add prompt text
            content.append({"type": "text", "text": prompt})
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            # Apply chat template and tokenize
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs
            image = doc_str.get("image", None)
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_for_last_token = outputs.logits[:, -1, :]
            
            # Calculate relevance score
            relevance_logits = logits_for_last_token[:, [self.true_token_id, self.false_token_id]]
            relevance_score = torch.softmax(relevance_logits, dim=-1)
            
            # Extract True probability
            true_prob = relevance_score[0, 0].item()
            scores.append(true_prob)
        
        return scores
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for MonoQwen2-VL.
        
        MonoQwen2-VL supports:
        - text query to image document
        - text query to text document
        - text query to multimodal (text+image) document
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        text_image = frozenset([Modality.TEXT, Modality.IMAGE])
        
        return {
            (text, text),
            (text, image),
            (text, text_image),
        }

