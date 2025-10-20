"""BGE-VL-MLLM multimodal reranker implementation."""
# TODO: TypeError: tuple indices must be integers or slices, not tuple

from typing import List, Set
import torch
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, Modality


class BgeVlMllmReranker(BaseReranker):
    """
    BGE-VL-MLLM multimodal reranker (BAAI/BGE-VL-MLLM-S1).
    
    Supports text-to-image, image-to-text, image-to-image, text-to-text,
    and multimodal retrieval with optional task instructions.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/BGE-VL-MLLM-S1",
        device: str = "cuda",
        instruction: str = None,
        **kwargs
    ):
        """
        Initialize BGE-VL-MLLM reranker.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda' or 'cpu')
            instruction: Task instruction for retrieval
            **kwargs: Additional model arguments
        """
        super().__init__(model_name, device, **kwargs)
        self.instruction = instruction or ""
        self._load_model()

    def _load_model(self) -> None:
        """Load the BGE-VL-MLLM model."""
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model.to(self.device)
        self.model.eval()

        # Set processor
        with torch.no_grad():
            self.model.set_processor(self.model_name)
            # Workaround for transformers bug: set patch_size on processor directly
            self.model.processor.patch_size = 14
    
    def _format(self, item: Query | Document) -> dict:
        """
        Convert Query or Document to BGE-VL-MLLM format.
        
        Returns a dict with 'text' and 'images' keys.
        """
        result = {"text": None, "images": None}
        
        if item.text is not None:
            result["text"] = item.text
        
        if item.image is not None:
            result["images"] = item.image
        
        return result

    def _compute_scores(
        self,
        query_str: dict,
        doc_strs: List[dict],
        query_type: str,
        doc_type: str,
        instruction: str = None,
        **kwargs
    ) -> List[float]:
        """
        Compute scores using BGE-VL-MLLM embeddings.
        
        Args:
            query_str: Formatted query dict with 'text' and 'images'
            doc_strs: List of formatted document dicts
            query_type: Query type (not used by BGE-VL-MLLM)
            doc_type: Document type (not used by BGE-VL-MLLM)
            instruction: Task instruction for retrieval
            **kwargs: Additional arguments
        """
        task_instruction = instruction if instruction is not None else self.instruction
        
        with torch.no_grad():
            # Process query - only pass non-None values
            query_kwargs = {k: v for k, v in {
                "text": query_str["text"],
                "images": query_str["images"],
                "task_instruction": task_instruction
            }.items() if v is not None}
            query_inputs = self.model.data_process(q_or_c="q", **query_kwargs)
            
            # Process candidates
            has_text = any(d["text"] is not None for d in doc_strs)
            has_images = any(d["images"] is not None for d in doc_strs)
            
            candi_kwargs = {}
            if has_text and not has_images:
                candi_kwargs["text"] = [d["text"] for d in doc_strs]
            elif has_images and not has_text:
                candi_kwargs["images"] = [d["images"] for d in doc_strs]
            else:
                candi_kwargs["text"] = [d["text"] for d in doc_strs]
                candi_kwargs["images"] = [d["images"] for d in doc_strs]
            
            candi_inputs = self.model.data_process(q_or_c="c", **candi_kwargs)
            
            # Get embeddings
            query_embs = self.model(**query_inputs, output_hidden_states=True)[:, -1, :]
            candi_embs = self.model(**candi_inputs, output_hidden_states=True)[:, -1, :]
            
            # Normalize and compute scores
            query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
            candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)
            scores = torch.matmul(query_embs, candi_embs.T)
            
            # Convert to list
            scores = scores.squeeze(0).cpu().float().tolist()
            if not isinstance(scores, list):
                scores = [scores]
        
        return scores
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for BGE-VL-MLLM.
        
        BGE-VL-MLLM supports comprehensive multimodal retrieval:
        - text to text
        - text to image
        - image to text
        - image to image
        - multimodal (text+image) combinations
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        text_image = frozenset([Modality.TEXT, Modality.IMAGE])
        
        return {
            (text, text),
            (text, image),
            (image, text),
            (image, image),
            (text_image, text),
            (text_image, image),
            (text_image, text_image),
            (text, text_image),
            (image, text_image),
        }

