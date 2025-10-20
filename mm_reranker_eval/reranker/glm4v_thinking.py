"""GLM-4.1V-9B-Thinking multimodal reranker implementation.

This reranker supports three scoring modes:
1. logits_pyes: Extract P(yes) from Yes/No token logits (similar to MonoQwen2-VL)
2. generative_ranking: Generate a ranking sequence (e.g., "1, 3, 2")
3. listwise_scores: Generate a score list (e.g., "0.95, 0.72, 0.83")

The model supports text-to-image, image-to-text, text-to-text, and image-to-image retrieval.
"""

from typing import List, Set, Dict, Optional, Any
import torch
import re
from PIL import Image
from mm_reranker_eval.reranker.base import BaseReranker
from mm_reranker_eval.data.types import Query, Document, Modality


class GLM4VThinkingReranker(BaseReranker):
    """
    GLM-4.1V-9B-Thinking multimodal reranker.
    
    A versatile reranker that leverages the thinking capabilities of GLM-4.1V
    to rank documents with respect to queries across multiple modalities.
    
    Supported scoring modes:
        - logits_pyes: Extract P(Yes) from Yes/No token logits (fast, efficient)
        - generative_ranking: Generate ranking sequence (interpretable, flexible)
        - listwise_scores: Generate score list (fine-grained control)
    
    Requirements:
        - transformers >= 4.47.0
        - torch >= 2.0.0
        - PIL
    
    Supported modalities:
        - Text query → Text document
        - Text query → Image document  
        - Image query → Text document
        - Image query → Image document
        - Mixed (text+image) combinations
    """
    
    SCORING_MODES = ["logits_pyes", "generative_ranking", "listwise_scores"]
    
    def __init__(
        self,
        model_name: str = "THUDM/GLM-4.1V-9B-Thinking",
        device: str = "cuda",
        scoring_mode: str = "logits_pyes",
        batch_size: int = 1,
        max_new_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize GLM-4.1V-9B-Thinking reranker.
        
        Args:
            model_name: HuggingFace model name/path (default: THUDM/GLM-4.1V-9B-Thinking)
            device: Device to run on ('cuda' or 'cpu')
            scoring_mode: Scoring mode to use, one of:
                - "logits_pyes": Extract P(Yes) from Yes/No token logits
                - "generative_ranking": Generate ranking sequence
                - "listwise_scores": Generate list-wise scores
            batch_size: Batch size for processing (currently only 1 is supported)
            max_new_tokens: Maximum new tokens to generate (for generative modes)
            **kwargs: Additional model arguments (e.g., torch_dtype, attn_implementation)
        
        Raises:
            ValueError: If scoring_mode is not supported
        """
        super().__init__(model_name, device, **kwargs)
        
        if scoring_mode not in self.SCORING_MODES:
            raise ValueError(
                f"Unsupported scoring_mode: '{scoring_mode}'. "
                f"Must be one of {self.SCORING_MODES}"
            )
        
        self.scoring_mode = scoring_mode
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model_kwargs = kwargs
        
        if batch_size != 1:
            print(
                f"Warning: batch_size={batch_size} is not yet supported. "
                f"Using batch_size=1 for now."
            )
            self.batch_size = 1
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the GLM-4.1V-9B-Thinking model and processor."""
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=True
        )
        
        # Prepare model loading kwargs
        model_load_kwargs = {"device_map": self.device}
        
        # Add optional kwargs like torch_dtype, attn_implementation
        if "torch_dtype" in self.model_kwargs:
            model_load_kwargs["torch_dtype"] = self.model_kwargs["torch_dtype"]
        else:
            # Default to bfloat16 for efficiency
            model_load_kwargs["torch_dtype"] = torch.bfloat16
        
        if "attn_implementation" in self.model_kwargs:
            model_load_kwargs["attn_implementation"] = self.model_kwargs["attn_implementation"]
        
        # Load model
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_load_kwargs
        )
        self.model.eval()
        
        # Get Yes/No token IDs for logits_pyes mode
        if self.scoring_mode == "logits_pyes":
            # Try common token representations
            yes_candidates = ["Yes", "yes", "YES", "True", "true"]
            no_candidates = ["No", "no", "NO", "False", "false"]
            
            self.yes_token_id = None
            self.no_token_id = None
            
            for yes_token in yes_candidates:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(yes_token)
                if token_id != self.processor.tokenizer.unk_token_id:
                    self.yes_token_id = token_id
                    break
            
            for no_token in no_candidates:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(no_token)
                if token_id != self.processor.tokenizer.unk_token_id:
                    self.no_token_id = token_id
                    break
            
            if self.yes_token_id is None or self.no_token_id is None:
                print(
                    f"Warning: Could not find Yes/No tokens in vocabulary. "
                    f"Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}. "
                    f"Logits-based scoring may not work correctly."
                )
    
    def _format(self, item: Query | Document) -> dict:
        """
        Convert Query or Document to GLM-4.1V format.
        
        Returns a dict with 'text' and/or 'image' keys.
        
        Args:
            item: Query or Document object
            
        Returns:
            Dict with 'text' (str) and/or 'image' (PIL.Image or str) keys
        """
        result = {}
        
        if item.text is not None:
            result["text"] = item.text
        
        if item.image is not None:
            # Load image if it's a path
            if isinstance(item.image, str):
                try:
                    result["image"] = Image.open(item.image).convert("RGB")
                except Exception as e:
                    print(f"Warning: Failed to load image '{item.image}': {e}")
                    result["image"] = None
            else:
                result["image"] = item.image
        
        return result
    
    def _construct_prompt_logits_pyes(
        self,
        query_dict: dict,
        doc_dict: dict
    ) -> str:
        """
        Construct prompt for logits_pyes mode (Yes/No classification).
        
        Args:
            query_dict: Formatted query dict
            doc_dict: Formatted document dict
            
        Returns:
            Prompt string
        """
        # Determine query and document types
        has_query_text = "text" in query_dict
        has_query_image = "image" in query_dict and query_dict["image"] is not None
        has_doc_text = "text" in doc_dict
        has_doc_image = "image" in doc_dict and doc_dict["image"] is not None
        
        # Build prompt based on modality combinations
        if has_query_text and has_doc_image and not has_doc_text:
            # Text query → Image document
            prompt = (
                "Task: Determine if the given image is relevant to the query.\n\n"
                f"Query: {query_dict['text']}\n\n"
                "Question: Is the above image relevant to this query? "
                "Answer with Yes or No."
            )
        elif has_query_text and has_doc_text and not has_doc_image:
            # Text query → Text document
            prompt = (
                "Task: Determine if the given document is relevant to the query.\n\n"
                f"Query: {query_dict['text']}\n\n"
                f"Document: {doc_dict['text']}\n\n"
                "Question: Is this document relevant to the query? "
                "Answer with Yes or No."
            )
        elif has_query_image and has_doc_text and not has_doc_image:
            # Image query → Text document
            prompt = (
                "Task: Determine if the given document is relevant to the query image.\n\n"
                f"Document: {doc_dict['text']}\n\n"
                "Question: Is this document relevant to the query image above? "
                "Answer with Yes or No."
            )
        elif has_query_image and has_doc_image and not has_query_text and not has_doc_text:
            # Image query → Image document
            prompt = (
                "Task: Determine if the second image is relevant to the first image (query).\n\n"
                "Question: Are these two images related or similar? "
                "Answer with Yes or No."
            )
        else:
            # Mixed modality (text + image in query or document)
            query_parts = []
            if has_query_text:
                query_parts.append(f"Text: {query_dict['text']}")
            if has_query_image:
                query_parts.append("Image: [shown above]")
            
            doc_parts = []
            if has_doc_text:
                doc_parts.append(f"Text: {doc_dict['text']}")
            if has_doc_image:
                doc_parts.append("Image: [shown above]")
            
            prompt = (
                "Task: Determine if the document is relevant to the query.\n\n"
                f"Query: {' | '.join(query_parts)}\n\n"
                f"Document: {' | '.join(doc_parts)}\n\n"
                "Question: Is this document relevant to the query? "
                "Answer with Yes or No."
            )
        
        return prompt
    
    def _construct_prompt_generative_ranking(
        self,
        query_dict: dict,
        doc_dicts: List[dict]
    ) -> str:
        """
        Construct prompt for generative_ranking mode (ranking sequence generation).
        
        Args:
            query_dict: Formatted query dict
            doc_dicts: List of formatted document dicts
            
        Returns:
            Prompt string
        """
        num_docs = len(doc_dicts)
        
        # Build query description
        query_parts = []
        if "text" in query_dict:
            query_parts.append(f"Text: {query_dict['text']}")
        if "image" in query_dict and query_dict["image"] is not None:
            query_parts.append("Image: [shown above]")
        
        query_str = " | ".join(query_parts)
        
        # Build document descriptions
        doc_descriptions = []
        for i, doc_dict in enumerate(doc_dicts, 1):
            doc_parts = []
            if "text" in doc_dict:
                # Truncate very long text for prompt efficiency
                text = doc_dict["text"]
                if len(text) > 200:
                    text = text[:200] + "..."
                doc_parts.append(f"Text: {text}")
            if "image" in doc_dict and doc_dict["image"] is not None:
                doc_parts.append("Image: [shown]")
            
            doc_str = " | ".join(doc_parts)
            doc_descriptions.append(f"Document {i}: {doc_str}")
        
        docs_str = "\n".join(doc_descriptions)
        
        # Construct prompt
        prompt = (
            f"Task: Rank the following {num_docs} documents by their relevance to the query. "
            f"Output the ranking as a comma-separated list of document numbers, "
            f"from most relevant to least relevant.\n\n"
            f"Query: {query_str}\n\n"
            f"Documents:\n{docs_str}\n\n"
            f"Please think carefully and provide your ranking in the format: 1, 2, 3, ...\n"
            f"Ranking: "
        )
        
        return prompt
    
    def _construct_prompt_listwise_scores(
        self,
        query_dict: dict,
        doc_dicts: List[dict]
    ) -> str:
        """
        Construct prompt for listwise_scores mode (score list generation).
        
        Args:
            query_dict: Formatted query dict
            doc_dicts: List of formatted document dicts
            
        Returns:
            Prompt string
        """
        num_docs = len(doc_dicts)
        
        # Build query description
        query_parts = []
        if "text" in query_dict:
            query_parts.append(f"Text: {query_dict['text']}")
        if "image" in query_dict and query_dict["image"] is not None:
            query_parts.append("Image: [shown above]")
        
        query_str = " | ".join(query_parts)
        
        # Build document descriptions
        doc_descriptions = []
        for i, doc_dict in enumerate(doc_dicts, 1):
            doc_parts = []
            if "text" in doc_dict:
                # Truncate very long text for prompt efficiency
                text = doc_dict["text"]
                if len(text) > 200:
                    text = text[:200] + "..."
                doc_parts.append(f"Text: {text}")
            if "image" in doc_dict and doc_dict["image"] is not None:
                doc_parts.append("Image: [shown]")
            
            doc_str = " | ".join(doc_parts)
            doc_descriptions.append(f"Document {i}: {doc_str}")
        
        docs_str = "\n".join(doc_descriptions)
        
        # Construct prompt
        prompt = (
            f"Task: Score the following {num_docs} documents based on their relevance to the query. "
            f"Assign each document a relevance score between 0.0 (not relevant) and 1.0 (highly relevant). "
            f"Output the scores as a comma-separated list of numbers.\n\n"
            f"Query: {query_str}\n\n"
            f"Documents:\n{docs_str}\n\n"
            f"Please think carefully and provide your scores in the format: 0.95, 0.72, 0.83, ...\n"
            f"Scores: "
        )
        
        return prompt
    
    def _prepare_messages(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None
    ) -> List[dict]:
        """
        Prepare messages in GLM-4.1V format.
        
        Args:
            prompt: Text prompt
            images: Optional list of PIL Images
            
        Returns:
            List of message dicts
        """
        content = []
        
        # Add images first (if any)
        if images:
            for image in images:
                if image is not None:
                    content.append({
                        "type": "image",
                        "image": image
                    })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def _compute_scores_logits_pyes(
        self,
        query_str: dict,
        doc_strs: List[dict],
        **kwargs
    ) -> List[float]:
        """
        Compute scores using logits_pyes mode (P(Yes) extraction).
        
        Args:
            query_str: Formatted query dict
            doc_strs: List of formatted document dicts
            **kwargs: Additional arguments
            
        Returns:
            List of relevance scores
        """
        scores = []
        
        for doc_str in doc_strs:
            # Construct prompt
            prompt = self._construct_prompt_logits_pyes(query_str, doc_str)
            
            # Collect all images for this query-document pair
            images = []
            if "image" in query_str and query_str["image"] is not None:
                images.append(query_str["image"])
            if "image" in doc_str and doc_str["image"] is not None:
                images.append(doc_str["image"])
            
            # Prepare messages
            messages = self._prepare_messages(prompt, images if images else None)
            
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_for_last_token = outputs.logits[:, -1, :]
            
            # Extract Yes/No probabilities
            if self.yes_token_id is not None and self.no_token_id is not None:
                relevance_logits = logits_for_last_token[:, [self.yes_token_id, self.no_token_id]]
                relevance_probs = torch.softmax(relevance_logits, dim=-1)
                yes_prob = relevance_probs[0, 0].item()
                scores.append(yes_prob)
            else:
                # Fallback: use generation to get Yes/No
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
                output_text = self.processor.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Parse Yes/No from generated text
                output_text_lower = output_text.lower().strip()
                if "yes" in output_text_lower:
                    scores.append(0.9)
                elif "no" in output_text_lower:
                    scores.append(0.1)
                else:
                    scores.append(0.5)  # Neutral if unclear
        
        return scores
    
    def _compute_scores_generative_ranking(
        self,
        query_str: dict,
        doc_strs: List[dict],
        **kwargs
    ) -> List[float]:
        """
        Compute scores using generative_ranking mode (ranking sequence).
        
        Args:
            query_str: Formatted query dict
            doc_strs: List of formatted document dicts
            **kwargs: Additional arguments
            
        Returns:
            List of relevance scores
        """
        num_docs = len(doc_strs)
        
        # Construct prompt
        prompt = self._construct_prompt_generative_ranking(query_str, doc_strs)
        
        # For now, we only support text documents in generative ranking due to complexity
        # Collect query images (if any)
        images = []
        if "image" in query_str and query_str["image"] is not None:
            images.append(query_str["image"])
        
        # Note: For simplicity, we don't include all document images in this mode
        # as it would make the context too long. This mode works best with text documents.
        
        # Prepare messages
        messages = self._prepare_messages(prompt, images if images else None)
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate ranking
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse ranking from output
        # Expected format: "1, 3, 2, ..." or similar
        scores = self._parse_ranking_sequence(output_text, num_docs)
        
        return scores
    
    def _compute_scores_listwise_scores(
        self,
        query_str: dict,
        doc_strs: List[dict],
        **kwargs
    ) -> List[float]:
        """
        Compute scores using listwise_scores mode (score list).
        
        Args:
            query_str: Formatted query dict
            doc_strs: List of formatted document dicts
            **kwargs: Additional arguments
            
        Returns:
            List of relevance scores
        """
        num_docs = len(doc_strs)
        
        # Construct prompt
        prompt = self._construct_prompt_listwise_scores(query_str, doc_strs)
        
        # Collect query images (if any)
        images = []
        if "image" in query_str and query_str["image"] is not None:
            images.append(query_str["image"])
        
        # Prepare messages
        messages = self._prepare_messages(prompt, images if images else None)
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate scores
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse scores from output
        # Expected format: "0.95, 0.72, 0.83, ..."
        scores = self._parse_score_list(output_text, num_docs)
        
        return scores
    
    def _parse_ranking_sequence(self, output_text: str, num_docs: int) -> List[float]:
        """
        Parse ranking sequence from generated text and convert to scores.
        
        Args:
            output_text: Generated text containing ranking
            num_docs: Number of documents
            
        Returns:
            List of scores (higher score = more relevant)
        """
        # Extract numbers from output
        # Look for patterns like "1, 3, 2" or "1 3 2" or "1. 3. 2."
        numbers = re.findall(r'\d+', output_text)
        
        if not numbers:
            # Fallback: uniform scores
            return [1.0 / (i + 1) for i in range(num_docs)]
        
        # Convert to integers and validate
        try:
            ranking = [int(n) for n in numbers[:num_docs]]
        except ValueError:
            # Fallback
            return [1.0 / (i + 1) for i in range(num_docs)]
        
        # Convert ranking to scores
        # Document with rank 1 gets highest score, rank N gets lowest score
        scores = [0.0] * num_docs
        for idx, rank in enumerate(ranking):
            if 1 <= rank <= num_docs:
                # Score inversely proportional to rank
                scores[rank - 1] = 1.0 - (idx / num_docs)
        
        # If some documents weren't ranked, assign them low scores
        for i in range(num_docs):
            if scores[i] == 0.0:
                scores[i] = 0.01
        
        return scores
    
    def _parse_score_list(self, output_text: str, num_docs: int) -> List[float]:
        """
        Parse score list from generated text.
        
        Args:
            output_text: Generated text containing scores
            num_docs: Number of documents
            
        Returns:
            List of scores
        """
        # Extract numbers (integers and floats) from output
        # Look for patterns like "0.95, 0.72, 0.83" or "0.95 0.72 0.83"
        numbers = re.findall(r'\d+\.?\d*', output_text)
        
        if not numbers:
            # Fallback: uniform scores
            return [0.5] * num_docs
        
        # Convert to floats and validate
        try:
            scores = [float(n) for n in numbers[:num_docs]]
            
            # Normalize to [0, 1] range if needed
            for i in range(len(scores)):
                if scores[i] > 1.0:
                    scores[i] = scores[i] / 100.0  # Assume percentage
                scores[i] = max(0.0, min(1.0, scores[i]))
            
        except ValueError:
            # Fallback
            scores = [0.5] * num_docs
        
        # Pad with low scores if we didn't get enough
        while len(scores) < num_docs:
            scores.append(0.1)
        
        return scores[:num_docs]
    
    def _compute_scores(
        self,
        query_str: dict,
        doc_strs: List[dict],
        query_type: str,
        doc_type: str,
        **kwargs
    ) -> List[float]:
        """
        Compute relevance scores using the configured scoring mode.
        
        Args:
            query_str: Formatted query dict
            doc_strs: List of formatted document dicts
            query_type: Query type ('text', 'image', 'auto')
            doc_type: Document type ('text', 'image', 'auto')
            **kwargs: Additional arguments
            
        Returns:
            List of relevance scores
        """
        # Warn about unused parameters
        if "instruction" in kwargs:
            self._warn_unused_param("instruction", kwargs["instruction"])
        if "max_length" in kwargs:
            self._warn_unused_param("max_length", kwargs["max_length"])
        
        # Delegate to appropriate scoring method
        if self.scoring_mode == "logits_pyes":
            return self._compute_scores_logits_pyes(query_str, doc_strs, **kwargs)
        elif self.scoring_mode == "generative_ranking":
            return self._compute_scores_generative_ranking(query_str, doc_strs, **kwargs)
        elif self.scoring_mode == "listwise_scores":
            return self._compute_scores_listwise_scores(query_str, doc_strs, **kwargs)
        else:
            raise ValueError(f"Unsupported scoring_mode: {self.scoring_mode}")
    
    def supported_modalities(self) -> Set[tuple]:
        """
        Get supported modality combinations for GLM-4.1V-9B-Thinking.
        
        GLM-4.1V-9B-Thinking supports comprehensive multimodal retrieval:
        - text to text
        - text to image
        - image to text
        - image to image
        - multimodal (text+image) combinations
        
        Returns:
            Set of (query_modalities, doc_modalities) tuples
        """
        text = frozenset([Modality.TEXT])
        image = frozenset([Modality.IMAGE])
        text_image = frozenset([Modality.TEXT, Modality.IMAGE])
        
        return {
            # Text query
            (text, text),
            (text, image),
            (text, text_image),
            
            # Image query
            (image, text),
            (image, image),
            (image, text_image),
            
            # Multimodal query
            (text_image, text),
            (text_image, image),
            (text_image, text_image),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device='{self.device}', "
            f"scoring_mode='{self.scoring_mode}')"
        )

