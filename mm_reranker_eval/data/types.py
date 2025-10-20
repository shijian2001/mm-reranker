"""Core data types for multimodal reranker evaluation."""

import os
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
from enum import Enum


class Modality(str, Enum):
    """Supported modalities."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    # Future extensions can add more modalities


@dataclass
class Query:
    """
    Represents a query which can be text, image, video, or mixed.
    
    For single modality:
        Query(text="some text")
        Query(image="path/to/image.jpg")
        
    For mixed modality:
        Query(text="some text", image="path/to/image.jpg")
    """
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None
    
    def get_modalities(self) -> List[Modality]:
        """Get list of modalities present in this query."""
        modalities = []
        if self.text is not None:
            modalities.append(Modality.TEXT)
        if self.image is not None:
            modalities.append(Modality.IMAGE)
        if self.video is not None:
            modalities.append(Modality.VIDEO)
        return modalities
    
    def is_empty(self) -> bool:
        """Check if query has no content."""
        return self.text is None and self.image is None and self.video is None
    
    @staticmethod
    def from_raw(data: Union[str, Dict[str, str]], base_dir: Optional[str] = None) -> "Query":
        """
        Create Query from raw data format.
        
        Args:
            data: Either a string (single modality) or dict with modality keys
            base_dir: Base directory to prepend to image/video paths
            
        Returns:
            Query object
        """
        def _resolve_path(path: str) -> str:
            """Resolve relative paths with base directory."""
            if base_dir and path and not os.path.isabs(path):
                return os.path.join(base_dir, path)
            return path
        
        if isinstance(data, str):
            # Try to infer modality from extension or assume text
            if data.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')):
                return Query(image=_resolve_path(data))
            elif data.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return Query(video=_resolve_path(data))
            else:
                return Query(text=data)
        elif isinstance(data, dict):
            return Query(
                text=data.get("text"),
                image=_resolve_path(data.get("image")) if data.get("image") else None,
                video=_resolve_path(data.get("video")) if data.get("video") else None
            )
        else:
            raise ValueError(f"Unsupported query data type: {type(data)}")


@dataclass
class Document:
    """
    Represents a document which can be text, image, video, or mixed.
    
    Similar to Query, supports single or mixed modality content.
    """
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None
    
    def get_modalities(self) -> List[Modality]:
        """Get list of modalities present in this document."""
        modalities = []
        if self.text is not None:
            modalities.append(Modality.TEXT)
        if self.image is not None:
            modalities.append(Modality.IMAGE)
        if self.video is not None:
            modalities.append(Modality.VIDEO)
        return modalities
    
    def is_empty(self) -> bool:
        """Check if document has no content."""
        return self.text is None and self.image is None and self.video is None
    
    @staticmethod
    def from_raw(data: Union[str, Dict[str, str]], base_dir: Optional[str] = None) -> "Document":
        """
        Create Document from raw data format.
        
        Args:
            data: Either a string (single modality) or dict with modality keys
            base_dir: Base directory to prepend to image/video paths
            
        Returns:
            Document object
        """
        def _resolve_path(path: str) -> str:
            """Resolve relative paths with base directory."""
            if base_dir and path and not os.path.isabs(path):
                return os.path.join(base_dir, path)
            return path
        
        if isinstance(data, str):
            # Try to infer modality from extension or assume text
            if data.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')):
                return Document(image=_resolve_path(data))
            elif data.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return Document(video=_resolve_path(data))
            else:
                return Document(text=data)
        elif isinstance(data, dict):
            return Document(
                text=data.get("text"),
                image=_resolve_path(data.get("image")) if data.get("image") else None,
                video=_resolve_path(data.get("video")) if data.get("video") else None
            )
        else:
            raise ValueError(f"Unsupported document data type: {type(data)}")


@dataclass
class EvalSample:
    """
    Evaluation sample containing a query and its matching document.
    
    Used for loading evaluation datasets.
    """
    query: Query
    match: Document
    dataset: str
    task_id: Optional[int] = None
    dataset_id: Optional[int] = None
    id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def from_json(data: Dict[str, Any], base_dir: Optional[str] = None) -> "EvalSample":
        """
        Load evaluation sample from JSON format.
        
        Args:
            data: JSON dict containing query, match, and metadata
            base_dir: Base directory to prepend to image/video paths
            
        Returns:
            EvalSample object
        """
        return EvalSample(
            query=Query.from_raw(data["query"], base_dir=base_dir),
            match=Document.from_raw(data["match"], base_dir=base_dir),
            dataset=data.get("dataset", "unknown"),
            task_id=data.get("task_id"),
            dataset_id=data.get("dataset_id"),
            id=data.get("id"),
            metadata={
                k: v for k, v in data.items()
                if k not in ["query", "match", "dataset", "task_id", "dataset_id", "id"]
            }
        )


@dataclass
class RankResult:
    """
    Result from ranking operation.
    
    Contains ranked document indices and their scores.
    """
    ranked_indices: List[int]
    scores: Optional[List[float]] = None
    
    def __len__(self) -> int:
        """Get number of ranked documents."""
        return len(self.ranked_indices)

