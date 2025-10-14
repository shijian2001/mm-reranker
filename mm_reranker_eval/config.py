"""Configuration handling for mm_reranker_eval."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for reranker model."""
    name: str
    device: str = "cuda"
    num_gpus: Optional[int] = None
    use_flash_attention: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for evaluation data."""
    eval_data_path: str
    candidate_docs_path: str
    max_queries: Optional[int] = None


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    recall_k: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 10])
    ndcg_k: Optional[List[int]] = None


@dataclass
class EvalConfig:
    """Complete evaluation configuration."""
    model: ModelConfig
    data: DataConfig
    metrics: MetricsConfig
    output_dir: str
    rank_kwargs: Dict[str, Any] = field(default_factory=dict)
    save_per_query: bool = False
    
    @staticmethod
    def from_yaml(yaml_path: str) -> "EvalConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            EvalConfig object
            
        Example YAML:
            model:
              name: jinaai/jina-reranker-m0
              device: cuda
              num_gpus: 2
              use_flash_attention: true
              
            data:
              eval_data_path: data/eval.jsonl
              candidate_docs_path: data/candidates.jsonl
              max_queries: 100
              
            metrics:
              recall_k: [1, 3, 5, 7, 10]
              ndcg_k: [5, 10]
              
            output_dir: results/experiment_1
            rank_kwargs:
              max_length: 2048
            save_per_query: true
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return EvalConfig.from_dict(config_dict)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "EvalConfig":
        """
        Create EvalConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            EvalConfig object
        """
        model_cfg = config_dict.get("model", {})
        data_cfg = config_dict.get("data", {})
        metrics_cfg = config_dict.get("metrics", {})
        
        return EvalConfig(
            model=ModelConfig(
                name=model_cfg.get("name"),
                device=model_cfg.get("device", "cuda"),
                num_gpus=model_cfg.get("num_gpus"),
                use_flash_attention=model_cfg.get("use_flash_attention", True),
                kwargs=model_cfg.get("kwargs", {})
            ),
            data=DataConfig(
                eval_data_path=data_cfg.get("eval_data_path"),
                candidate_docs_path=data_cfg.get("candidate_docs_path"),
                max_queries=data_cfg.get("max_queries")
            ),
            metrics=MetricsConfig(
                recall_k=metrics_cfg.get("recall_k", [1, 3, 5, 7, 10]),
                ndcg_k=metrics_cfg.get("ndcg_k")
            ),
            output_dir=config_dict.get("output_dir", "results"),
            rank_kwargs=config_dict.get("rank_kwargs", {}),
            save_per_query=config_dict.get("save_per_query", False)
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        config_dict = {
            "model": {
                "name": self.model.name,
                "device": self.model.device,
                "num_gpus": self.model.num_gpus,
                "use_flash_attention": self.model.use_flash_attention,
                "kwargs": self.model.kwargs
            },
            "data": {
                "eval_data_path": self.data.eval_data_path,
                "candidate_docs_path": self.data.candidate_docs_path,
                "max_queries": self.data.max_queries
            },
            "metrics": {
                "recall_k": self.metrics.recall_k,
                "ndcg_k": self.metrics.ndcg_k
            },
            "output_dir": self.output_dir,
            "rank_kwargs": self.rank_kwargs,
            "save_per_query": self.save_per_query
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model.name:
            raise ValueError("model.name is required")
        
        if not self.data.eval_data_path:
            raise ValueError("data.eval_data_path is required")
        
        if not self.data.candidate_docs_path:
            raise ValueError("data.candidate_docs_path is required")
        
        if not Path(self.data.eval_data_path).exists():
            raise ValueError(f"Eval data file not found: {self.data.eval_data_path}")
        
        if not Path(self.data.candidate_docs_path).exists():
            raise ValueError(f"Candidate docs file not found: {self.data.candidate_docs_path}")

